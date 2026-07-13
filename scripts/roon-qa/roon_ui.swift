import Cocoa
import ApplicationServices
import ImageIO
import Vision

enum QAError: Error, CustomStringConvertible {
    case message(String)
    var description: String {
        if case let .message(value) = self { return value }
        return "unknown error"
    }
}

func attribute(_ element: AXUIElement, _ name: CFString) -> AnyObject? {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, name, &value) == .success else { return nil }
    return value
}

func elementText(_ element: AXUIElement) -> [String] {
    [kAXTitleAttribute, kAXDescriptionAttribute, kAXValueAttribute]
        .compactMap { attribute(element, $0 as CFString) as? String }
}

func descendants(_ root: AXUIElement, limit: Int = 20_000) -> [AXUIElement] {
    var result: [AXUIElement] = []
    var queue = [root]
    while !queue.isEmpty && result.count < limit {
        let item = queue.removeFirst()
        result.append(item)
        if let children = attribute(item, kAXChildrenAttribute as CFString) as? [AXUIElement] {
            queue.append(contentsOf: children)
        }
    }
    return result
}

func runningRoon() throws -> NSRunningApplication {
    let matches = NSRunningApplication.runningApplications(withBundleIdentifier: "com.roon.Roon")
    guard matches.count == 1, let app = matches.first else {
        throw QAError.message("expected exactly one running /Applications/Roon.app")
    }
    return app
}

func exactMatches(_ elements: [AXUIElement], _ value: String) -> [AXUIElement] {
    elements.filter { elementText($0).contains(value) }
}

func requireZone(_ elements: [AXUIElement], zone: String) throws {
    guard exactMatches(elements, zone).count == 1 else {
        throw QAError.message("refusing UI action: exact QA zone '\(zone)' is not uniquely visible")
    }
}

func ocrRoonWindow(pid: pid_t) throws -> [String] {
    let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
    let info = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] ?? []
    let windows = info.filter { ($0[kCGWindowOwnerPID as String] as? pid_t) == pid }
    guard windows.count >= 1,
          let windowID = windows[0][kCGWindowNumber as String] as? CGWindowID else {
        throw QAError.message("Roon has no visible window for OCR")
    }
    let qaHome = ProcessInfo.processInfo.environment["ROOMEQ_ROON_QA_HOME"]
        ?? NSHomeDirectory() + "/Library/Application Support/SotF/RoonExportQA"
    let path = qaHome + "/tmp/roon-ocr-\(getpid()).png"
    defer { try? FileManager.default.removeItem(atPath: path) }

    let capture = Process()
    capture.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
    capture.arguments = ["-x", "-l", String(windowID), path]
    try capture.run()
    capture.waitUntilExit()
    guard capture.terminationStatus == 0,
          let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw QAError.message("failed to capture private Roon window for OCR")
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    try VNImageRequestHandler(cgImage: image).perform([request])
    return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
}

func main() throws {
    guard CommandLine.arguments.count >= 3 else {
        throw QAError.message("usage: roon-ui <assert-zone|click|readback> <zone> [label]")
    }
    let trusted = AXIsProcessTrustedWithOptions(
        [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
    )
    guard trusted else { throw QAError.message("Accessibility permission is required") }

    let command = CommandLine.arguments[1]
    let zone = CommandLine.arguments[2]
    let app = try runningRoon()
    app.activate(options: [.activateAllWindows])
    let root = AXUIElementCreateApplication(app.processIdentifier)
    let elements = descendants(root)
    try requireZone(elements, zone: zone)

    switch command {
    case "assert-zone":
        print("{\"qa_zone\":\"\(zone)\",\"visible\":true}")
    case "click":
        guard CommandLine.arguments.count == 4 else {
            throw QAError.message("click requires one exact accessibility label")
        }
        let label = CommandLine.arguments[3]
        let matches = exactMatches(elements, label)
        guard matches.count == 1 else {
            throw QAError.message("refusing click: exact label '\(label)' has \(matches.count) matches")
        }
        guard AXUIElementPerformAction(matches[0], kAXPressAction as CFString) == .success else {
            throw QAError.message("Roon element '\(label)' does not support AXPress")
        }
    case "readback":
        let axValues = elements.flatMap(elementText).filter { !$0.isEmpty }.sorted()
        let ocrValues = try ocrRoonWindow(pid: app.processIdentifier).sorted()
        let data = try JSONSerialization.data(withJSONObject: [
            "qa_zone": zone,
            "accessibility_text": axValues,
            "ocr_text": ocrValues,
        ])
        print(String(decoding: data, as: UTF8.self))
    default:
        throw QAError.message("unsupported command '\(command)'")
    }
}

do { try main() } catch {
    fputs("roon-ui: \(error)\n", stderr)
    exit(2)
}
