import json
import re
import sys

def norm_log(path: str) -> list[str]:
    lines = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = re.sub(r"^\[[^\]]+\]\s*", "", line.rstrip("\n"))
        line = re.sub(r"/Volumes/home_worktrees/autoeq-[a-zA-Z0-9-]+", "WT", line)
        lines.append(line)
    return lines

a = norm_log("/Volumes/home_tmp/tmp/hc-refactor-serial.log")
b = norm_log("/Volumes/home_tmp/tmp/hc-base-serial.log")
print(f"log lines: refactor={len(a)} base={len(b)} identical={a == b}")
if a != b:
    import difflib
    diff = list(difflib.unified_diff(a, b, lineterm="", n=1))
    print("\n".join(diff[:40]))

VOLATILE = {"duration_ms", "elapsed_ms", "wall_time_ms", "temp_dir", "output_dir"}

def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [strip(v) for v in o]
    if isinstance(o, str):
        return re.sub(r"/Volumes/home_worktrees/autoeq-[a-zA-Z0-9-]+", "WT", o)
    return o

ja = strip(json.load(open("/Volumes/home_tmp/tmp/hc-refactor-serial.json")))
jb = strip(json.load(open("/Volumes/home_tmp/tmp/hc-base-serial.json")))
print("verdict JSON identical:", ja == jb)
if ja != jb:
    import difflib
    da = json.dumps(ja, indent=1, sort_keys=True).splitlines()
    db = json.dumps(jb, indent=1, sort_keys=True).splitlines()
    print("\n".join(list(difflib.unified_diff(da, db, lineterm="", n=1))[:40]))
    sys.exit(1)
