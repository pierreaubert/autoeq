# ----------------------------------------------------------------------
# BASIC DOCKER BUILDS (Linux + Windows)
# ----------------------------------------------------------------------
# - Linux:
#   - build: just docker-linux-<arch>
#   - test: just test-linux-<arch>
#   - qa: just qa-linux-<arch>
#
# - Windows:
#   - build: just docker-windows-<arch>
#   - test: just test-windows-<arch>
#   - qa: just qa-windows-<arch>
#
# Usage:
#   just docker-linux-arm64
#   just docker-windows-arm64
#   just docker-cross-arm
#   just docker-linux-x86
#   just docker-windows-x86
#   just docker-cross
#   just docker-cross-native
#   just test-linux-arm64
#   just qa-linux-arm64

[group('docker')]
docker-base-linux-x86:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building base Linux x86_64 Docker image..."
    docker build --platform linux/amd64 \
      -t autoeq-base-linux-x86 \
      -f ./builds/from_linux_x86/Dockerfile.base.x86_64-unknown-linux-gnu \
      .

[group('docker')]
docker-base-win-x86:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building base Windows x86_64 Docker image..."
    docker build --platform linux/amd64 \
      -t autoeq-base-win-x86 \
      -f ./builds/from_macos_arm/Dockerfile.base.x86_64-pc-windows-gnullvm \
      .

[group('docker')]
docker-base-linux-arm64:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building base Linux ARM64 Docker image..."
    docker build --platform linux/arm64 \
      -t autoeq-base-linux-arm64 \
      -f ./builds/from_macos_arm/Dockerfile.base.aarch64-unknown-linux-gnu \
      .

[group('docker')]
docker-base-win-arm64:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building base Windows ARM64 Docker image..."
    docker build --platform linux/arm64 \
      -t autoeq-base-win-arm64 \
      -f ./builds/from_macos_arm/Dockerfile.base.aarch64-pc-windows-gnullvm \
      .

[group('docker')]
docker-linux-x86: docker-base-linux-x86
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building Linux x86_64 image..."
    docker build -t autoeq-linux-x86 \
      -f ./builds/from_linux_x86/Dockerfile.x86_64-unknown-linux-gnu \
      .
    docker run --rm \
      --platform linux/amd64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-linux-x86 \
      bash -lc "cargo build --release --target x86_64-unknown-linux-gnu --lib --bins"
    mkdir -p dist/linux
    find target/x86_64-unknown-linux-gnu/release -maxdepth 1 -type f -executable -print0 \
      | xargs -0 -I {} cp -f "{}" dist/linux/

[group('docker')]
docker-linux-arm64: docker-base-linux-arm64
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building Linux ARM64 image..."
    docker build -t autoeq-linux-arm64 \
      -f ./builds/from_linux_x86/Dockerfile.aarch64-unknown-linux-gnu \
      .
    docker run --rm \
      --platform linux/arm64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-linux-arm64 \
      bash -lc "cargo build --release --target aarch64-unknown-linux-gnu --lib --bins"
    mkdir -p dist/linux
    find target/aarch64-unknown-linux-gnu/release -maxdepth 1 -type f -executable -print0 \
      | xargs -0 -I {} cp -f "{}" dist/linux/

[group('docker')]
docker-windows-x86: docker-base-win-x86
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building Windows x86_64 image..."
    docker run --rm \
      --platform linux/amd64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-base-win-x86 \
      bash -lc "cargo build --release --target x86_64-pc-windows-gnullvm --lib --bins"
    mkdir -p dist/windows
    find target/x86_64-pc-windows-gnullvm/release -maxdepth 1 -type f -name '*.exe' -print0 \
      | xargs -0 -I {} cp -f "{}" dist/windows/

[group('docker')]
docker-windows-arm64: docker-base-win-arm64
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building Windows ARM64 image..."
    docker run --rm \
      --platform linux/arm64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-base-win-arm64 \
      bash -lc "cargo build --release --target aarch64-pc-windows-gnullvm --lib --bins"
    mkdir -p dist/windows
    find target/aarch64-pc-windows-gnullvm/release -maxdepth 1 -type f -name '*.exe' -print0 \
      | xargs -0 -I {} cp -f "{}" dist/windows/

[group('docker')]
test-linux-x86: docker-base-linux-x86
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running tests in Linux x86_64 Docker image..."
    docker run --rm \
      --platform linux/amd64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-base-linux-x86 \
      bash -lc "cargo test --lib --bins --tests --examples --release --target x86_64-unknown-linux-gnu"

[group('docker')]
test-linux-arm64: docker-base-linux-arm64
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running tests in Linux ARM64 Docker image..."
    docker run --rm \
      --platform linux/arm64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-base-linux-arm64 \
      bash -lc "cargo test --lib --bins --tests --examples --release --target aarch64-unknown-linux-gnu"

[group('docker')]
qa-linux-x86: docker-base-linux-x86
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running QA in Linux x86_64 Docker image..."
    docker run --rm \
      --platform linux/amd64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-base-linux-x86 \
      bash -lc "cargo build --release --bin roomeq-qa-quality --bin roomeq-qa-coverage --bin roomeq-qa-features --bin roomeq-qa-synthetic"

[group('docker')]
qa-linux-arm64: docker-base-linux-arm64
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running QA in Linux ARM64 Docker image..."
    docker run --rm \
      --platform linux/arm64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-base-linux-arm64 \
      bash -lc "cargo build --release --bin roomeq-qa-quality --bin roomeq-qa-coverage --bin roomeq-qa-features --bin roomeq-qa-synthetic"

[group('docker')]
test-windows-x86: docker-base-win-x86
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running tests in Windows x86_64 Docker image..."
    docker run --rm \
      --platform linux/amd64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-base-win-x86 \
      bash -lc "cargo test --lib --bins --tests --examples --release --target x86_64-pc-windows-gnullvm --no-run"

[group('docker')]
qa-windows-x86: docker-base-win-x86
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running QA in Windows x86_64 Docker image..."
    docker run --rm \
      --platform linux/amd64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-base-win-x86 \
      bash -lc "cargo build --release --target x86_64-pc-windows-gnullvm --bin roomeq-qa-quality --bin roomeq-qa-coverage --bin roomeq-qa-features --bin roomeq-qa-synthetic"

[group('docker')]
test-windows-arm64: docker-base-win-arm64
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running tests in Windows ARM64 Docker image..."
    docker run --rm \
      --platform linux/arm64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-base-win-arm64 \
      bash -lc "cargo test --lib --bins --tests --examples --release --target aarch64-pc-windows-gnullvm --no-run"

[group('docker')]
qa-windows-arm64: docker-base-win-arm64
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running QA in Windows ARM64 Docker image..."
    docker run --rm \
      --platform linux/arm64 \
      -v "$(pwd)":/project \
      -w /project \
      autoeq-base-win-arm64 \
      bash -lc "cargo build --release --target aarch64-pc-windows-gnullvm --bin roomeq-qa-quality --bin roomeq-qa-coverage --bin roomeq-qa-features --bin roomeq-qa-synthetic"

[group('cross')]
docker-cross-arm: docker-linux-arm64 docker-windows-arm64

[group('cross')]
docker-cross: docker-linux-x86 docker-windows-x86

[group('cross')]
docker-cross-native:
    #!/usr/bin/env bash
    set -euo pipefail
    arch=$(uname -m)
    case "$arch" in
      aarch64|arm64) just docker-cross-arm ;;
      x86_64|amd64) just docker-cross ;;
      *) echo "Unsupported architecture: $arch" ; exit 1 ;;
    esac

[group('cross')]
test-cross-arm: test-linux-arm64 test-windows-arm64

[group('cross')]
test-cross: test-linux-x86 test-windows-x86

[group('cross')]
qa-cross-arm: qa-linux-arm64 qa-windows-arm64

[group('cross')]
qa-cross: qa-linux-x86 qa-windows-x86

[group('cross')]
test-cross-native:
    #!/usr/bin/env bash
    set -euo pipefail
    arch=$(uname -m)
    case "$arch" in
      aarch64|arm64) just test-cross-arm ;;
      x86_64|amd64) just test-cross ;;
      *) echo "Unsupported architecture: $arch" ; exit 1 ;;
    esac

[group('cross')]
qa-cross-native:
    #!/usr/bin/env bash
    set -euo pipefail
    arch=$(uname -m)
    case "$arch" in
      aarch64|arm64) just qa-cross-arm ;;
      x86_64|amd64) just qa-cross ;;
      *) echo "Unsupported architecture: $arch" ; exit 1 ;;
    esac
