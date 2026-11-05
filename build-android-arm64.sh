#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

API_LEVEL="${ANDROID_API_LEVEL:-24}"
NDK="${NDK_PATH:-${ANDROID_NDK_HOME:-${ANDROID_NDK_LATEST_HOME:-}}}"
if [[ -z "$NDK" ]]; then
    SDK_ROOT="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-/usr/local/lib/android/sdk}}"
    NDK_ROOT="$SDK_ROOT/ndk"
    [[ -d "$NDK_ROOT" ]] || { echo "ERROR: Android NDK not found; set NDK_PATH." >&2; exit 1; }
    NDK="$(find "$NDK_ROOT" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
fi

TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
CXX="$TOOLCHAIN/aarch64-linux-android${API_LEVEL}-clang++"
OBJDUMP="$TOOLCHAIN/llvm-objdump"
READELF="$TOOLCHAIN/llvm-readelf"
NM="$TOOLCHAIN/llvm-nm"
for tool in "$CXX" "$OBJDUMP" "$READELF" "$NM"; do
    [[ -x "$tool" ]] || { echo "ERROR: Missing Android toolchain tool: $tool" >&2; exit 1; }
done

HOST_CXX="${HOST_CXX:-g++}"
command -v "$HOST_CXX" >/dev/null || { echo "ERROR: Host C++20 compiler '$HOST_CXX' not found." >&2; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required." >&2; exit 1; }
command -v curl >/dev/null || { echo "ERROR: curl is required." >&2; exit 1; }
command -v zip >/dev/null || { echo "ERROR: zip is required." >&2; exit 1; }

NETWORK_TAG="$(tr -d '[:space:]' < network.txt)"
[[ -n "$NETWORK_TAG" ]] || { echo "ERROR: network.txt is empty." >&2; exit 1; }
NETWORK_FILE="${TARNISHED_NETWORK_FILE:-${NETWORK_TAG}.bin}"
NETWORK_URL="${TARNISHED_NETWORK_URL:-https://github.com/Bobingstern/tarnished-nets/releases/download/${NETWORK_TAG}/${NETWORK_TAG}.bin}"
if [[ ! -s "$NETWORK_FILE" ]]; then
    echo "Downloading Tarnished network: $NETWORK_TAG"
    curl --fail --location --retry 3 --retry-delay 2 "$NETWORK_URL" --output "$NETWORK_FILE"
fi

# Validate the binary model layout before reading it into QuantisedNetwork.
python3 - "$NETWORK_FILE" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
expected = (
    (16 * 1792 * 768 + 1792) * 2
    + 1792 * 8 * 16
    + 8 * 16 * 4
    + (16 * 2) * 8 * 32 * 4
    + 8 * 32 * 4
    + 32 * 8 * 4
    + 8 * 4
)
actual = path.stat().st_size
print(f"NNUE file: {path} ({actual} bytes); expected QuantisedNetwork size: {expected}")
if actual != expected:
    raise SystemExit("ERROR: NNUE byte size does not match Tarnished v6.0 QuantisedNetwork.")
PY

mkdir -p build
rm -rf dist
mkdir -p dist
rm -f processed.bin preprocess/permute

# CRITICAL: the preprocessing executable runs on the x86_64 CI host, but its
# output is embedded in the ARM binary. Force the ARM NEON 128-bit layout and
# skip all AVX2/AVX-512-only FT lane permutations.
echo "Building ARM-NEON-layout network preprocessor on host: $HOST_CXX"
"$HOST_CXX" -O3 -std=c++20 -DNDEBUG -pthread -DTARNISHED_LAYOUT_NEON=1 \
    preprocess/permute.cpp src/parameters.cpp -o build/permute-neon-layout
build/permute-neon-layout "$NETWORK_FILE" processed.bin

EXPECTED_PROCESSED_SIZE=44308544
ACTUAL_PROCESSED_SIZE="$(stat -c%s processed.bin)"
[[ "$ACTUAL_PROCESSED_SIZE" == "$EXPECTED_PROCESSED_SIZE" ]] || {
    echo "ERROR: processed NNUE size $ACTUAL_PROCESSED_SIZE != $EXPECTED_PROCESSED_SIZE" >&2
    exit 1
}
python3 scripts/verify-neon-preprocess-layout.py "$NETWORK_FILE" processed.bin

SOURCES=(src/*.cpp src/external/format.cpp)
COMMON_FLAGS=(
    -std=c++20 -O3 -DNDEBUG -pthread -flto
    -fno-finite-math-only -funroll-loops
    -ffunction-sections -fdata-sections
    -static-libstdc++
    -DEVALFILE='"processed.bin"'
)

# Clang emits LLVM bitcode for -flto objects and does not generate useful
# .su files from those objects. Compile a second, non-LTO audit pass with the
# exact same Android ISA/optimization flags to obtain per-function stack use.
STACK_COMMON_FLAGS=(
    -std=c++20 -O3 -DNDEBUG -pthread -fno-lto
    -fno-finite-math-only -funroll-loops
    -ffunction-sections -fdata-sections
    -fstack-usage -Wframe-larger-than=65536
    -DEVALFILE='"processed.bin"'
)

build_variant() {
    local flavor="$1"
    local march="$2"
    local objdir="build/${flavor}"
    local stackdir="build/${flavor}-stack-audit"
    local binary="dist/tarnished-6.0-android-arm64-${flavor}"
    local -a flags=("${COMMON_FLAGS[@]}" -march="$march" -mtune=cortex-a76)

    rm -rf "$objdir"
    mkdir -p "$objdir" "dist/stack-usage/${flavor}"
    echo "Building ${flavor} for Android API ${API_LEVEL}"

    for source in "${SOURCES[@]}"; do
        local object="$objdir/$(printf '%s' "$source" | tr '/.' '__').o"
        echo "  CXX $source"
        "$CXX" "${flags[@]}" -c "$source" -o "$object"
    done

    "$CXX" "${flags[@]}" -Wl,-z,max-page-size=16384 -Wl,--gc-sections \
        "$objdir"/*.o -o "$binary"
    chmod 755 "$binary"

    # Confirm the output is AArch64 and that the complete preprocessed network
    # was assembled inside the binary at the alignment Network requires.
    "$READELF" -h "$binary" > "dist/${flavor}-elf-header.txt"
    grep -q 'AArch64' "dist/${flavor}-elf-header.txt"
    "$NM" "$binary" > "dist/${flavor}-symbols.txt"
    local data_addr end_addr
    data_addr="$(awk '$NF == "gEVALData" {print $1}' "dist/${flavor}-symbols.txt" | tail -n 1)"
    end_addr="$(awk '$NF == "gEVALEnd" {print $1}' "dist/${flavor}-symbols.txt" | tail -n 1)"
    [[ -n "$data_addr" && -n "$end_addr" ]] || {
        echo "ERROR: embedded NNUE symbols are missing from $binary" >&2
        exit 1
    }
    python3 - "$data_addr" "$end_addr" "$EXPECTED_PROCESSED_SIZE" <<'PY'
import sys
start, end, expected = (int(x, 16) if i < 2 else int(x) for i, x in enumerate(sys.argv[1:]))
if start % 64:
    raise SystemExit(f"ERROR: embedded network starts at non-64-byte address 0x{start:x}")
if end - start != expected:
    raise SystemExit(f"ERROR: embedded network is {end-start} bytes; expected {expected}")
print(f"Embedded NNUE verified: {end-start} bytes at 0x{start:x} (64-byte aligned)")
PY

    "$READELF" -d "$binary" > "dist/${flavor}-dynamic.txt" || true
    if grep -q 'libc++_shared.so' "dist/${flavor}-dynamic.txt"; then
        echo "ERROR: binary depends on libc++_shared.so; this package must be self-contained." >&2
        exit 1
    fi

    "$OBJDUMP" -d "$binary" > "dist/${flavor}-disassembly.txt"
    if [[ "$flavor" == "dotprod" ]]; then
        grep -Eq '[[:space:]]sdot[[:space:]]' "dist/${flavor}-disassembly.txt" || {
            echo "ERROR: Dotprod build contains no SDOT instruction." >&2
            exit 1
        }
    else
        if grep -Eq '[[:space:]]sdot[[:space:]]' "dist/${flavor}-disassembly.txt"; then
            echo "ERROR: Baseline NEON binary unexpectedly contains SDOT." >&2
            exit 1
        fi
        grep -Eq '[[:space:]](smull|smull2|sqxtun|fmla)[[:space:]]' "dist/${flavor}-disassembly.txt" || {
            echo "ERROR: Baseline binary has no expected NEON arithmetic." >&2
            exit 1
        }
    fi

    # The LTO objects above are bitcode, so generate independent stack-usage
    # reports with LTO disabled but otherwise identical target optimizations.
    rm -rf "$stackdir"
    mkdir -p "$stackdir"
    echo "Generating non-LTO AArch64 stack-usage report for ${flavor}"
    for source in "${SOURCES[@]}"; do
        local audit_object="$stackdir/$(printf '%s' "$source" | tr '/.' '__').o"
        "$CXX" "${STACK_COMMON_FLAGS[@]}" -march="$march" -mtune=cortex-a76 \
            -c "$source" -o "$audit_object"
    done
    find "$stackdir" -maxdepth 1 -name '*.su' -type f -exec cp {} "dist/stack-usage/${flavor}/" \;
    python3 scripts/summarize-stack-usage.py "dist/stack-usage/${flavor}" \
        > "dist/${flavor}-stack-summary.txt"
    {
        echo "Flavor: $flavor"
        echo "Compiler: $CXX"
        echo "Target flags: -march=$march -mtune=cortex-a76"
        echo "Embedded NNUE bytes: $EXPECTED_PROCESSED_SIZE"
        echo "Binary bytes: $(stat -c%s "$binary")"
        echo "Network SHA256: $(sha256sum "$NETWORK_FILE" | awk '{print $1}')"
        echo "NEON verification: PASS"
        if [[ "$flavor" == "dotprod" ]]; then echo "SDOT verification: PASS"; else echo "SDOT absent: PASS"; fi
        echo "Stack usage summary: dist/${flavor}-stack-summary.txt"
        echo "Stack usage reports: dist/stack-usage/${flavor}/"
    } > "dist/${flavor}-build-info.txt"
}

build_variant neon armv8-a+simd
build_variant dotprod armv8.2-a+dotprod

# Keep only distributable binaries and diagnostics; the network and intermediate
# preprocessed file are intentionally not placed beside the UCI executables.
rm -f dist/Tarnished-Android-ARM64.zip
(
    cd dist
    zip -q -r -9 Tarnished-Android-ARM64.zip \
        tarnished-6.0-android-arm64-neon \
        tarnished-6.0-android-arm64-dotprod \
        neon-build-info.txt dotprod-build-info.txt \
        neon-elf-header.txt dotprod-elf-header.txt \
        neon-dynamic.txt dotprod-dynamic.txt \
        neon-stack-summary.txt dotprod-stack-summary.txt \
        stack-usage
)

echo "Build complete: $ROOT/dist/Tarnished-Android-ARM64.zip"
