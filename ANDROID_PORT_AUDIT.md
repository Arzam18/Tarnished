# Tarnished v6.0 Android ARM64 — source audit and repair notes

## Scope and source provenance

The user supplied `Tarnished-master(original).zip` and `Tarnished-master(android-modified).zip`.
Both archives are byte-for-byte identical (SHA-256:
`1b9317de5dcaaea577f3794d22b03e3b1b423b8ef1e55d6fe87370f39fa568a7`).
The extracted directory diff is empty. Neither archive contains a GitHub Actions workflow,
Android build script, or an ARM NEON branch in `src/simd.h`. The supplied source is the
Tarnished v6.0 (Eternal) source tree. Therefore, there is no previous Android-only source
diff or workflow in these uploads to compare directly. The previous Android artifact's
actual compiler flags and preprocessing output remain unverified.

## Main likely cause of strength loss: host-native NNUE preprocessing

The upstream root `Makefile` and `preprocess/Makefile` both default to `ARCH_LEVEL=native`,
which expands to `-march=native`. The preprocessing program is built and executed on the
GitHub runner host, then its output is embedded in the target engine. On x86-64, this can
select AVX2/AVX-512 and execute x86-specific Feature Transformer (FT) lane permutations.
Those FT permutations are not the layout consumed by a 128-bit ARM NEON runtime. The
preprocessor also chooses the L1 grouped-weight layout from host vector constants. Using
an x86-native preprocessed network in an ARM runtime can produce a syntactically valid,
embedded NNUE that evaluates incorrectly, explaining a severe strength drop without a
load error or crash.

This is a conditional root-cause finding, not proof about the previous binary: the actual
previous workflow was not included. The new build explicitly compiles the host preprocessor
with `TARNISHED_LAYOUT_NEON=1`, sets ARM NEON dimensions, skips x86 FT lane shuffles, and
emits L1 weights grouped for the ARM 4-input/4-output dot-product kernel.

The new `scripts/verify-neon-preprocess-layout.py` verifies every segment: FT weights and
biases retain ARM order; L1 weights are grouped in four-input/four-output blocks; L1 biases
are preserved; L2 weights are transposed to bucket/input order; L2 biases and L3 biases are
preserved; L3 weights are transposed. It also verifies the exact raw and processed byte
counts before the model can be embedded.

## NEON implementation and numerical path

The supplied `src/simd.h` selected x86 AVX512/AVX2 or scalar AUTOVEC; it did not contain the
ARM NEON path described in the request. The modified version adds a 128-bit Advanced SIMD
implementation for the operations used by `NNUE::activateL1`, `forwardL1`, `forwardL2`, and
`forwardL3`.

- Baseline target: `-march=armv8-a+simd`; uses widening multiply/pairwise-add for signed
  byte dot products and must not contain SDOT.
- Dotprod target: `-march=armv8.2-a+dotprod`; uses `vdotq_s32` and must contain SDOT.
- Both target variants consume the same ARM-layout processed model and keep the upstream
  network architecture and inference topology.
- The u8-by-s8 dot-product implementation centers unsigned activations by 128, performs
  signed dot products, then adds the mathematically exact `128 * sum(weights)` correction.
  Tarnished's L1 activations are clipped and scaled to 0..63, so the x86 `maddubs` pairwise
  signed-16 saturation edge case is not reached by the model's activation range.
- ARM embedded network alignment is increased to 64 bytes, matching the alignment required
  by `Network`.

No upstream evaluation formula, network parameters, pruning thresholds, reduction formula,
move ordering, or search depth was tuned or weakened.

## Search/runtime issues addressed

1. `Search::Limit::nodeCounts` was uninitialized. `outOfTimeSoft()` reads this table to
   estimate the fraction of nodes spent on the current best root move. Random stack data
   could affect soft-time decisions. It is now value-initialized and reset in `start()`.
2. `ThreadInfo::startSearching()` previously assigned `bucketCache = InputBucketCache()`.
   The RHS temporary contains a large 2 x 2 x 16 cache of 1792-element `int16_t` feature
   arrays. On the host stack-usage test this contributed to a 266,904-byte frame. It now
   calls `bucketCache.clear()` in place. `clear()` resets both validity flags and cached
   piece masks, which is necessary because the next refresh computes piece deltas relative
   to those masks; stale masks would corrupt the accumulator.
3. `iterativeDeepening()` previously received a `Limit` by value. Each worker now copies
   the shared searcher's limit into its own `ThreadInfo::limit` and passes that local object
   by reference. The `Searcher::startSearching()` parameter is also a const reference. This
   removes avoidable large copies while preserving one independent Limit per worker.

Host Clang `-O3 -fstack-usage` comparison:

| Function | Before | After |
|---|---:|---:|
| `ThreadInfo::startSearching()` | 266,904 bytes | 56 bytes |
| `Search::iterativeDeepening()` | by-value copy | 2,744-byte frame (by reference) |
| `Search::search<false/true>()` | no semantic changes | 8,632 / 8,600 bytes |
| `Search::qsearch<false/true>()` | no semantic changes | 4,312 / 4,296 bytes |

These figures are host x86-64 diagnostic measurements, not Android AArch64 stack measurements.
The Android build script makes a separate non-LTO AArch64 stack-usage pass for both ISA
variants and uploads the raw `.su` reports plus summaries. Review those reports before any
further thread-stack policy change. No thread count, search limits, or pruning behavior was
changed to mask a stack issue.

## Files changed

- `src/simd.h`: ARM NEON wrapper backend.
- `src/parameters.h`: architecture-independent constants for the host-side ARM-layout
  preprocessing pass.
- `preprocess/permute.cpp`: bypass host x86 FT permutation and use ARM L1 grouping.
- `src/external/incbin.h`: 64-byte AArch64 embedded-network alignment.
- `src/nnue.h`: in-place input-bucket cache reset.
- `src/search.h`, `src/search.cpp`, `src/searcher.h`, `src/searcher.cpp`: deterministic root
  node counts and removal of large by-value Limit/cache temporaries.
- `.github/workflows/android-arm64.yml`, `build-android-arm64.sh`: reproducible NDK r27c
  workflow building baseline NEON and Dotprod variants, checking embedding/alignment/ISA,
  and collecting target stack reports.
- `scripts/verify-neon-preprocess-layout.py`, `scripts/summarize-stack-usage.py`: validation.
- `README.md`, `.gitignore`: build instructions and generated-file ignores.

## Validation actually completed in this workspace

- The two uploaded source ZIP SHA-256 values match; extracted trees have no differences.
- Host C++20 syntax check passed for all engine translation units.
- The network layout verifier passed all eight sections using a deterministic synthetic
  raw network of the exact expected input size; output size was 44,308,544 bytes.
- The NEON wrapper code cross-compiled with Clang 17 for `aarch64-none-elf` both with
  `armv8-a+simd` and `armv8.2-a+dotprod`. Baseline assembly had no SDOT; Dotprod assembly
  contained SDOT.
- The in-place stack fix reduced the host `ThreadInfo::startSearching()` frame from
  266,904 bytes to 56 bytes. The stack-summary parser was tested on the resulting reports.
- `bash -n build-android-arm64.sh` passed; the workflow YAML parsed and its
  `workflow_dispatch` trigger was checked.

## Not yet verified

This workspace did not have Android NDK r27c installed and could not resolve GitHub to fetch
`lichdragon-3.bin`. Therefore no complete Android binary was built here, and no real network
was available for inference or play testing. The new GitHub Actions workflow is the first
full target build validation. Do not treat the source audit or synthetic layout test as a
completed strength recovery until Actions passes and both binaries complete a phone-side
`bench`/UCI search test.

## Suggested acceptance checks after Actions

1. Confirm both artifacts contain exactly one engine executable each plus diagnostics;
   no external `lichdragon-3.bin` is required beside the executable.
2. Check `neon-build-info.txt` and `dotprod-build-info.txt`: exact processed NNUE size,
   64-byte symbol alignment, no `libc++_shared.so`; baseline has NEON but no SDOT, Dotprod
   has SDOT.
3. Review `neon-stack-summary.txt` and `dotprod-stack-summary.txt`, especially `main`,
   `UCIGo`, `ThreadInfo::startSearching`, `iterativeDeepening`, recursive `search`, and
   `qsearch` frames.
4. On the Galaxy A16 5G, run `bench` and a direct UCI search on each binary. Use the baseline
   NEON binary for general AArch64 compatibility and the Dotprod binary only on devices
   that support ARMv8.2-A dot product.
5. Compare identical settings, openings, time control, hash size, and thread count in an
   A/B tournament. The official v6.0 release lists a reference bench of 3,044,947 nodes;
   treat it as a diagnostic reference rather than a strict equality test because ARM and
   x86 floating-point reduction order can differ slightly.

