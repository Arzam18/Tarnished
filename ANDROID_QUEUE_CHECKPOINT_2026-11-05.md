# Android ARM64 chess-engine queue checkpoint — 2026-11-05

## Completed / confirmed

1. Askaig — Android ARM64 NEON build and phone UCI runtime confirmed.
2. Chessnix — Android ARM64 NEON build and phone UCI runtime confirmed.
3. NARC Genesys — Android runtime confirmed.
4. Pea — Android ARM64 build/runtime confirmed.
5. RukChess — Android ARM64 build/runtime confirmed.
6. Solo — Android ARM64 build/runtime confirmed.
7. Peacekeeper — build/runtime confirmed; extended strength and long-game validation was in progress.
8. Quirky — Android ARM64 build/runtime confirmed after the stack-bounds fix.
9. Kociolek 2.3 — user confirmed successful build and Termux UCI search after NNUE embedding and workflow permission repairs.

## Current engine

10. **Tarnished v6.0 — active; source audit and ARM NEON/Dotprod rebuild package prepared.**

The user-uploaded `Tarnished-master(original).zip` and `Tarnished-master(android-modified).zip`
are byte-identical. Their extracted trees have no Android-specific changes, no ARM NEON
backend, and no workflow. The likely missing-strength cause is host-native x86 NNUE
preprocessing being embedded into an ARM runtime, but the previous workflow/binary was not
provided to prove the exact old flags. The new source forces a matching ARM 128-bit NNUE
layout, adds ARM NEON wrappers, and verifies every processed-network tensor section.
Search soft-time node counts and large stack temporaries were also fixed without changing
search heuristics or thread count.

**Tarnished acceptance remains pending:** GitHub Actions must complete the NDK build; then
both binaries must produce legal UCI moves and pass device `bench`/strength checks. The
workspace could not run NDK or fetch the actual network, so this is not yet a runtime success.

## Next queue

11. Lynx — next after Tarnished acceptance.
12. Booot — following Lynx.

## Build convention reminder

- Target: Android AArch64 API 24, NDK r27c.
- Baseline: `-march=armv8-a+simd`.
- Dotprod: `-march=armv8.2-a+dotprod`; verify SDOT in disassembly before distributing.
- Keep original NNUE network bytes/hashes intact; only preprocess into a derived target-layout
  model and embed it.
- Keep complete source ZIP, modified-files-only ZIP, build workflow/script, notes, and this
  checkpoint together for each engine.
