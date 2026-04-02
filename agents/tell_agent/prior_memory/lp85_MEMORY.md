# Rotation Puzzle Game — Final Manual

## Goal
Click-based rotation puzzle. Move Y-colored 2x2 blocks to target positions marked by Y sparkle diamonds. Fewer actions = better score.

## Controls
- `click` on button diamonds (Red/Green pairs) to rotate/shift block groups
- `reset` to return level to initial state
- Red and Green buttons in a pair are inverses of each other

## Core Mechanics
- Board contains 2x2 colored blocks arranged in geometric patterns
- Buttons perform fixed position permutations (independent of block colors)
- Y sparkle markers (single Y chars in diamond pattern) are FIXED TARGET MARKERS
- Level auto-completes when all Y blocks reach target positions

## Key Strategies
1. **Probe buttons individually** (click R, observe changes, click G to undo) — cost: 2 actions per button
2. **Use baseline diffing** to identify exactly which blocks each button moves
3. **Track Y blocks only** — non-Y block movements are irrelevant to solving
4. **Identify cycle structures** — buttons create permutation cycles; find cycle lengths
5. **Use modular arithmetic** — find minimum button presses via cycle length optimization
6. **Prefer inverse buttons** — G2×2 can replace R2×4 when cycle length is 6 (saves 2 actions)

## Patterns
- **Diagonal Cycles**: Buttons often rotate blocks along diagonal lines in triangular/fan layouts
- **Column Paths**: After diagonals reach grid columns, blocks travel straight down columns
- **Coupled Rotations**: One button (R4) can create long cycles that span multiple regions
- **Pre-conditioning**: Use small-cycle buttons (R2/R3) to position blocks before applying the main rotation
- **Cycle Optimization**: When needing +k steps on an n-cycle, use min(k, n-k) clicks (forward or inverse)
