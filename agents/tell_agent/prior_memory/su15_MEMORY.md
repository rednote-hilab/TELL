# Memory — Puzzle Game

## Goal
Place target-colored blocks in blue circles. Status area (rows 0-9) shows target color and count.

## Stable Rules
- **Click near block** (≤8 Manhattan from nearest cell): block centers at click point. Block is immune during action.
- **Block placement**: For NxN block, click at (y,x) places block with top-left at approximately (y-floor(N/2), x-floor(N/2)).
- **Merge**: Click within ≤8 of exactly 2 same-type blocks → next-tier block at click point. Blocks immune during merge.
- **Block sizes**: b=2×2, M=2×2(darker), u=3×3, Y=4×4, O=5×5, R=7×7
- **Merge chain UP**: 2b→M, 2M→u, 2u→Y, 2Y→O, 2O→R
- **Downgrade chain DOWN**: R→O→Y→u→M→b→gone (when pink catches block)
- **Pink diamond**: 8 cells in diamond pattern around center. Every click (even empty space) moves ALL pinks.
- **Pink movement**: 4 rows toward target + min(4, |col_diff|) cols toward target per click.
- **Pink catch**: Diamond overlaps non-immune block → block demoted 1 tier + pushed ~9-10 cells in pink's movement direction.
- **Overshoot**: Diamond overlaps immune block → pink travels 2× initial distance, passing through block.
- **Freeze**: Click on pink diamond cell with no block ≤8 Manhattan → freezes that pink for 1 action.
- **Wrong-tier block in circle**: Does NOT complete level. Must match target color.

## Failed Ideas
- Placing wrong-tier block in circle does nothing
