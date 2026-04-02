# Memory

## Goal
Grid puzzle: navigate Green (2×2, moves 2px/step, even coords only) to Yellow (2×2) target.

## Stable Rules
- **Passable**: '.', B, R, O, W, u, w, p, M, G, Y, n, b, ':'
- **Blocking**: 'd' (dark-gray), K (black)
- Green moves 2px per arrow key. Positions always at even (col, row).
- 64×64 grid. Game area cols 0-37, right panel cols 38-63.
- R piece moves 4 cols per R-right/left, 4 rows per R-up/down.
- R piece bounces (3 frames) when move is blocked.
