# Memory — Level 5

## Goal
Cover maroon crosses at (33,52),(34,51),(34,53),(35,52) inside u-walled arena.

## Game Rules
- `click(x,y)` and `reset`. Blocks resize ±3/click. Min dim=2. Sizes: {2,5,8,11,14,...}
- Handle (`:`) = fixed anchor edge. Block grows AWAY from handle. All blocks 3 wide perpendicular.
- Same-color blocks share ONE panel, grow/shrink simultaneously.
- **Push**: Growing block pushes another if full perpendicular coverage. Chain pushes work.
- **Pull-back**: Shrinking pulls adjacent blocks toward shrunken block. ALWAYS works regardless of perpendicular coverage. CHAINS confirmed.
- **Toggle bounce**: If ANY block's mapped position overlaps u-wall or goes off-grid, ALL blocks bounce back.
- u-walls DON'T move from toggles. Right top bar (rows 27-29, cols 18-59) is ONE piece, unpushable.
