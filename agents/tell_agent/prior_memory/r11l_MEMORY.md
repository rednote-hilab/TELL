# Memory

## Goal
Click-based puzzle: move 3 colored balls to matching target circles on 64×64 grid. Actions: "click" and "reset". Fewer clicks = better.

## Stable Rules

### Movement Formulas (VERIFIED)
- **u and B cursors**: `dx = (click_x - cursor_x) // 2`, `dy = (click_y - cursor_y) // 2` (Python floor division)
- **G cursor**: `dx = (click_x - cursor_x) // 3`, `dy = (click_y - cursor_y) // 3`
- Ball matching cursor color moves by (dx, dy). Cursor moves to click position on SUCCESS.
- **On FAILED ball move**: cursor does NOT move. frames=1 means failure or instant action.

### Clickable Cells
- Only 'K' (black) and ':' (diamond outline) cells can be clicked
- 'W' (white border), '.' (light-gray), 'b' (obstacle), etc. are NOT clickable
- Clicks on non-clickable cells are silently rejected (frames=1, no effect)

### Path Check
- Ball moves along Bresenham line. At EACH integer position, 21-cell body (5×5 minus 4 corners) checked against non-passable cells
- Passable: Ball0(OGu): {O,G,u,M,K,w,W}; Ball1(BR): {B,R,M,K,w,W}; Ball2(GY): {G,Y,M,K,w,W}
- ':' blocks ALL balls. 'b' blocks all balls. '.' blocks all balls.

### Diamond Interaction
- Click within manhattan ≤ 2 of diamond center → diamond action (not ball move)
- Same-color: cursor teleports to diamond center, new diamond at old cursor pos, old diamond consumed
- Different-color: SWAP - cursor changes color, moves to diamond center, new diamond of old color at old pos

### Auto-Interaction (CRITICAL)
- After cursor arrives at new position (via teleport/swap), if cursor's 5×5 body overlaps any ':' cell of a DIFFERENT diamond → automatic swap occurs
- This chains: cursor goes to that diamond, old position gets new diamond

### Cursor Color → Ball
- u cursor → Ball0 (OGu)
- B cursor → Ball1 (BR)  
- G cursor → Ball2 (GY)
