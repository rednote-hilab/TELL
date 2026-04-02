# Memory — Level 6

## Goal
Place 3 pieces on targets simultaneously. Level 6, ~889 actions used of 2048.

## Confirmed Mechanics
1. **Arrow keys move Green A only.** Clicks don't switch selection.
2. **Every 6th arrow press = trigger** (multi-frame animation, u-walls temporarily open)
3. **Push mechanic**: When Green A moves INTO another piece, Green A STAYS, the other piece SLIDES in that direction until hitting wall/void/piece. Push works at any press, not just triggers.
4. **Trigger push through u-wall**: Push at trigger → pushed piece passes through opened u-wall and continues sliding. CONFIRMED: press 18 pushed Green B UP from (11-12,17) through row 12 u-wall to (11-12,11). Animation: 12 frames, Green B slid from row 17→16→15→14→13→12→11 (stopped at row 11).
5. **Self-slide at trigger**: Green A can slide UP through horizontal u-wall when making parallel (LEFT) move adjacent to it (confirmed press 12 historical). Slide DOWN does NOT work (confirmed press 36 historical).

## Failed Ideas
- Clicking to switch pieces: doesn't work
- Sliding DOWN through u-wall: doesn't work for self-movement
