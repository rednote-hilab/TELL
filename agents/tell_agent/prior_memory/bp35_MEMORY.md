# Memory

## Goal
Solve grid puzzle level 1. ~1076 total actions used, 9 game overs. 2048 max.

## Stable Rules
- Grid: 64x64. 7 column centers at x=15,21,27,33,39,45,51 (6px spacing).
- Actions: left, right, click(x,y), undo, reset. All cost 1 action.
- 64 actions per attempt before game over (progress bar row 63).
- Player sprite: rows 37-41, starts col 21.
- **Click green** → clears to blue (5 frames). Click non-green → no-op (2 frames).
- **Undo** → reverses last action (restores cleared green OR undoes movement). 2 frames.
- No wrapping: can't go right past col 51 or left past col 15.

## Movement Rules (Band 31-35 dependent)
- Safe (5 frames): both origin and destination have SAME band-31-35 green status
- Deadly (44 frames, game over): origin has green, destination cleared at band 31-35
- Blocked (3-4 frames, no move): structural mismatch (origin cleared, dest green?)

## Band 31-35 Click at Player Column
- K-ceiling → DEADLY always (28 frames for K-wall, game over)
- Blue-ceiling → 44-frame bounce, creates next phase (not deadly)

## Mini-bounce
- Click K-wall at (player_col, 30) between green band above and band 31-35 below
- Requires green above row 30; if no green → DEADLY (28 frames)
- 10 frames, SAFE, transforms grid
