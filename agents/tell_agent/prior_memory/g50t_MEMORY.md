# Memory - Grid Puzzle Game

## Goal
Solve grid puzzles level by level. Currently on level 3. ~1195 of 2048 actions used. ~118 actions on this level.

## Stable Rules
- **Movement**: 7+ frames = successful move. 1 frame = blocked. 13-16 frames = corridor toggle.
- **Corridor entries**: Stepping on entry toggles exit cells. Toggle active while ON entry. Reverts when leaving UNLESS enough compressions.
- **SPACE at entry**: Consumes current piece, returns to start, resets toggles, adds 1 compression to THAT specific entry cell only.
- **3 Pieces**: Indicator `.`=consumed, `B`=active, `w`=unused. Piece 3 = last, can't SPACE.
- **Compressed entry**: Shows `.....` dots pattern. Re-entering gives 7 frames (normal move, no toggle).
