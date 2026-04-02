# Game Manual - Grid Puzzle (Two Markers)

## Goal
Unite two 'b' markers on the same cell across 6 levels.

## Core Mechanics
- **13x13 grid** (0-indexed), R=wall, K=empty, b=marker, B=movable block
- **Movement**: UP/DOWN move both markers same direction. RIGHT=converge (L→right, R→left). LEFT=diverge (L→left, R→right).
- **Wall blocking**: If either marker hits R/wall/closed-barrier, BOTH markers stay put.
- **B blocking**: B blocks ONE marker, but the OTHER marker STILL MOVES. This is the key split-movement mechanic.
- **B leaving R-wall**: Creates permanent K gap (only for R cells, not barriers).

## Barriers & Switches
- G barriers opened by G switches; O barriers by O switches. Close when switch vacated.
- Markers on switches keep barriers open. Frozen markers (w) in Y-mode also keep switches active.

## Y-Mode (Block Repositioning)
- Click B → Y-mode: markers become 'w' (frozen), cursor Y appears at B position.
- Move Y on K cells only. Y CANNOT enter: R, G/O switches, w cells, barriers (unless open).
- Click any w marker → exit Y-mode: B placed at Y's current position, markers unfreeze.

## Strategy Pattern
1. Position markers on switches to open barriers
2. Use Y-mode to place B strategically to block one marker
3. Move in a direction — blocked marker stays, other moves through open barrier
4. Repeat Y-mode repositioning + directional moves to converge markers

## Key Insights
- Plan the full sequence of B placements before executing
- Each B placement enables one directional movement phase
- Barriers + switches create the need for careful ordering
- Y-mode paths must account for which barriers are open (frozen markers keep switches active)
