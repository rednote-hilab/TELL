# Sokoban Puzzle Game - Final Manual

## Goal
Deliver boxes to blue-bordered target slots on a 64×64 grid. Complete levels by filling one full COLUMN of targets.

## Core Mechanics
- Player/boxes are 4×4 blocks; each move = 4 pixels
- **Grab**: Face box (move toward it even if blocked), then `space`. Offset locked at grab direction.
- **Release**: `space` again. Box stays at offset position. If on target = delivered.
- Offsets: up=(-4,0), left=(0,-4), right=(0,+4), down=(+4,0)
- `K` = wall (solid), `B` = target border (passable), `w` = empty
- KBBK/uuuu blocks may be grabbable like boxes
- **WARNING**: `space` near delivered boxes can accidentally grab them!

## Completion
- Fill ONE COLUMN of target slots → level complete (not all slots needed)

## Key Strategies
- Plan grab offset to align with delivery direction
- Route through wall gaps; verify box fits through gap rows
- On rows with delivered boxes, carried boxes may block passage
- Orange enemies move each turn; avoid and plan around them
- Column-fill completion means prioritize filling all rows in one column
