# Memory

## Goal
Solve grid-based puzzle game. Levels 0-3 solved. Currently on level 4 with 616 actions used.

## Stable Rules
- Controls: LEFT/RIGHT move cursor between 8 grid positions (G0-G7, wraps), UP/DOWN cycle through 7 options (mod 7)
- All 8 grids are independently controllable
- Level solves when ALL 8 positions simultaneously show correct grids
- Navigation order: G0→G1→G2→G3→G4→G5→G6→G7→G0 (RIGHT increases, LEFT decreases)
