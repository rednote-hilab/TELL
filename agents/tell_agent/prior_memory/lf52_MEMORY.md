# Memory

## Goal
Grid puzzle game. Solve levels with fewest actions. Currently on Level 1. ~193 actions remaining (1855 of 2048 used).

## Stable Rules
- Actions: up, down, left, right, click(x,y), undo, reset
- Solitaire: click peg (2f select), click empty 2-away (10f jump over adjacent peg)
- Dead end auto-triggers corridor mode (27f on last jump action)
- Corridor: arrow keys move orange along K corridors (3f=moved, 2f=blocked)
- Solitaire mode: L/R move box along H corridor, U/D work at corridor junctions
- Box position = orange starting position in corridor mode
- Emptying a border cell OPENS the external K wall (tested: C7 empty → notch→C7 opens)
- Internal W walls between grid cells do NOT open when cells empty
