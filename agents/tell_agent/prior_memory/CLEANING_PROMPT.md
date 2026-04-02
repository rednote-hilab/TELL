# Memory Cleaning Prompt

Clean prior game MEMORY.md files for the ARC-AGI-3 agent memory warm-start experiment.

## Philosophy

- **Delete only, never rewrite** — remove entire sections/blocks, do not rephrase, summarize, or add anything
- **When in doubt, delete** — if a section is borderline, delete it
- Only preserve content that would help a NEW agent who has never seen this game before understand the game mechanics and general strategies

## What to DELETE

### Entire sections
1. **Solved Levels / Completed Levels** — entire section
2. **Current Level / Current State / Progress** — entire section
3. **Next Steps / TODO / Priority** — entire section (stale action plans)
4. **What Was Tried & Failed** with specific parameters — if the section has specific grid values, coordinates, d-values, action sequences, delete the ENTIRE section

### Specific content types
5. **Specific grid/pixel data** — B[0]: 10001/11111/..., exact 5x5 grid patterns, base grids, cycle values
6. **Specific coordinates and positions** — TC8,TR8; rows 10-14 cols 48-50; (33,52); target at (5,4). Anything with exact row/col numbers for game objects
7. **Test Pairs / Input-Output pairs** — TI0->TO0 with specific values
8. **Mathematical formulas with specific values** — G0[d] = rot90cw(B[(d+1)%7]), d0=k-1 mod 7, etc.
9. **Toggle/transformation state snapshots** — "After Y->G: B at rows 15-17, cols 27-34"
10. **Initial block/object positions** — "Y: horiz, rows 9-11, cols 40-47"
11. **Action sequences** — R-R-D-D-L, up up right, step-by-step paths
12. **Solution attempts with specific values** — "Solution Z (k=4): d=[3,1,3,0,6,2,4,0]"
13. **Progress bar / budget specific numbers** — "39w/25d", "used 45 actions"
14. **Gate test results with coordinates** — "TC8 UP with Y1+Y2: ALL patterns pass"
15. **Specific puzzle data tables** — panel coordinates, arena layout with exact col/row ranges, divider positions

## What to KEEP

1. **Goal** — one-line game objective
2. **Controls** — what buttons/keys do (generic: "arrow keys move", "space grabs", "click(x,y)")
3. **Core mechanic rules** — how the game engine works in general terms (e.g., "buttons pull liquid between spaces", "growing block pushes adjacent blocks", "mirrors create reflections across dividers")
4. **General strategies** — approach patterns that work across levels (e.g., "probe buttons individually to learn their effect", "plan grab offset to align with delivery direction")
5. **Win/lose conditions** — what triggers level completion or game over
6. **Qualitative failure lessons** — "X approach doesn't work because Y" BUT only if stated without specific coordinates/values (e.g., "push-based approach fails because toggle bounces all blocks, not just the colliding one" is OK; "B toggle from rows 30+ bounces because r+39>=69" is NOT OK)

## Process

For each `{game}_MEMORY.md`:

1. Read the entire file
2. For each section (## heading to next ## heading): does it contain specific coordinates, grid data, position values, or level-specific puzzle data? If yes, delete the ENTIRE section
3. For bullet points within kept sections: does the bullet have specific coordinates or values? Delete that bullet
4. Write the cleaned file — should be SHORT, typically 15-40 lines max
5. The result should read like a game manual, not a level walkthrough

## Verification

Every line in the cleaned output must exist verbatim in the original input. No rephrasing, no new text.

## Example

**Before (80 lines):**
```
## Goal
Solve grid puzzle.

## Rules
- Controls: LEFT/RIGHT move cursor between positions
- Level solves when all positions show correct grids

## Base Grids B[0]-B[6]
B[0]: 10001/11111/00100/11111/10001
B[1]: 11111/00001/00111/00100/11111
...

## Solution Formula
d0=k-1, d1=k+4, d2=k-1 (all mod 7)

## What Was Tried
1. Solution Z (k=4): d=[3,1,3,0,6,2,4,0] -> didn't solve
2. Cycled G3 through all 7 values -> none solved

## Progress Bar
Line 63: 39w/25d initially, 21w/43d after k=4

## Next Steps
1. Use progress bar for hill-climbing
```

**After (8 lines):**
```
## Goal
Solve grid puzzle.

## Rules
- Controls: LEFT/RIGHT move cursor between positions
- Level solves when all positions show correct grids
```
