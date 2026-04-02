# Mirror Puzzle Game — Final Manual

## Goal
Cover all yellow (Y) cells by placing black (K) shapes and their mirror reflections via divider positioning.

## Controls (SPACE cycles modes)
- **Mode A** (default after reset): Move horizontal divider (UP/DOWN)
- **Mode B** (1st SPACE): Move vertical divider (LEFT/RIGHT)
- **Mode C** (2nd SPACE): Move first shape (all arrows)
- **Mode D** (3rd SPACE): Move second shape (all arrows)
- Additional modes for more shapes. Cycle wraps: ...→A

## Mirror Mechanics
- Vertical divider at x=v: mirror_x = 2v − x
- Horizontal divider at y=h: mirror_y = 2h − y
- Both dividers → 4 copies: original, v-mirror, h-mirror, both-mirror
- Win: all Y cells covered by union of all shape copies

## Solver Strategy
1. Parse board: find K shapes (8-connected components), Y cells, initial divider positions, d (mirror) cells
2. Analyze Y symmetry to find best (v, h) — look for max symmetric pairs
3. Brute-force search: for each (v,h), try all shape origin placements to cover all Y cells
4. Pick solution minimizing total moves (divider moves + shape moves + spaces)
5. Execute: Mode A → set h-div, SPACE, Mode B → set v-div, SPACE, Mode C → move shape 1, SPACE, Mode D → move shape 2

## Key Insights
- Shapes are 8-connected (diagonal adjacency counts)
- Mode C always controls the shape with smaller origin coordinates
- Gray (d) cells confirm current mirror positions — useful for verifying divider formulas
- Yellow pattern symmetry analysis quickly narrows divider search space
