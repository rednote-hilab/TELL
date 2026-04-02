# ARC Puzzle Game - Final Manual

## Goal
Navigate a blue block to a blue target on grid puzzles. Fewer actions = better score.

## Core Mechanics

### Block
- Two sizes: 2×2 and 4×4 (in character cells)
- Movement step = block size (2×2 moves 2 chars, 4×4 moves 4 chars)
- Block top-left always on odd row/col

### 3×3 Puzzle Grid
Located at bottom of screen. Click cells to toggle `.`→`G`. Auto-submits when valid pattern matched.

### Three Puzzle Patterns

   - Toggles block 4×4↔2×2
   - Expansion: 2×2→4×4 expands UP 2, LEFT 2 (2×2 = bottom-right of 4×4). Ignores walls.
   - Contraction: 4×4→2×2 keeps top-left 2×2

   - Teleports block to marked areas
   - 2×2 → teleports to `u`-marked area
   - 4×4 → teleports to `Y`-bordered area; **alternates** between Y areas if multiple exist

   - Fires projectile in direction of **last movement**
   - Passes through ALL walls, stops on magenta box hit
   - Destroys box + removes matching colored barriers (box interior color = barrier color removed)

### Barriers & Boxes
- Magenta-bordered boxes with colored interiors (O=orange, n=maroon, etc.)
- Destroying a box removes all barriers of matching interior color
- Destroyed box cells show `G` (green) residue at border positions

### Action Budget
- Levels have an action budget (clicks + moves both count)
- Exceeding budget causes game_over

## Key Lessons
- Column direction depends on LAST MOVEMENT - set it up carefully before firing
- Cross expansion can overlap walls temporarily (useful before immediate teleport)
- 4×4 Yellow alternates destinations - plan teleport order
- Action budget is real - optimize for fewest total actions (clicks + moves)
