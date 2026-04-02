# Peg Alignment Puzzle - Game Manual

## Goal
Solve multi-level grid puzzle by moving/rotating shapes so ALL R (red) pegs overlap with at least one other shape's R peg.

## Stable Rules
- **Grid**: 64x64 pixels. Shapes use 3x3 pixel blocks on offset-2 grid (logical position k → pixel 2+3k).
- **Controls**: Arrow keys move selected shape. Click shape body to select it. Space rotates selected shape 90° CW. Reset restarts level.
- **Selection**: First shape is selected by default (no click needed). Only selected shape's R pegs are visible.
- **Unselected shapes** render as dark-gray (`d`). Selected shape renders in its unique color.
- **Shape colors** (when selected): vary per level — e.g., `b`=light-blue, `G`=green, `W`=white, `Y`=yellow, `O`=orange, `u`=purple.
- **Win condition**: Every R peg must overlap with at least one R peg from a different shape. Overlapping pegs render as `:` (gray). Level advances when all pegs are paired.
- **Peg visibility**: Only the selected shape's pegs show as `R`. Other shapes' pegs are hidden (show as `d` or background). Overlap marker `:` appears when selected shape's peg coincides with an unselected shape's peg.
- **Rotation**: 90° CW around bounding box center. Works with half-integer centers.

## Key Patterns

### Peg Difference Matching
Each shape has 2 R pegs with a characteristic difference vector (dr, dc). Two shapes can have BOTH pegs overlap only if their diff vectors match (possibly after rotation: 0°, 90°, 180°, 270° CW).
- Rotation transforms diff (a,b) → (b,-a) → (-a,-b) → (-b,a)
- Group shapes by diff magnitude for pairing.

### Pairing Strategy (3+ shapes)
With N shapes, each having 2 pegs → 2N pegs total. Need N pairs where each pair has pegs from different shapes.
- **Same-magnitude groups**: Shapes whose peg diff magnitudes match can pair both pegs simultaneously.
- **Mixed pairings**: Possible but harder — system of linear equations must be consistent (overdetermined).

### Displacement Optimization
For two shapes with matching diffs to overlap: displacement = |position_difference|. Split between shapes doesn't help (total moves = L1 distance of relative offset). Move one shape entirely when possible to save clicks.

## Transferable Insights
1. **Parse grid at offset-2 step-3** to extract logical positions.
2. **Click each shape** to discover its R peg positions (only visible when selected).
3. **Compute peg diffs**, group by magnitude, determine rotation needs.
4. **Solve linear system** for displacements, minimize total L1 distance.
5. **Execute moves** in batches of ≤10 per ActionSession.
6. Overlap marker `:` confirms successful peg alignment during execution.
