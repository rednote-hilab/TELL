# Cube Stamp Puzzle — Game Manual

## Goal
Paint a 10×10 answer grid to match a reference pattern using color stamps and face clicks.

## Layout
- **Reference**: rows 3-12, cols 3-12
- **Palette**: y=4. W(x=23), u(x=29), O(x=35), Y(x=41), G(x=47), R(x=53), B(x=59)
- **Cube preview**: front view ~(31,21)
- **Answer block**: rows 34-43, cols 27-36 (starts all W)

## Stamp Patterns (space key from each view)
| View | Nav from front | Region |
|---|---|---|
| Front | — | r=0-4 (top half) |
| Left | left | c+r ≤ 9 (upper-left triangle) |
| Left+down | left,down | c ≤ 4 (left half) |
| Left+down+down | left,down,down | c ≤ r (bottom-left triangle) |
| Right | right | c ≥ r (top-right triangle) |
| Right+down | right,down | c ≥ 5 (right half) |
| Right+down+down | right,down,down | c+r ≥ 9 (bottom-right triangle) |
| Flipped front | l,d,d,r or r,d,d,l | r=5-9 (bottom half) |

## Navigation
- left/right: toggle horizontal (left ↔ front ↔ right)
- down: tilts (left→l+d→l+d+d, right→r+d→r+d+d)
- up: reverses tilt
- **Shortcuts**: l+d+d → right,right → r+d+d (via flipped); reverse with left,left

## Face Click System
- Cube: 4w×3h×2d (all levels tested)
- **Front face click** (~31,21): stamps 4×3 at r=0-2, c=3-6
- **Left+down face click** (~14,39): stamps 3×4 at c=0-2, r=3-6
- Last color on a cell wins (overwrite order matters)

## Strategy
1. Decompose reference into stamp-shaped regions (triangles, halves, rectangles)
2. Plan overlay order: later stamps overwrite earlier ones at boundaries
3. Use face clicks for small rectangles that don't match any full stamp
4. W (white) is default — leave cells unpainted where W is needed
5. Navigate efficiently using shortcuts (especially l+d+d↔r+d+d via flipped)
