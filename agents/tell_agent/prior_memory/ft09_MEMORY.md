# Memory

## Goal
Grid puzzle: click cells to toggle between two colors. Match target pattern derived from center cell clues. Currently stuck on level 4 (493 total actions, 449 on this level).

## Stable Rules (PROVEN)
- Grid of colored cells. Two colors per level (level 0: B/R, level 1: B/O, level 4: G/u).
- Click non-center cells to toggle between colors. Centers are static (unclickable).
- Centers display 3x3 patterns using symbols: W, M, G, ., :, and center's own color.
- **Symbol meanings (PROVEN from levels 0 and 1):**
  - W = same as center color
  - . = different from center color  
  - : = no constraint (used for non-existent neighbors)
  - M = same as center color (only appears in G-center patterns)
  - G = different from center color (only appears in G-center patterns)
- Center-to-center constraints are informational only (centers can't change)
- Reset restores all non-center cells to initial color (all same default color)
- Top-right 4x4 blocks show the two available colors (decorative legend)
- Level completes when last correct cell is clicked (produces 2-frame animation)
- Available actions: up, down, left, right, space, click, reset
