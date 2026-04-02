# Liquid Puzzle Game Manual

## Goal
Route colored markers through divider gaps to align each marker with its matching colored gap in a divider wall. All alignments must be simultaneous.

## Controls
- `click(x, y)` where x=column, y=row. Grid is 64×64. Row 0 = budget (p chars).
- `reset` on game_over.

## Core Mechanics

### Spaces & Liquid
- Rectangular regions containing liquid (W). Liquid fills in one direction (typically downward).
- Markers sit at the liquid front (first empty row after liquid).

### Buttons (BB)
- Blue cells at space boundaries near divider walls.
- Click transfers **2 rows** of liquid per click (confirmed level 6; was noted as 3 in earlier levels — may vary).
- "Buttons pull liquid FROM the adjacent space INTO the button's own space."

### Dividers
- Walls (KK) between spaces with special features:
  - **ww gaps**: Transit gaps for marker movement. Turn orange (OO) when opening condition met.
  - **Colored gaps** (GG/YY/uu): Alignment targets. Markers must be in adjacent space with liquid front at the gap's row.

### Gap Opening Condition
- A ww gap turns orange when **both** adjacent spaces have liquid level = (gap_start_row − space_top_row).
- Formula: liquid_rows_needed = gap_row − space_first_row
- Both sides must meet their condition simultaneously.

### Marker Transit
- Click an orange (OO) gap to move marker through.
- If markers on both sides → they swap.
- If marker on one side only → it moves to the other side.
- Animation: ~37-55 frames. Marker appears at destination space's liquid front.
- Liquid levels do NOT change from gap clicks.

### Liquid Barriers
- Liquid seeps through K sections of dividers. Colored gaps (GG/YY/uu) act as barriers.
- Opening a divider gap removes its barrier, allowing liquid to extend further.

### Win Condition
- All markers aligned with their matching colored gap simultaneously.
- Marker at row R aligns with colored gap at row R in the adjacent divider.
- Win triggers on the click that completes the last alignment.

## Strategy Patterns

### Level Measurement
- Count consecutive W rows from fill direction start.
- Marker position = liquid_level + space_start_row (first empty row).

### Routing
1. Identify which space each marker must end in (adjacent to its colored gap).
2. Plan transit sequence through ww gaps, avoiding two markers in same space.
3. Each gap opening requires specific liquid levels → plan liquid redistribution.
4. After all markers positioned, adjust levels for final alignment.

### Efficiency
- Minimize C oscillations (filling/draining center space for gap conditions).
- Use marker swaps when both sides have markers at a gap.
- Plan transfer sequences to avoid wasted clicks.

## Key Insights
- Gap condition = distance from space top to gap row (in rows).
- Transfer amount may vary by level (was 2 rows in level 6).
- Center space spanning both halves needs large liquid levels for bottom gaps.
- Plan marker routing to avoid concurrent markers in same space.
- Colored gaps are alignment targets only — don't need to be "opened."
