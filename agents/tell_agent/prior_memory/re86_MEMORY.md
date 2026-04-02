# Game Memory

## Goal
Solve level 6. ~1327 of 2048 actions used. 721 remaining.

## Stable Rules
- **Actions**: up/down/left/right (±3 pixels), space (cycle selected shape), reset
- **Grid**: 64×64
- **3 Shapes** (space cycles: b_cross → O_rect → p_cross → b_cross):
  - **b_cross**: cross, arm=9 both directions (19×19 span, 37 pixels). Default color 'b'.
  - **O_rect**: TALL NARROW 22×4 border rectangle (48 pixels). Offsets from center: top=cr-11, bottom=cr+10, left=cc-2, right=cc+1. Default color 'O'.
  - **p_cross**: cross, h-arm=18, v-arm=9 (37×19 span, 55 pixels). Default color 'p'.

## Painting
- Shape pixel touching palette rows 2-6 at specific cols → shape repaints to that color (22-frame animation)
- Palettes: B(cols 15-19), Y(cols 25-29), R(cols 35-39), G(cols 45-49), M(cols 55-59)
- Wrong color = game_over when progress reaches 64

## CRITICAL: Palette Avoidance
- **O_rect**: safe when center_row ≥ 18 (top at row 7). At row ≤ 17, top reaches palette zone.
  - O_rect cols triggering each palette: B=[14,21], Y=[24,31], R=[34,41], G=[44,51], M=[54,61]
- **b_cross/p_cross**: v-arm reaches center_row-9. Safe when center_row ≥ 16 (v-arm at row 7).
  - V-arm at center_col. Triggers palette if center_col in palette col range AND center_row ≤ 15.
  - H-arm at center_row. Triggers palette if center_row in 2-6 AND h-arm cols overlap palette cols.
  - b_cross h-arm is 19 wide (center±9) - hits MULTIPLE palettes if center_row ≤ 6! Avoid.

## Diamond Obstacle
- Rows 28-35, cols ~27-35 (hollow diamond shape of 'w' pixels)
- Blocks ALL shape pixel overlap. Shapes get pushed/stuck.

## Progress Bar (Row 63)
- 'u'=empty, 'w'=filled (64 chars). Ratcheted within attempt (never decreases), resets to 0 on reset.
- Increases ~1 per 5 actions. Sweeping new ground slightly faster (~1 per 3.3).
- Reaching 64 with correct colors = win. Wrong colors at 64 = game_over.

## Sweep Strategy
- After painting all 3 shapes, sweep across board to build progress to 64
- Keep O_rect center_row ≥ 18, crosses center_row ≥ 16 during sweeps
- ~300 actions needed for 64 progress from scratch
- Navigate around diamond carefully
