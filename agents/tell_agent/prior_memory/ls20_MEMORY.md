# Game Memory

## Goal
Navigate block through maze, transform 3×3 pattern via PASS tile to match target, submit at BOX.

## Stable Rules
- Arrow keys move 1 tile. `d`=wall, `:`=floor.
- Hidden gate fail = 6-frame anim (1-frame if last life), lose 1 life, reset to START w/ original pattern, all flags cleared.
- PASS at TC8,TR8: each entry = 90°CW rotation (any direction).
- Y1 at TC1,TR2: D entry from TC1,TR1 activates Y1 flag. LEFT entry does nothing.
- Y2 at TC6,TR9: entering activates Y2 flag.
- BOX bounce (wrong pattern) = deactivates Y1, back to TC1,TR6, no life lost.
