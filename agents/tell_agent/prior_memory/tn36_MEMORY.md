# Game Manual — Grid Puzzle (Programming Movement)

## Goal
Program movement instructions (max 6 per run) on a 7×7 checkerboard to guide a piece into a target cell. Multiple runs needed per level.

## Movement Rules
- One cell per instruction, step-by-step.
- **M cells** (magenta) = walls, impassable.
- **Y8-checker cells** (8Y pixels, checkerboard pattern) = valid stopping points (waypoints).
- **Programs must end on Y8 waypoint or target cell.** Otherwise piece reverts to pre-run position (program kept).
- **Piece has 14/16 Y pixels** with 2-pixel notch (transparent, shows background).
- **Target cell** has Y at exactly the notch positions → landing creates 16/16 Y = level solved.

## Strategy
1. Identify piece position, notch orientation, target position, and Y8 waypoints.
2. Plan multi-run path using Y8 waypoints as intermediate stops.
3. Avoid walls (M), G portals, and known death cells.
4. Test uncertain cells with short probe runs before committing.
