# Game Manual - Grid Puzzle (Complete)

## Goal
Navigate player (B) to goal (G) on grid-based maze levels. Minimize total actions. 9 levels total.

## Core Mechanics
- **Grid**: 3x3 pixel blocks, spacing=6. Walls=K, passable=W/., player=B, goal=G.
- **Movement**: up/down/left/right/reset. One cell per action.
- **Animations**: Successful moves = ~13 frames. Failed moves = 1 frame. Push/kill = 15 frames.
- **Turn order**: B moves first, then all entities move simultaneously.

## Entity Types

### n (Maroon) - Charging Chase Enemy
- **Indicator**: u=inactive, Y=active. Position in 3x3 block shows facing direction.
- **Activation**: B enters same column as n, below n (row > n's row).
- **Behavior sequence**: 
  1. Activation turn: stays put
  2. Charging: moves 1 cell in facing direction per B move until blocked
  3. Chase mode: moves 1 cell toward B's OLD position (pre-move) via BFS
- **Tie-breaking**: BFS explores neighbors in sorted (row,col) order, BUT multiple valid first steps may exist. The actual game may use a different tie-break than strict row-col.
- **Lethal**: Entering n's cell or n entering B's cell = game_over.
- **R blocking**: n's pathfinding avoids cells occupied by R entities.

### R (Red) - Pushable Stationary Enemy
- **Stationary** unless pushed by B.
- **Front cell** (cell in R's facing direction): B entering = game_over.
- **Side/back push**: B enters R's cell from perpendicular or behind → R pushed in B's movement direction.
- **Push into wall** (no valid destination cell): R destroyed.
- **Push to valid cell**: R moves there (and may survive - not fully tested).

### O (Orange) - Patrolling Lethal Entity
- **Patrol**: Moves 1 cell in facing direction each turn B moves successfully. Reverses 180° when reaching dead end (can't continue in current direction).
- **Reversal**: On the turn O reaches a dead end, it reverses direction but stays at the dead-end cell. Next turn it moves in the new direction.
- **LETHAL**: O entering B's cell = game_over. B entering O's pre-move cell = uncertain (never tested, but BFS solutions avoided it).
- **O timing manipulation**: B can waste moves (back-and-forth loops) to shift O positions to favorable timing.

## Key Strategies

### O Timing Loops
- O positions are deterministic based on turn count (since O moves each successful B move).
- Inserting back-and-forth loops in B's path shifts O timing without changing B/n relative positions.
- Critical for levels where O blocks mandatory corridors.

### n Evasion
- n follows B's exact trail, typically 1-2 steps behind.
- In narrow corridors, n catches B if B backtracks.
- Endgame: need n to be 2+ steps behind when making final moves to goal.
- n can't pathfind through R cells, creating barriers that limit n's access.

### R Destruction Sequence
- Destroy R entities by pushing into walls (perpendicular approach).
- Order matters: destroy outer R first if inner R's front cell is blocked by outer R.

## Grid Parsing
- Entity detection: look for B,G,n,R,O characters in 3x3 blocks
- Direction indicators: u/Y/d at edges of 3x3 block (left=←, right=→, top=↑, bottom=↓)
- Connectivity: check 3x3 connector pixels between cells (all K = wall, any non-K = connected)

## Observation Tips
- `get_latest_observation()` returns FIRST frame. Use `get_observations(idx)['frames'][-1]` for final state.
- `current_level` returns integer.
- Parse boards with Python, not visual scanning.
