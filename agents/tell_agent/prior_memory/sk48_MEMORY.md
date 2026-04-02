## Stable Rules — Player Mode
- **RIGHT**: extends trail +1 cell right. Pushes block if next empty; threads if next occupied/edge.
- **LEFT**: retracts trail -1 cell. Shifts ALL threaded blocks LEFT by 1. Blocked if leftmost block would hit c0.
- **UP/DOWN**: moves player+trail+threaded blocks to adjacent row. 1 frame=blocked, 2 frames=success.
- DOWN onto row with free block at trail position = BLOCKED entirely.

## Stable Rules — Portal Mode
- Colors swap in portal mode: player W→d, portal d→W, trails swap w/.↔./:
- **DOWN**: extends vertical trail 1 row deeper. Pushes free block if next row empty; threads if occupied/edge. 3 frames when threading at edge.
- **UP**: retracts vertical trail 1 row. **Shifts ALL threaded blocks UP by 1.** ← KEY DISCOVERY
- Portal horizontal movement does NOT affect grid blocks.
