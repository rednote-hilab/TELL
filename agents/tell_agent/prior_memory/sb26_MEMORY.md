# Color Container Puzzle - Game Manual

## Goal
Fill nested containers with colors/markers to produce a reading that matches the target sequence. Press space to submit.

## Core Mechanics

### Containers
- **Outer container**: White corners (W). This is the root of the reading.
- **Sub-containers**: Colored borders only (Blue, Green, etc.). Referenced via hollow markers.

### Palette Items
- **Solid blocks** (4×4 colored): Place as color values in slots.
- **Hollow markers** (colored border, dark center): Reference a sub-container of that border color.

### Interaction
- Click palette item to select (white border appears), click empty slot (`..`) to place.
- `undo` removes the last placement.
- `reset` clears everything.
- `space` submits for verification.

## Critical Rule: Reading Order (Depth-First Walk)

The reading is a **depth-first sequential walk** through container slots:
1. Start at slot 1 of the outer container.
2. For each slot left-to-right:
   - **Regular color**: Output it as the next item in the sequence.
   - **Hollow marker**: Jump to slot 1 of the referenced sub-container (like a function call). Walk that container's slots L→R. When the sub-container ends, return to the next slot in the parent.
3. This is recursive: sub-containers can contain markers to other sub-containers.

### Key Implications
- **Marker position matters**: A marker at slot 1 of a sub-container immediately jumps away before outputting any of that container's colors. Place markers AFTER the colors you want output first.
- **Mutual references create cycles**: Red→Blue→Red→Blue→... produces an infinite repeating pattern. The game accepts this IF the first N items match the N-item target.
- **Same marker twice**: Referencing the same sub-container multiple times re-reads it each time (proven L4).

## Key Patterns

1. **Direct Fill**: All slots = regular colors. Target length = slot count.
2. **Single Expansion**: One marker in outer expands to sub-container contents.
3. **Multiple Markers**: Same marker repeated = sub-container read multiple times.
4. **Deep Nesting**: Markers in sub-containers reference deeper sub-containers.
5. **Mutual/Cyclic Reference**: Two containers reference each other. Creates infinite repeating pattern. Game accepts if first N items match N-item target. **Marker must be at the END of sub-container slots** so colors are output before jumping back.

## Strategy
1. Parse target sequence and available containers/palette.
2. Count slots vs target items to determine how many markers needed.
3. For repeated patterns in target, consider same-marker-twice or cyclic references.
4. Place colors in reading order, markers where expansion should occur.
5. Remember: marker position = where the "function call" happens in the sequence.
