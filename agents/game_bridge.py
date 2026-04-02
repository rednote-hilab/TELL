from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from arcengine import FrameData, GameAction

ACTION_BY_LABEL: Dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "space": GameAction.ACTION5,
    "click": GameAction.ACTION6,
    "undo": GameAction.ACTION7,
    "reset": GameAction.RESET,
}

LABEL_BY_ACTION: Dict[GameAction, str] = {value: key for key, value in ACTION_BY_LABEL.items()}


@dataclass
class BoundGameBridge:
    game_id: str

    def available_action_labels(self, frame: FrameData) -> List[str]:
        labels: List[str] = []
        for action_id in frame.available_actions:
            try:
                action = GameAction.from_id(action_id)
            except Exception:
                continue
            label = LABEL_BY_ACTION.get(action, action.name.lower())
            labels.append(label)
        # ARC wrappers support RESET as a global action; expose it even if the
        # per-frame list only contains gameplay actions.
        if "reset" not in labels:
            labels.append("reset")
        return labels

    def build_action(
        self,
        frame: FrameData,
        action_label: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> GameAction:
        normalized = action_label.strip().lower()
        if normalized not in ACTION_BY_LABEL:
            raise ValueError(f"Unknown action: {action_label}")
        if normalized == "reset":
            return GameAction.RESET
        available = self.available_action_labels(frame)
        if normalized not in available:
            raise ValueError(
                f"Action '{normalized}' not in available_actions={available}"
            )
        action = ACTION_BY_LABEL[normalized]
        if normalized == "click":
            if x is None or y is None:
                raise ValueError("click requires x and y")
            if not (0 <= int(x) <= 63 and 0 <= int(y) <= 63):
                raise ValueError("click x/y must be within 0..63")
            action.set_data({"x": int(x), "y": int(y)})
        return action
