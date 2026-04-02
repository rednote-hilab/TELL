from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TransitionRule:
    type: str
    to_stage: str
    pattern: str = ""
    regex: bool = False
    ignore_case: bool = False


@dataclass
class StageNode:
    name: str
    tools: List[str]
    system_template: str
    user_template: str
    transitions: List[TransitionRule]
    resume: bool = False
    resume_policy: "ResumePolicy" | None = None


@dataclass
class ResumePolicy:
    on_level_up: str = "compact"
    on_action_submitted: str = "keep"
    on_context_limit: str = "compact"


class StageWorkflow:
    def __init__(self, entry_stage: str, stages: Dict[str, StageNode]) -> None:
        if not entry_stage:
            raise ValueError("workflow entry_stage is required")
        if entry_stage not in stages:
            raise ValueError(f"workflow entry_stage not found: {entry_stage}")
        self.entry_stage = entry_stage
        self.stages = stages

    def get_stage(self, name: str) -> StageNode:
        if name not in self.stages:
            raise ValueError(f"unknown workflow stage: {name}")
        return self.stages[name]

    def resolve_next_stage(self, stage_name: str, model_text: str) -> str:
        stage = self.get_stage(stage_name)
        text = model_text or ""
        for rule in stage.transitions:
            rtype = (rule.type or "").strip().lower()
            if rtype == "default":
                return rule.to_stage
            if rtype == "match":
                if _match_text(text, rule.pattern, regex=rule.regex, ignore_case=rule.ignore_case):
                    return rule.to_stage
        return stage_name


def build_stage_workflow(prompts_cfg: Dict[str, Any]) -> StageWorkflow:
    workflow_cfg = prompts_cfg.get("workflow")
    if not isinstance(workflow_cfg, dict):
        raise ValueError("prompts.workflow is required")

    entry = str(workflow_cfg.get("entry_stage") or "").strip()
    stages_cfg = workflow_cfg.get("stages")
    if not isinstance(stages_cfg, dict):
        raise ValueError("prompts.workflow.stages must be a mapping")
    stage_prompts = prompts_cfg.get("stages")
    if not isinstance(stage_prompts, dict):
        stage_prompts = {}

    stages: Dict[str, StageNode] = {}
    for name, raw in stages_cfg.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            continue
        prompt_ref = str(raw.get("prompt_ref") or name).strip()
        prompt_block = stage_prompts.get(prompt_ref)
        if not isinstance(prompt_block, dict):
            prompt_block = {}
        system_template = str(raw.get("system") or prompt_block.get("system") or "").strip()
        user_template = str(raw.get("user") or prompt_block.get("user") or "").strip()
        if not system_template:
            raise ValueError(f"workflow stage '{name}' missing system prompt")
        if not user_template:
            raise ValueError(f"workflow stage '{name}' missing user prompt")

        tools_raw = raw.get("tools")
        tools: List[str] = []
        if isinstance(tools_raw, list):
            tools = [str(t).strip() for t in tools_raw if str(t).strip()]
        transitions = _parse_transitions(raw.get("transitions"), stage_name=name)
        stages[name] = StageNode(
            name=name,
            tools=tools,
            system_template=system_template,
            user_template=user_template,
            transitions=transitions,
            resume=bool(raw.get("resume", False)),
            resume_policy=_parse_resume_policy(raw.get("resume_policy")),
        )

    if not stages:
        raise ValueError("workflow has no valid stages")
    return StageWorkflow(entry_stage=entry, stages=stages)


def _parse_transitions(raw: Any, stage_name: str) -> List[TransitionRule]:
    if not isinstance(raw, list):
        raise ValueError(f"workflow stage '{stage_name}' transitions must be a list")
    out: List[TransitionRule] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"workflow stage '{stage_name}' transition[{i}] must be a mapping")
        t = str(item.get("type") or "").strip().lower()
        to_stage = str(item.get("to") or "").strip()
        if t not in {"default", "match"}:
            raise ValueError(f"workflow stage '{stage_name}' transition[{i}] has invalid type: {t!r}")
        if not to_stage:
            raise ValueError(f"workflow stage '{stage_name}' transition[{i}] missing 'to'")
        pattern = str(item.get("pattern") or item.get("contains") or "").strip()
        if t == "match" and not pattern:
            raise ValueError(f"workflow stage '{stage_name}' transition[{i}] missing pattern/contains")
        out.append(
            TransitionRule(
                type=t,
                to_stage=to_stage,
                pattern=pattern,
                regex=bool(item.get("regex", False)),
                ignore_case=bool(item.get("ignore_case", False)),
            )
        )
    if not out:
        raise ValueError(f"workflow stage '{stage_name}' must define at least one transition")
    return out


def _parse_resume_policy(raw: Any) -> ResumePolicy:
    if not isinstance(raw, dict):
        return ResumePolicy()

    return ResumePolicy(
        on_level_up=_normalize_resume_action(raw.get("on_level_up"), "compact"),
        on_action_submitted=_normalize_resume_action(raw.get("on_action_submitted"), "keep"),
        on_context_limit=_normalize_resume_action(raw.get("on_context_limit"), "compact", allow_keep=False),
    )


def _normalize_resume_action(value: Any, default: str, *, allow_keep: bool = True) -> str:
    text = str(value or "").strip().lower()
    allowed = {"compact", "clear"}
    if allow_keep:
        allowed.add("keep")
    if text in allowed:
        return text
    return default


def _match_text(text: str, pattern: str, *, regex: bool, ignore_case: bool) -> bool:
    if not pattern:
        return False
    if regex:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            return re.search(pattern, text, flags) is not None
        except re.error:
            return False
    if ignore_case:
        return pattern.lower() in text.lower()
    return pattern in text
