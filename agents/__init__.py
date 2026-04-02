from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .tell_agent import TELLAgent
from .recorder import Recorder
from .swarm import Swarm

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

AVAILABLE_AGENTS["tell_agent"] = TELLAgent
AVAILABLE_AGENTS["tellagent"] = TELLAgent

__all__ = [
    "Swarm",
    "Agent",
    "TELLAgent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
