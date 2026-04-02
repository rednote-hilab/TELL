from __future__ import annotations

import json
import logging
import os
from threading import Thread
from typing import TYPE_CHECKING, Optional, Type

from pathlib import Path

from arc_agi import Arcade, OperationMode
from arc_agi.scorecard import EnvironmentScorecard

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger()


class Swarm:
    """Orchestration for many agents playing many ARC-AGI-3 games."""

    GAMES: list[str]
    ROOT_URL: str
    COUNT: int
    agent_name: str
    agent_class: Type[Agent]
    threads: list[Thread]
    agents: list[Agent]
    record_games: list[str]
    cleanup_threads: list[Thread]
    headers: dict[str, str]
    card_id: Optional[str]
    _arc: Arcade

    def __init__(
        self,
        agent: str,
        ROOT_URL: str,
        games: list[str],
        tags: list[str] = [],
        local: bool = False,
        env_dir: Optional[str] = None,
    ) -> None:
        from . import AVAILABLE_AGENTS

        self.GAMES = games
        self.ROOT_URL = ROOT_URL
        self.agent_name = agent
        self.agent_class = AVAILABLE_AGENTS[agent]
        self.threads = []
        self.agents = []
        self.cleanup_threads = []
        self.local = local
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        self.tags = tags.copy() if tags is not None else []
        self._arc = self._create_arc(local=local, env_dir=env_dir)

        # Set up base tags for tracing
        if self.agent_name.endswith(".recording.jsonl"):
            # Extract GUID from playback filename
            # Format: game.agent.count.guid.recording.jsonl
            parts = self.agent_name.split(".")
            guid = parts[-3] if len(parts) >= 4 else "unknown"
            self.tags.extend(["playback", guid])
        else:
            self.tags.extend(["agent", self.agent_name])

    def main(self) -> EnvironmentScorecard | None:
        """The main orchestration loop, continues until all agents are done."""

        # submit start of scorecard
        if self.local:
            print("***** LOCAL MODE: SCORECARD DISABLED")
            self.card_id = ""
        else:
            print("***** MAKING SCORECARD")
            self.card_id = self.open_scorecard()

        print(f"***** MAKING ALL AGENTS with card id: {self.card_id or 'N/A'}")
        # create all the agents
        for i in range(len(self.GAMES)):
            g = self.GAMES[i % len(self.GAMES)]
            a = self.agent_class(
                card_id=self.card_id,
                game_id=g,
                agent_name=self.agent_name,
                ROOT_URL=self.ROOT_URL,
                record=True,
                arc_env=self._arc.make(g, scorecard_id=self.card_id or None),
                tags=self.tags,
            )
            self.agents.append(a)

        # create all the threads
        for a in self.agents:
            self.threads.append(Thread(target=a.main, daemon=True))

        # start all the threads
        for t in self.threads:
            t.start()

        # wait for all agent to finish
        for t in self.threads:
            t.join()

        # all agents are now done
        card_id = self.card_id
        scorecard = None if self.local else self.close_scorecard(card_id)
        if scorecard:
            logger.info("--- FINAL SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))

        # Provide web link to scorecard
        if card_id:
            if self._arc.operation_mode == OperationMode.ONLINE:
                scorecard_url = f"{self.ROOT_URL}/scorecards/{card_id}"
                logger.info(f"View your scorecard online: {scorecard_url}")
            else:
                logger.info(
                    "Online scorecard is not available, to use the online API set the ONLINE_ONLY envvar to True"
                )

        self.cleanup(scorecard)

        return scorecard

    @staticmethod
    def _default_env_dir() -> str:
        repo_root = Path(__file__).resolve().parents[2]
        default_env_dir = repo_root.parent / "ARC-AGI" / "environment_files"
        if default_env_dir.exists():
            return str(default_env_dir)
        return str(repo_root.parent / "ARC-AGI" / "test_environment_files")

    @classmethod
    def _create_arc(cls, local: bool, env_dir: Optional[str]) -> Arcade:
        if local:
            resolved_env_dir = env_dir or cls._default_env_dir()
            return Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=resolved_env_dir,
            )
        return Arcade()

    @classmethod
    def discover_games(
        cls,
        root_url: str,
        headers: dict[str, str],
        local: bool = False,
        env_dir: Optional[str] = None,
    ) -> list[str]:
        if local:
            arc = cls._create_arc(local=True, env_dir=env_dir)
            return [g.game_id for g in arc.get_environments()]

        import requests

        try:
            with requests.Session() as session:
                session.headers.update(headers)
                r = session.get(f"{root_url}/api/games", timeout=10)

            if r.status_code == 200:
                return [g["game_id"] for g in r.json()]
            logger.error(f"API request failed with status {r.status_code}: {r.text[:200]}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to API server: {e}")
            return []

    def open_scorecard(self) -> str:
        if self.local:
            return ""
        return self._arc.open_scorecard(tags=self.tags)  # type: ignore[no-any-return]

    def close_scorecard(self, card_id: str) -> Optional[EnvironmentScorecard]:
        if self.local:
            return None
        self.card_id = None

        return self._arc.close_scorecard(card_id)

    def cleanup(self, scorecard: Optional[EnvironmentScorecard] = None) -> None:
        """Cleanup all agents."""
        for a in self.agents:
            a.cleanup(scorecard)
        if hasattr(self, "_session"):
            self._session.close()
