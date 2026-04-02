# TELL — Test-time Experiential Lifelong Learning

A single LLM agent that learns from experience at test time — no fine-tuning, no training, no task-specific engineering. Just autonomous hypothesis-driven memory updates within one conversation.

TELL achieves **43.9%** on [ARC-AGI-3](https://arcprize.org/) with a single Claude Opus 4.6 conversation per game, using three tools and one persistent memory file. The framework is model-agnostic: the score reflects the current result of this framework running on Claude Opus 4.6.

## Results

| Method | Type | Score | Cost |
|--------|------|-------|------|
| **TELL (Ours)** | Single-agent, training-free | **43.9%** | **$1,406** |
| Agentica (Symbolica) | Multi-agent SDK | 36.08% | $1,005 |
| GPT-5.4 (CoT) | Chain-of-thought | 0.3% | $5,200 |
| Claude Opus 4.6 (CoT) | Chain-of-thought | 0.2% | $8,900 |
| Gemini 3.1 Pro (CoT) | Chain-of-thought | 0.2% | $2,200 |
| Grok 4.20 (CoT) | Chain-of-thought | 0.0% | $3,800 |

- **25 games** tested in offline mode, each run exactly once — no cherry-picking, no reruns, no ensembling
- **11/25 games won** (all levels cleared)
- Total API cost: **$1,406** with prompt caching (58% savings vs uncached)

See [the report](https://rednote-hilab.github.io/TELL/) for detailed per-game results, agent trajectories, and learned world memories.

## How It Works

TELL is not a multi-agent system. It runs a **single LLM conversation** per game with three tools:

| Tool | Purpose |
|------|---------|
| `screen_shot` | Observe the current game state |
| `bash_exec` | Run Python scripts to analyze frames, execute actions |
| `write_file` | Write to `MEMORY.md` — the agent's persistent world memory |

The learning loop:

1. **Observe** — Take a screenshot of the game state
2. **Hypothesize** — Form theories about game mechanics from visual evidence
3. **Test** — Execute actions to validate or falsify hypotheses
4. **Learn** — Update `MEMORY.md` with confirmed rules, patterns, and strategies
5. **Reuse** — Apply learned knowledge to new levels, adapting only what changed

The key insight: the agent builds a **world model** within `MEMORY.md` that accumulates across levels. When a new level appears, it doesn't start from scratch — it compares against solved levels and reuses stable rules, only re-exploring what's genuinely new.

## Architecture

```
main.py                    # Entry point — launches one game
run_all_tasks.sh           # Parallel launcher for multiple games
agents/
  agent.py                 # Base Agent class
  game_bridge.py           # Action label mapping
  swarm.py                 # Multi-game orchestrator
  recorder.py              # Game recording/playback
  tell_agent/
    agent.py               # TELLAgent — main agent class (dual-thread: game + LLM)
    state_machine.py       # LLM conversation loop with tool dispatch
    compaction.py          # Context window management (compaction at 80% usage)
    tool_handlers.py       # bash_exec, write_file, screen_shot implementations
    claude_client.py       # Claude API client (streaming, via official Anthropic SDK)
    config.py              # YAML config loader + env var helpers
    tell_agent.yaml        # Default agent configuration
    configs/
      offline.yaml         # Baseline experiment config (25 games)
      prior_memory.yaml    # Prior memory experiment config
    prior_memory/          # Cleaned world memories from baseline runs (25 games)
      CLEANING_PROMPT.md   # Instructions for how memories were cleaned
      <game_id>_MEMORY.md  # Per-game cleaned memory files
```

## Setup

### Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- ARC-AGI-3 game environments (included as submodule)

### Install

```bash
git clone --recurse-submodules https://github.com/rednote-hilab/TELL.git
cd TELL

# Install dependencies
uv sync --dev
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Configure

Copy `.env.example` to `.env` and set your Anthropic API key:

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Run a single game

```bash
uv run python main.py -a tell_agent -g <game_id> -m offline
```

### Run all 25 games (baseline)

```bash
bash run_all_tasks.sh --config agents/tell_agent/configs/offline.yaml
```

### Run with prior memory

The prior memory experiment injects cleaned world knowledge from a completed baseline run into a fresh attempt, testing whether the agent's learned world models are transferable.

```bash
bash run_all_tasks.sh --config agents/tell_agent/configs/prior_memory.yaml
```

The cleaned memories in `agents/tell_agent/prior_memory/` were produced by taking the final `MEMORY.md` from each baseline game and stripping level-specific data (coordinates, solutions, action sequences) with an LLM — **delete only, never rewrite**. See `CLEANING_PROMPT.md` in that directory for the exact cleaning instructions.

On co-completed levels, prior memory reduces total actions by ~51%. See the report for details.

## Configuration

See `agents/tell_agent/configs/offline.yaml` for the experiment config. Key settings:

| Setting | Description |
|---------|-------------|
| `env.TELL_AGENT_MODEL` | Model name (e.g., `claude-opus-4-6`) |
| `env.CLAUDE_THINKING_TYPE` | Thinking mode: `enabled`, `disabled`, or `adaptive` |
| `env.MAX_ACTIONS` | Max game actions per run (2048) |
| `env.PRIOR_MEMORY_DIR` | Directory with cleaned memories (prior memory experiment only) |
| `run.parallel_jobs` | Number of games to run in parallel |
| `compaction.max_context_tokens` | Context window size in tokens |
| `compaction.trigger_ratio` | Trigger compaction at this ratio of context usage |

## License

MIT
