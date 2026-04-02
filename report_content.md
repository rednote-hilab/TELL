# TELL: Test-time Experiential Lifelong Learning

A single LLM agent that learns from experience at test time — no fine-tuning, just autonomous hypothesis-driven memory updates.


## Opening

For decades, the promise of general intelligence has included one deceptively simple property: the ability to get better through experience. Not through retraining. Not through fine-tuning. Simply by doing — autonomously observing, forming beliefs, testing them, and updating, with no external intervention.

Current LLMs don't do this. Each inference is stateless. Each episode begins from the same weights, the same prompt. There is no mechanism for experience to compound.

TELL is our early attempt to change that. With **no training, no fine-tuning, and no task-specific engineering**, we ask a simple question: can a single LLM agent learn from experience at test time — forming hypotheses, verifying them through interaction, and genuinely improving across episodes? We see early evidence that it can. We believe this is an early spark of what lifelong learning in large language models could look like.

## Score vs Cost

Unverified results on the same 25-game offline evaluation. CoT baselines score below 1% regardless of cost.

[Agentica](https://www.symbolica.ai/blog/arc-agi-3) (Symbolica) uses a multi-agent SDK with orchestrated agent coordination. TELL uses a **single LLM conversation** — one agent, three tools (`screen_shot`, `bash_exec`, `write_file`), one persistent memory file — with no multi-agent scaffolding, no task-specific modules, and **no training of any kind**. The 42.7% score is the current result of this framework running on Claude Opus 4.6; TELL itself is model-agnostic.

## The Missing Loop

LLM agents can reason, plan, and use tools — but they don't get better at a task by doing more of it. Each episode starts from scratch. There is no learning loop. Humans improve differently: each attempt updates a mental model of how the world works, and that model persists.

TELL asks whether an LLM agent can do the same — not through training, but at test time. As the agent interacts with an unknown environment, it doesn't just try to solve the current task. It actively reverse-engineers the hidden rules of the system: forming hypotheses about what each element does, testing them through action, and committing verified knowledge to memory. That memory persists across episodes, carrying an evolving world model into every future decision.

The critical question is not whether the agent can solve a single task. It's whether early discoveries compound into later performance. That cross-episode improvement is what TELL is designed to produce, and what we observe.

## Why ARC-AGI-3

ARC-AGI is the benchmark series most widely adopted by frontier AI labs to measure the gap between human and artificial intelligence. Created by François Chollet and maintained by the ARC Prize Foundation, it is designed to evaluate fluid adaptive intelligence — the ability to generalize to genuinely novel tasks, not to recall memorized patterns. It has repeatedly identified key inflection points in AI progress, from the emergence of reasoning systems to the rise of capable agents.

ARC-AGI-3, released in March 2026, marks the series' first shift from static puzzles to interactive environments. Agents must explore, infer goals, build world models, and adapt — all without instructions or prior exposure. Humans solve 100% of environments. Frontier LLMs score below 1% (GPT-5.4 at 0.3%, Claude Opus 4.6 at 0.2%, Gemini 3.1 Pro at 0.2%, Grok 4.20 at 0%). The benchmark measures not just whether a goal is reached, but how efficiently — scoring action efficiency against human baselines.

We chose ARC-AGI-3 because it demands exactly what TELL is designed for: rapid hypothesis formation, iterative testing, and knowledge accumulation across episodes, all without task-specific scaffolding. ARC-AGI-3 is a starting point, not a constraint. The underlying mechanism — an agent that reverse-engineers the rules of an unknown system through experience at test time — is not specific to any particular domain. Any environment where understanding deepens with interaction is a candidate.

## The Mechanism in Action

The most striking pattern we observe is this: the agent's breakthroughs are never about trying harder. They are about modeling differently. In one environment, it realizes that obstacles aren't just blockers — they permanently deform shapes, making previously unreachable targets accessible. In another, it discovers that a visual panel is actually a command language with three distinct primitives. Each insight is written to memory and immediately reshapes how the agent approaches every subsequent episode.

These are not lucky guesses. They are the result of accumulated context: prior failures, partial confirmations, and growing structural understanding that no single-episode agent could have reached. This is what test-time experiential learning looks like in practice.

## Showcase — 3 games

### 01. tu93 — Grid Maze: Three enemies, three models, one taxonomy
Starting from zero knowledge, the agent reverse-engineers **three distinct enemy types** through observation: Orange patrols that bounce at dead ends, Red guards destroyable via perpendicular entry, and Maroon chasers that activate on proximity and maintain distance 2. Each discovery compounds into the next level's strategy.
> "Orange: mobile patrol, reverses at dead ends. Red: stationary guard — enter OWN cell from PERPENDICULAR = destroy. Maroon: activates into CHASER, moves 1 cell/turn toward player."

### 02. g50t — Compression Puzzle: SPACE is not placement — it's ejection
Agent initially assumes SPACE places a piece. Then discovers it does the **exact opposite**: ejects the piece backward along its entire path to start, planting a delayed compression seed that matures into a new passage. The entire puzzle's world model flips.
> "Space inside colored cell: EJECTS the piece backward along its ENTIRE path to start. Leaves a delayed compression seed."

### 03. su15 — Block + Hazard Puzzle: One radius rule unifies everything
Instead of treating each click effect as a special case, the agent extracts a **unified mechanic**: 8-cell Euclidean radius determines teleport, merge, upgrade, or conflict.
> "If within 8 Euclidean there is 1 block → teleport. 2+ same-type → merge. Different-type → conflict/fail."

## Cost Analysis

Each game was run **exactly once** — no cherry-picking, no reruns, no ensembling. The agent runs as a single multi-turn conversation with Claude Opus 4.6 (128K context, streaming, adaptive thinking). Total: **8,053 LLM requests** across 25 games, **430M input** + **32M output** tokens. Because each request's input largely overlaps with the previous one, **prompt caching** reduces cost dramatically: 91.3% of input tokens are cache reads at 1/10 the base price.

Without caching, the same experiment would cost $2,955.54 (input at $5/MTok). Prompt caching saves 58% — conversational agents naturally reuse most of their context each turn. Average cost per game: $49.39.

| Category | Tokens | Rate | Cost |
| --- | --- | --- | --- |
| Cache write (8.7%) | 37,646,356 | $6.25 / MTok | $235.29 |
| Cache read (91.3%) | 392,850,345 | $0.50 / MTok | $196.43 |
| Output | 32,122,135 | $25.00 / MTok | $803.05 |
| Total | 462M |  | $1,234.77 |

## After the Spark

These results use zero training. What to retain, what to discard, how to restructure knowledge as understanding deepens — all of this is emergent behavior from pretrained capabilities, not a learned skill.

The next frontier is teaching models to explicitly learn how to update memory itself. Through RL on long, multi-episode trajectories, the model would learn to treat memory update as a core reasoning skill — when to generalize a pattern, when to commit a hypothesis, when to discard a belief that no longer fits. Rather than hoping that good memory management emerges, we aim to optimize for it directly.

TELL shows that the spark is there. What comes next is a model that knows how to keep it alive.
