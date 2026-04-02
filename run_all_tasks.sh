#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_CFG_FILE="${SCRIPT_DIR}/agents/tell_agent/tell_agent.yaml"
CFG_FILES=()
REPLAY_PREFIX=""
GAMES_OVERRIDE=""
ALL_GAMES=0
PARALLEL_OVERRIDE=""
RUNS_OVERRIDE=""

if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON_CMD=("${SCRIPT_DIR}/.venv/bin/python")
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  echo "ERROR: neither 'uv', 'python', nor 'python3' is available on PATH"
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CFG_FILES+=("$2")
      shift 2
      ;;
    --configs)
      IFS=',' read -r -a _cfgs <<< "$2"
      for _cfg in "${_cfgs[@]}"; do
        _cfg="${_cfg#"${_cfg%%[![:space:]]*}"}"
        _cfg="${_cfg%"${_cfg##*[![:space:]]}"}"
        [[ -n "$_cfg" ]] && CFG_FILES+=("$_cfg")
      done
      shift 2
      ;;
    --games)
      GAMES_OVERRIDE="$2"
      shift 2
      ;;
    --all)
      ALL_GAMES=1
      shift
      ;;
    --parallel-jobs)
      PARALLEL_OVERRIDE="$2"
      shift 2
      ;;
    --runs)
      RUNS_OVERRIDE="$2"
      shift 2
      ;;
    --replay-prefix)
      REPLAY_PREFIX="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      echo "Usage: $0 [--config <path>]... [--configs <a,b,c>] [--games <g1,g2|all>] [--all] [--runs <n>] [--parallel-jobs <n>] [--replay-prefix <prefix>]"
      exit 1
      ;;
  esac
done

if (( ${#CFG_FILES[@]} == 0 )); then
  CFG_FILES=("$DEFAULT_CFG_FILE")
fi

resolve_config_path() {
  local cfg="$1"
  if [[ "$cfg" != /* ]]; then
    cfg="${SCRIPT_DIR}/${cfg}"
  fi
  printf '%s\n' "$cfg"
}

load_config_env_file() {
  local cfg_file="$1"
  local env_file="$2"
  "${PYTHON_CMD[@]}" - "$cfg_file" >"$env_file" <<'PY'
import json
import shlex
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

run = cfg.get("run") or {}
env = cfg.get("env") or {}

def q(x):
    return shlex.quote("" if x is None else str(x))

print(f"RUNS={q(run.get('runs', 1))}")
print(f"AGENT={q(run.get('agent', 'tell_agent'))}")
config_name = cfg_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
print(f"CONFIG_BASENAME={q(config_name)}")
game = run.get("game", "ls20")
if isinstance(game, list):
    game = ",".join(str(x) for x in game if str(x).strip())
if isinstance(game, str) and game.strip().lower() == "all":
    game = ""
print(f"GAME={q(game)}")
print(f"ENV_DIR={q(run.get('env_dir', '../ARC-AGI/environment_files'))}")
print(f"TAGS={q(run.get('tags', 'local,dev'))}")
print(f"PARALLEL_JOBS={q(run.get('parallel_jobs', 1))}")

# Export every env key from YAML env: so runtime no longer depends on .env.
import re
for k, v in (env.items() if isinstance(env, dict) else []):
    if not isinstance(k, str):
        continue
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k):
        continue
    print(f"export {k}={q(v)}")
PY
}

resolve_games() {
  local game_spec="$1"
  local env_dir="$2"
  if [[ -n "$game_spec" ]]; then
    local normalized="$game_spec"
    normalized="${normalized//,/ }"
    read -r -a _games <<< "$normalized"
    for g in "${_games[@]}"; do
      g="${g// /}"
      [[ -n "$g" ]] && echo "$g"
    done
    return
  fi
  # all games (offline env dir)
  find "$env_dir" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort
}

has_completed_run() {
  local cfg_name="$1"
  local run_i="$2"
  local game_id="$3"
  local log_dir="${LOG_DIR:-}"

  [[ -n "$log_dir" ]] || return 1
  "${PYTHON_CMD[@]}" - "$log_dir" "$AGENT" "$cfg_name" "$game_id" "$run_i" <<'PY'
from pathlib import Path
import json
import sys

log_dir = Path(sys.argv[1])
agent = sys.argv[2]
cfg_name = sys.argv[3]
game_id = sys.argv[4]
run_i = sys.argv[5]

if not log_dir.is_dir():
    raise SystemExit(1)

pattern = f"{agent}-{cfg_name}-{game_id}-*-run{run_i}.log"
for path in sorted(log_dir.glob(pattern)):
    try:
        if "[run_end]" in path.read_text(encoding="utf-8", errors="ignore"):
            raise SystemExit(0)
    except OSError:
        continue

# Resume/fresh runs can also write logs named only by run_id, e.g. run_<game-hash>_<uuid>.log.
# Fall back to replay manifests in this LOG_DIR so completed resumed runs are not re-launched.
replays_dir = log_dir / "replays"
if replays_dir.is_dir():
    for d in sorted(replays_dir.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        m_game_id = str(manifest.get("game_id", ""))
        if not m_game_id.startswith(game_id):
            continue
        if str(manifest.get("closed_at", "")).strip():
            raise SystemExit(0)

raise SystemExit(1)
PY
}

has_won_any_run() {
  local cfg_name="$1"
  local game_id="$2"
  local log_dir="${LOG_DIR:-}"

  [[ -n "$log_dir" ]] || return 1
  "${PYTHON_CMD[@]}" - "$log_dir" "$AGENT" "$cfg_name" "$game_id" <<'PY'
from pathlib import Path
import sys

log_dir = Path(sys.argv[1])
agent = sys.argv[2]
cfg_name = sys.argv[3]
game_id = sys.argv[4]

if not log_dir.is_dir():
    raise SystemExit(1)

pattern = f"{agent}-{cfg_name}-{game_id}-*-run*.log"
for path in sorted(log_dir.glob(pattern)):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "[run_end]" in text and "state=GameState.WIN" in text:
            raise SystemExit(0)
    except OSError:
        continue

raise SystemExit(1)
PY
}

find_incomplete_run() {
  local cfg_name="$1"
  local run_i="$2"
  local game_id="$3"
  local log_dir="${LOG_DIR:-}"

  [[ -n "$log_dir" ]] || return 1
  "${PYTHON_CMD[@]}" - "$log_dir" "$cfg_name" "$run_i" "$game_id" <<'PY'
import json, sys
from pathlib import Path

log_dir = Path(sys.argv[1])
cfg_name = sys.argv[2]
run_i = str(sys.argv[3])
game_id = sys.argv[4]

replays_dir = log_dir / "replays"
if not replays_dir.is_dir():
    raise SystemExit(1)

# Scan all replay dirs, match by game_id prefix in manifest
candidates = []
for d in sorted(replays_dir.iterdir()):
    if not d.is_dir():
        continue
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        continue
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    # Match game by prefix (e.g., "dc22" matches "dc22-4c9bff3e")
    m_game_id = str(manifest.get("game_id", ""))
    if not m_game_id.startswith(game_id):
        continue
    # Incomplete = closed_at is empty string
    if str(manifest.get("closed_at", "")).strip():
        continue
    # Must have some progress (at least a few messages)
    if int(manifest.get("message_count", 0) or 0) < 2:
        continue
    run_id = str(manifest.get("run_id", "") or "")
    exact = (
        f"-{cfg_name}-{game_id}-" in run_id
        and run_id.endswith(f"-run{run_i}")
    )
    candidates.append((0 if exact else 1, d, manifest))

if not candidates:
    raise SystemExit(1)

# Prefer exact cfg/run matches when available, else fall back to older broad matching.
candidates.sort(
    key=lambda x: (
        x[0],
        -int(x[2].get("action_frame_count", 0) or 0),
        -int(x[2].get("message_count", 0) or 0),
    )
)
best_dir = candidates[0][1]
best_run_id = str(candidates[0][2].get("run_id", ""))
# Output: replay_dir \t run_id
print(f"{best_dir}\t{best_run_id}")
raise SystemExit(0)
PY
}

run_one_game() {
  local cfg_name="$1"
  local run_i="$2"
  local game_id="$3"
  local ts
  local run_tags
  local args
  local resume_info
  local resume_dir
  local resume_run_id

  echo "=== Config=$cfg_name | Run $run_i/$RUNS | Game=$game_id ==="

  # Check for incomplete run to resume
  unset RESUME_LOG_DIR 2>/dev/null || true
  if resume_info="$(find_incomplete_run "$cfg_name" "$run_i" "$game_id" 2>/dev/null)"; then
    resume_dir="$(printf '%s' "$resume_info" | cut -f1)"
    resume_run_id="$(printf '%s' "$resume_info" | cut -f2)"
    echo "  -> Resuming from: $resume_run_id (dir: $resume_dir)"
    export LOG_RUN_ID="$resume_run_id"
    export RESUME_LOG_DIR="$resume_dir"
  else
    if [[ -n "$REPLAY_PREFIX" ]]; then
      export LOG_RUN_ID="${REPLAY_PREFIX}-${cfg_name}-run${run_i}-${game_id}"
    else
      ts="$(date +%Y%m%d-%H%M%S)"
      export LOG_RUN_ID="${AGENT}-${cfg_name}-${game_id}-${ts}-run${run_i}"
    fi
  fi

  run_tags="$TAGS,config_${cfg_name},run_${run_i},game_${game_id}"
  if [[ -n "${ENV_DIR:-}" && -d "${ENV_DIR:-}" ]]; then
    args=(--local --env-dir "$ENV_DIR" -a "$AGENT" -t "$run_tags" -g "$game_id")
  else
    args=(-a "$AGENT" -t "$run_tags" -g "$game_id")
  fi
  echo "CONFIG=$cfg_name RUN=$run_i AGENT=$AGENT GAME=$game_id PARALLEL_JOBS=$PARALLEL_JOBS"
  "${PYTHON_CMD[@]}" main.py "${args[@]}"
}

wait_for_any_pid() {
  local pid
  while true; do
    for pid in "$@"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
        return $?
      fi
    done
    sleep 0.2
  done
}

run_config_batch() {
  local cfg_file="$1"
  (
    local env_exports_file
    local resolved_cfg
    local cfg_name
    local i
    local game_id
    local pid
    local -a resolved_games=()
    local -a pids=()

    resolved_cfg="$(resolve_config_path "$cfg_file")"
    if [[ ! -f "$resolved_cfg" ]]; then
      echo "ERROR: config file not found: $resolved_cfg"
      exit 1
    fi

    env_exports_file="$(mktemp)"
    load_config_env_file "$resolved_cfg" "$env_exports_file"
    # shellcheck disable=SC1090
    source "$env_exports_file"
    rm -f "$env_exports_file"
    export TELL_AGENT_CONFIG_PATH="$resolved_cfg"
    cfg_name="${CONFIG_BASENAME:-$(basename "$resolved_cfg" .yaml)}"

    if [[ -n "$RUNS_OVERRIDE" ]]; then
      RUNS="$RUNS_OVERRIDE"
    fi
    if [[ -n "$PARALLEL_OVERRIDE" ]]; then
      PARALLEL_JOBS="$PARALLEL_OVERRIDE"
    fi

    if [[ "$ALL_GAMES" == "1" ]]; then
      GAME=""
    elif [[ -n "$GAMES_OVERRIDE" ]]; then
      if [[ "$(printf '%s' "$GAMES_OVERRIDE" | tr '[:upper:]' '[:lower:]')" == "all" ]]; then
        GAME=""
      else
        GAME="$GAMES_OVERRIDE"
      fi
    fi

    while IFS= read -r game_id; do
      [[ -n "$game_id" ]] && resolved_games+=("$game_id")
    done < <(resolve_games "${GAME:-}" "$ENV_DIR")

    if (( ${#resolved_games[@]} == 0 )); then
      echo "ERROR: no games resolved from GAME='${GAME:-all}' and ENV_DIR='$ENV_DIR' for config '$resolved_cfg'"
      exit 1
    fi

    echo "=== Config batch: $resolved_cfg ==="

    for i in $(seq 1 "$RUNS"); do
      if (( PARALLEL_JOBS <= 1 )); then
        for game_id in "${resolved_games[@]}"; do
          if has_won_any_run "$cfg_name" "$game_id"; then
            echo "=== Config=$cfg_name | Run $i/$RUNS | Game=$game_id | Skip (already won) ==="
            continue
          fi
          if has_completed_run "$cfg_name" "$i" "$game_id"; then
            echo "=== Config=$cfg_name | Run $i/$RUNS | Game=$game_id | Skip existing completed run ==="
            continue
          fi
          run_one_game "$cfg_name" "$i" "$game_id"
        done
      else
        pids=()
        for game_id in "${resolved_games[@]}"; do
          if has_won_any_run "$cfg_name" "$game_id"; then
            echo "=== Config=$cfg_name | Run $i/$RUNS | Game=$game_id | Skip (already won) ==="
            continue
          fi
          if has_completed_run "$cfg_name" "$i" "$game_id"; then
            echo "=== Config=$cfg_name | Run $i/$RUNS | Game=$game_id | Skip existing completed run ==="
            continue
          fi
          run_one_game "$cfg_name" "$i" "$game_id" &
          pids+=("$!")
          if (( ${#pids[@]} >= PARALLEL_JOBS )); then
            wait_for_any_pid "${pids[@]}"
            next_pids=()
            for pid in "${pids[@]}"; do
              if kill -0 "$pid" 2>/dev/null; then
                next_pids+=("$pid")
              fi
            done
            pids=("${next_pids[@]}")
          fi
        done
        for pid in "${pids[@]}"; do
          wait "$pid"
        done
      fi
    done
  )
}

for cfg in "${CFG_FILES[@]}"; do
  run_config_batch "$cfg"
done
