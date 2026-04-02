from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)
    except (HTTPError, URLError, TimeoutError) as exc:
        return {"ok": False, "error": f"http_error:{type(exc).__name__}", "detail": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"decode_error:{type(exc).__name__}", "detail": str(exc)}


def get_observation(api: str) -> Dict[str, Any]:
    return _http_json("GET", f"{api}/observation")


def post_action(api: str, action: str, x: Any = None, y: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"action": action}
    if x is not None:
        payload["x"] = x
    if y is not None:
        payload["y"] = y
    return _http_json("POST", f"{api}/action", payload=payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--max-actions", type=int, default=64)
    ap.add_argument("--log", default="tmp/policy_run.jsonl")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    api = str(args.api).rstrip("/")
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    preferred = ["right", "down", "left", "up"]
    policy_md_path = Path("memory/policy/MEMORY.md")
    if policy_md_path.exists():
        policy_md = policy_md_path.read_text(encoding="utf-8")
        if "prefer:" in policy_md.lower():
            for line in policy_md.splitlines():
                if line.lower().strip().startswith("prefer:"):
                    raw = line.split(":", 1)[1].strip()
                    seq = [x.strip().lower() for x in raw.split(",") if x.strip()]
                    if seq:
                        preferred = seq
                    break

    for i in range(int(args.max_actions)):
        obs = get_observation(api)
        actions = obs.get("observation", {}).get("available_actions", []) if isinstance(obs, dict) else []
        latest_frame = ""
        try:
            latest_frame = str(obs.get("observation", {}).get("frames", [])[-1])
        except Exception:
            latest_frame = ""
        rows = latest_frame.splitlines() if latest_frame else []
        action_name = next((a for a in preferred if a in actions), actions[0] if actions else "")
        rec: Dict[str, Any] = {
            "i": i,
            "obs": {
                "ok": obs.get("ok"),
                "step": obs.get("step"),
                "format": obs.get("format"),
                "ascii_h": len(rows),
                "ascii_w": (len(rows[0]) if rows else 0),
            },
        }
        if not action_name:
            rec.update({"status": "no_available_actions"})
            log_path.open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=True) + "\n")
            return 2
        res = post_action(api, action_name)
        rec.update(
            {
                "action": {"name": action_name},
                "result": {"ok": res.get("ok"), "step": res.get("step"), "error": res.get("error", "")},
            }
        )
        log_path.open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=True) + "\n")
        if float(args.sleep) > 0:
            time.sleep(float(args.sleep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
