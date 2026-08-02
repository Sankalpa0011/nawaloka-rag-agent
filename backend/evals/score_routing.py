"""Score the router against evals/routing_set.jsonl.

    python -m evals.score_routing                 # full run
    python -m evals.score_routing --no-llm        # deterministic signals only
    python -m evals.score_routing --limit 10

Responses are cached to evals/.llm_cache so repeated runs do not burn free-tier
quota, and so the reported number is reproducible. Use --fresh to bypass it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from app.agent.router import route_query
from app.logging_conf import configure_logging

EVAL_PATH = Path(__file__).parent / "routing_set.jsonl"
CACHE_DIR = Path(__file__).parent / ".llm_cache"

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cache_key(case: dict, use_llm: bool) -> Path:
    raw = json.dumps(
        {"q": case["query"], "h": case.get("history"), "llm": use_llm}, sort_keys=True
    )
    return CACHE_DIR / f"{hashlib.sha1(raw.encode()).hexdigest()[:16]}.json"


async def run_case(case: dict, use_llm: bool, use_cache: bool) -> dict:
    cache_file = _cache_key(case, use_llm)
    if use_cache and cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        payload["cached"] = True
        return payload

    decision = await route_query(case["query"], history=case.get("history"), use_llm=use_llm)
    payload = decision.to_dict()
    payload["cached"] = False

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true",
                        help="route using deterministic signals only (baseline)")
    parser.add_argument("--fresh", action="store_true", help="ignore the response cache")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="show every case")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    configure_logging("WARNING")
    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    use_llm = not args.no_llm
    print(f"{BOLD}Routing evaluation{RESET}  "
          f"{len(cases)} cases  mode={'llm+signals' if use_llm else 'signals only'}")
    print("=" * 78)

    results = []
    started = time.perf_counter()
    for i, case in enumerate(cases, 1):
        outcome = await run_case(case, use_llm, use_cache=not args.fresh)
        expected, got = case["expected_route"], outcome["route"]
        ok = expected == got
        results.append({"case": case, "outcome": outcome, "ok": ok})

        if args.verbose or not ok:
            mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
            print(f"{mark} [{case['id']}] {case['query'][:56]}")
            if not ok:
                print(f"       expected {expected}, got {got}"
                      f"  (llm said {outcome.get('llm_route') or '-'},"
                      f" conf {outcome.get('confidence')})")
                if outcome.get("override_reason"):
                    print(f"       override: {outcome['override_reason']}")
                if outcome.get("error"):
                    print(f"       {RED}error: {outcome['error'][:120]}{RESET}")
        elif i % 10 == 0:
            print(f"{DIM}  ...{i}/{len(cases)}{RESET}")

    elapsed = time.perf_counter() - started
    correct = sum(1 for r in results if r["ok"])
    accuracy = correct / len(results) if results else 0.0

    # ---- per-class ------------------------------------------------------
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in results:
        route = r["case"]["expected_route"]
        per_class[route]["n"] += 1
        per_class[route]["ok"] += int(r["ok"])

    print("\n" + "=" * 78)
    print(f"{BOLD}ROUTING ACCURACY: {accuracy:.1%}  ({correct}/{len(results)}){RESET}")
    print("=" * 78)

    print(f"\n{'route':<16}{'n':>4}{'correct':>9}{'accuracy':>10}")
    print("-" * 42)
    for route in sorted(per_class, key=lambda r: -per_class[r]["n"]):
        st = per_class[route]
        acc = st["ok"] / st["n"]
        colour = GREEN if acc >= 0.9 else (YELLOW if acc >= 0.7 else RED)
        print(f"{route:<16}{st['n']:>4}{st['ok']:>9}{colour}{acc:>9.0%}{RESET}")

    # ---- confusion ------------------------------------------------------
    confusion = Counter(
        (r["case"]["expected_route"], r["outcome"]["route"])
        for r in results if not r["ok"]
    )
    if confusion:
        print(f"\n{BOLD}Confusions{RESET} (expected -> predicted)")
        print("-" * 42)
        for (exp, got), n in confusion.most_common():
            print(f"  {exp:<14} -> {got:<14} {n}")

    # ---- trap coverage --------------------------------------------------
    trap_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in results:
        for tag in r["case"].get("tags", []):
            if tag.startswith("trap:") or tag in ("multiturn", "injection"):
                trap_stats[tag]["n"] += 1
                trap_stats[tag]["ok"] += int(r["ok"])
    if trap_stats:
        print(f"\n{BOLD}Trap / edge-case routing{RESET}")
        print("-" * 42)
        for tag in sorted(trap_stats):
            st = trap_stats[tag]
            colour = GREEN if st["ok"] == st["n"] else RED
            print(f"  {tag:<24}{colour}{st['ok']}/{st['n']}{RESET}")

    # ---- overrides ------------------------------------------------------
    overridden = [r for r in results if r["outcome"].get("overridden")]
    if overridden:
        helped = sum(1 for r in overridden if r["ok"])
        print(f"\n{BOLD}Reconciliation overrides{RESET}: {len(overridden)} "
              f"({helped} correct after override)")

    cached = sum(1 for r in results if r["outcome"].get("cached"))
    print(f"\n{DIM}{elapsed:.1f}s total, {cached}/{len(results)} served from cache{RESET}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": len(results),
                    "per_class": {k: dict(v) for k, v in per_class.items()},
                    "confusions": {f"{e}->{g}": n for (e, g), n in confusion.items()},
                    "results": [
                        {"id": r["case"]["id"], "expected": r["case"]["expected_route"],
                         "got": r["outcome"]["route"], "ok": r["ok"]}
                        for r in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{DIM}wrote {args.json_out}{RESET}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
