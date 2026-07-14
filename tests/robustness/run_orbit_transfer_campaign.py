"""Run the deterministic orbit-transfer campaign and write a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from orbit_transfers import (
    DEFAULT_CAMPAIGN_SEED,
    build_transfer_mission,
    generate_transfer_scenarios,
    solution_checks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=100, help="Number of generated transfers.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CAMPAIGN_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("robustness-orbit-transfers.json"),
        help="JSON report path.",
    )
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for scenario in generate_transfer_scenarios(args.cases, seed=args.seed):
        case_started = time.perf_counter()
        record: dict[str, object] = {"scenario": scenario.to_dict()}
        try:
            solution = build_transfer_mission(scenario).solve()
            record.update(
                {
                    "status": "passed",
                    "metrics": solution_checks(scenario, solution),
                }
            )
        except Exception as exc:  # noqa: BLE001 - campaign must preserve every failure
            record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        record["runtime_s"] = time.perf_counter() - case_started
        records.append(record)
        print(f"[{len(records):03d}/{args.cases:03d}] {scenario.name}: {record['status']}")

    passed = sum(record["status"] == "passed" for record in records)
    report = {
        "campaign": "orbit_transfers",
        "seed": args.seed,
        "case_count": args.cases,
        "passed": passed,
        "failed": args.cases - passed,
        "success_rate": passed / args.cases,
        "runtime_s": time.perf_counter() - started,
        "cases": records,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({passed}/{args.cases} passed)")
    return 0 if passed == args.cases else 1


if __name__ == "__main__":
    sys.exit(main())
