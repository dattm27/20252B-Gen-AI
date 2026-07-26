"""CLI for FLAN-T5 report generation followed by factual verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report.factual_checker import check_reasoned_report, check_report
from src.report.flan_t5_report import generate_factual_report, select_report_rows
from src.report.stats_io import load_aspect_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", default="output/aspect_reasons_restaurant.json")
    parser.add_argument("--table", default="predicted")
    parser.add_argument("--model", default="google/flan-t5-base")
    parser.add_argument("--max-aspects", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--checker-mode",
        choices=("factual-only", "strict"),
        default="factual-only",
        help="Reason-aware reports: accept grounded claims only, or also require full aspect coverage",
    )
    parser.add_argument("--report-file", help="Check an existing report instead of running FLAN-T5")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    rows = load_aspect_stats(args.stats, table=args.table)
    if args.report_file:
        report = Path(args.report_file).read_text(encoding="utf-8").strip()
        selected = select_report_rows(rows, max_aspects=args.max_aspects)
        reason_aware = any(
            row.get(field)
            for row in selected
            for field in ("positive_reasons", "negative_reasons", "neutral_reasons")
        )
        factual_check = (
            check_reasoned_report(report, selected, mode=args.checker_mode)
            if reason_aware
            else check_report(report, rows)
        )
        warnings = factual_check.get("coverage_warnings", [])
        result = {
            "report": report,
            "factual_check": factual_check,
            "accepted": factual_check["passed"],
            "status": (
                "accepted_with_warnings"
                if factual_check["passed"] and warnings
                else "accepted"
                if factual_check["passed"]
                else "rejected"
            ),
            "checker_mode": args.checker_mode if reason_aware else "strict",
        }
    else:
        result = generate_factual_report(
            rows,
            model_name=args.model,
            max_aspects=args.max_aspects,
            max_attempts=args.max_attempts,
            checker_mode=args.checker_mode,
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
