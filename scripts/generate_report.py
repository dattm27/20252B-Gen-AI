"""CLI for FLAN-T5 report generation followed by factual verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report.factual_checker import check_report
from src.report.flan_t5_report import generate_factual_report
from src.report.stats_io import load_aspect_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", default="output/aspect_stats.txt")
    parser.add_argument("--table", default="predicted")
    parser.add_argument("--model", default="google/flan-t5-base")
    parser.add_argument("--max-aspects", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--report-file", help="Check an existing report instead of running FLAN-T5")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    rows = load_aspect_stats(args.stats, table=args.table)
    if args.report_file:
        report = Path(args.report_file).read_text(encoding="utf-8").strip()
        factual_check = check_report(report, rows)
        result = {"report": report, "factual_check": factual_check, "accepted": factual_check["passed"]}
    else:
        result = generate_factual_report(
            rows,
            model_name=args.model,
            max_aspects=args.max_aspects,
            max_attempts=args.max_attempts,
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
