"""Load and validate the aspect-statistics JSON used by report generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("aspect", "positive", "negative", "neutral", "total")


def load_aspect_stats(path: str | Path, table: str = "predicted") -> list[dict[str, Any]]:
    """Load a list of statistics rows or a named list inside a JSON object.

    ``output/aspect_stats.txt`` is JSON despite its extension and is therefore also
    accepted. Each row is checked before it reaches the prompt or factual checker.
    """
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get(table) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Statistics table '{table}' must be a non-empty JSON array")

    validated = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be a JSON object")
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Row {index} is missing fields: {', '.join(missing)}")
        clean = {"aspect": str(row["aspect"]).strip().lower()}
        if not clean["aspect"]:
            raise ValueError(f"Row {index} has an empty aspect")
        for field in REQUIRED_FIELDS[1:]:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Row {index} field '{field}' must be a non-negative integer")
            clean[field] = value
        if clean["positive"] + clean["negative"] + clean["neutral"] != clean["total"]:
            raise ValueError(f"Row {index} sentiment counts do not add up to total")
        clean["majority_sentiment"] = str(row.get("majority_sentiment", "")).lower()
        validated.append(clean)
    return validated
