"""Load and validate the aspect-statistics JSON used by report generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("aspect", "positive", "negative", "neutral", "total")
REASON_FIELDS = ("positive_reasons", "negative_reasons", "neutral_reasons")


def _validate_reasons(value: Any, row_index: int, field: str) -> list[list[Any]]:
    """Validate and normalize ``[[reason, count], ...]`` entries."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Row {row_index} field '{field}' must be a list")
    clean = []
    for reason_index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"Row {row_index} field '{field}' item {reason_index} must be [reason, count]"
            )
        reason = str(item[0]).strip().lower()
        count = item[1]
        if not reason:
            raise ValueError(f"Row {row_index} field '{field}' contains an empty reason")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f"Row {row_index} field '{field}' reason counts must be positive integers"
            )
        clean.append([reason, count])
    return clean


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
        for field in REASON_FIELDS:
            clean[field] = _validate_reasons(row.get(field), index, field)
        validated.append(clean)
    return validated
