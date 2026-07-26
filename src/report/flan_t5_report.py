"""Grounded prompt construction and inference for auditable FLAN-T5 reports."""
from __future__ import annotations

from typing import Any

REPORT_ROLES = ("most_discussed", "strongest_positive", "strongest_negative", "most_divided")


def select_report_rows(rows: list[dict[str, Any]], max_aspects: int = 4) -> list[dict[str, Any]]:
    """Select a popular aspect, a strength, a weakness, and a divided aspect.

    Remaining slots are filled by mention count. Selection is deterministic and avoids
    asking the language model to perform arithmetic or decide what the numbers mean.
    """
    if max_aspects < 1:
        raise ValueError("max_aspects must be at least 1")
    candidates = [row for row in rows if row["total"] >= 10] or rows
    by_total = sorted(candidates, key=lambda row: (-row["total"], row["aspect"]))
    has_reasons = any(
        row.get(field)
        for row in candidates
        for field in ("positive_reasons", "negative_reasons", "neutral_reasons")
    )
    if not has_reasons:
        representative_pool = by_total[:10]
        priorities = [
            by_total[0],
            max(candidates, key=lambda row: (row["positive"], row["total"])),
            max(candidates, key=lambda row: (row["negative"], row["total"])),
            min(
                representative_pool,
                key=lambda row: (abs(row["positive"] - row["negative"]), -row["total"]),
            ),
        ]
        selected = []
        for row in priorities + by_total:
            if row["aspect"] not in {item["aspect"] for item in selected}:
                selected.append(row)
            if len(selected) == max_aspects:
                break
        return selected

    representative_pool = by_total[:30]
    selected = [by_total[0]]

    def remaining() -> list[dict[str, Any]]:
        used = {item["aspect"] for item in selected}
        return [row for row in representative_pool if row["aspect"] not in used]

    if len(selected) < max_aspects and remaining():
        selected.append(
            max(remaining(), key=lambda row: (row["positive"] / row["total"], row["total"]))
        )
    if len(selected) < max_aspects and remaining():
        selected.append(
            max(remaining(), key=lambda row: (row["negative"] / row["total"], row["total"]))
        )
    if len(selected) < max_aspects and remaining():
        selected.append(
            min(
                remaining(),
                key=lambda row: (
                    abs(row["positive"] - row["negative"]) / row["total"],
                    -row["total"],
                ),
            )
        )
    for row in by_total:
        if len(selected) == max_aspects:
            break
        if row["aspect"] not in {item["aspect"] for item in selected}:
            selected.append(row)
    return selected


def _top_reasons(row: dict[str, Any], sentiment: str, limit: int = 2) -> list[tuple[str, int]]:
    return [
        (str(reason), int(count))
        for reason, count in row.get(f"{sentiment}_reasons", [])[:limit]
    ]


def _reason_text(row: dict[str, Any], limit: int = 2) -> str:
    parts = []
    for sentiment in ("positive", "negative", "neutral"):
        reasons = _top_reasons(row, sentiment, limit)
        if reasons:
            rendered = ", ".join(f"{reason} ({count})" for reason, count in reasons)
            parts.append(f"{sentiment}_reasons={rendered}")
    return " | ".join(parts) or "reasons=none"


def build_reasoned_prompt(
    rows: list[dict[str, Any]], max_aspects: int = 4, retry_number: int = 0
) -> str:
    """Build the compact prompt used by the reason-aware fine-tuned model."""
    selected = select_report_rows(rows, max_aspects=max_aspects)
    table = "\n".join(
        f"aspect={row['aspect']} | total={row['total']} | positive={row['positive']} | "
        f"negative={row['negative']} | neutral={row['neutral']} | {_reason_text(row)}"
        for row in selected
    )
    retry = (
        "Previous output was not fully grounded. Correct every aspect, count, and reason.\n"
        if retry_number
        else ""
    )
    return (
        retry
        + "Write one natural analytical restaurant-feedback paragraph from the data below. "
        "Compare the aspects, identify the clearest strength and concern, and explain why using "
        "only the supplied reasons. Mention exact counts as supporting evidence. Do not invent "
        "facts, percentages, dishes, brands, or reasons. Avoid a row-by-row list.\n"
        f"DATA:\n{table}\nREPORT:"
    )


def build_prompt(rows: list[dict[str, Any]], max_aspects: int = 4, retry_number: int = 0) -> str:
    """Build a few-shot prompt for a natural but numerically auditable paragraph."""
    selected = select_report_rows(rows, max_aspects=max_aspects)
    aspect_order = " | ".join(row["aspect"] for row in selected)
    table = "\n".join(
        f"role={role} | aspect_start {r['aspect']} aspect_end | total={r['total']} | positive={r['positive']} | "
        f"negative={r['negative']} | neutral={r['neutral']}"
        for role, r in zip(REPORT_ROLES, selected)
    )
    retry = (
        "RETRY: A previous answer was invalid. Copy the rows mechanically; do not interpret them.\n\n"
        if retry_number
        else ""
    )
    return (
        retry
        + "Task: Write one analytical customer-feedback paragraph from the role-labelled statistics.\n"
        "Definitions: total is the number of review mentions. Positive, negative, and neutral "
        "are sentiment mention counts. The numbers are not prices, hours, ratings, dimensions, "
        "resolutions, or product specifications.\n"
        "Rules:\n"
        "1. Copy every aspect and number exactly.\n"
        "2. Explain most discussed, clearest strength, main concern, and most divided in that order.\n"
        "3. Use each required aspect exactly once; never reuse an aspect for another role.\n"
        "4. Do not name a product, brand, company, or model.\n"
        "5. Do not calculate percentages or add unsupported facts.\n"
        "6. Include total, positive, negative, and neutral counts as evidence in every sentence.\n\n"
        f"Required aspect order: {aspect_order}\n\n"
        "Example input:\n"
        "role=most_discussed | aspect_start display aspect_end | total=10 | positive=5 | negative=4 | neutral=1\n"
        "role=strongest_positive | aspect_start price aspect_end | total=8 | positive=7 | negative=1 | neutral=0\n"
        "role=strongest_negative | aspect_start battery aspect_end | total=7 | positive=1 | negative=6 | neutral=0\n"
        "role=most_divided | aspect_start keyboard aspect_end | total=6 | positive=3 | negative=3 | neutral=0\n"
        "Example output:\n"
        "Display attracted the most attention, with 10 mentions: 5 positive, 4 negative, and 1 neutral. Price emerged as the clearest strength, receiving 7 positive mentions out of 8, alongside 1 negative and 0 neutral. In contrast, battery was the main concern, with 6 negative mentions out of 7, compared with 1 positive and 0 neutral. Keyboard feedback was the most divided, with 3 positive, 3 negative, and 0 neutral mentions among 6 total.\n\n"
        f"Input rows:\n{table}\n\nOutput:"
    )


def _generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=4,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def generate_report(
    rows: list[dict[str, Any]],
    model_name: str = "google/flan-t5-base",
    max_aspects: int = 4,
    max_new_tokens: int = 192,
) -> str:
    """Generate a deterministic report. Transformers is imported only when needed."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "FLAN-T5 dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    prompt = build_prompt(rows, max_aspects=max_aspects)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _generate(model, tokenizer, prompt, max_new_tokens)


def generate_factual_report(
    rows: list[dict[str, Any]],
    model_name: str = "google/flan-t5-base",
    max_aspects: int = 4,
    max_new_tokens: int = 192,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Generate, validate, and retry without reloading the model between attempts."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "FLAN-T5 dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc
    from src.report.factual_checker import check_report

    selected = select_report_rows(rows, max_aspects=max_aspects)
    required_aspects = [row["aspect"] for row in selected]
    reason_aware = any(
        row.get(field)
        for row in selected
        for field in ("positive_reasons", "negative_reasons", "neutral_reasons")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    history = []
    for attempt in range(max_attempts):
        prompt = (
            build_reasoned_prompt(rows, max_aspects=max_aspects, retry_number=attempt)
            if reason_aware
            else build_prompt(rows, max_aspects=max_aspects, retry_number=attempt)
        )
        report = _generate(model, tokenizer, prompt, max_new_tokens)
        if reason_aware:
            from src.report.factual_checker import check_reasoned_report

            factual_check = check_reasoned_report(report, selected)
        else:
            factual_check = check_report(
                report,
                rows,
                required_aspects=required_aspects,
                required_roles=list(REPORT_ROLES[: len(required_aspects)]),
            )
        history.append({"attempt": attempt + 1, "report": report, "factual_check": factual_check})
        if factual_check["passed"]:
            break

    final = history[-1]
    return {
        "report": final["report"],
        "factual_check": final["factual_check"],
        "accepted": final["factual_check"]["passed"],
        "selected_aspects": required_aspects,
        "reason_aware": reason_aware,
        "generation_attempts": len(history),
        "attempt_history": history,
    }
