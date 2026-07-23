"""Rule-based checker for numeric claims in generated aspect reports."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

STRUCTURED_CLAIM_PATTERN = re.compile(
    r"(?P<aspect>[A-Za-z][A-Za-z0-9 /_-]*?):\s*"
    r"(?P<total>\d+)\s+mentions?\s*\(\s*"
    r"(?P<positive>\d+)\s+positive\s*,\s*"
    r"(?P<negative>\d+)\s+negative\s*,\s*"
    r"(?P<neutral>\d+)\s+neutral\s*\)",
    re.IGNORECASE,
)

NATURAL_CLAIM_PATTERN = re.compile(
    r"(?:Finally,\s*)?(?P<aspect>[A-Za-z][A-Za-z0-9 /_-]*?)\s+"
    r"received\s+(?P<total>\d+)\s+mentions?\s*,\s*"
    r"(?:including|with)\s+(?P<positive>\d+)\s+positive\s*,\s*"
    r"(?P<negative>\d+)\s+negative\s*,\s*and\s*"
    r"(?P<neutral>\d+)\s+neutral",
    re.IGNORECASE,
)

ANALYTICAL_PATTERNS = {
    "most_discussed": re.compile(
        r"(?P<aspect>[A-Za-z][A-Za-z0-9 /_-]*?)\s+attracted\s+the\s+most\s+attention\s*,\s*"
        r"with\s+(?P<total>\d+)\s+mentions?\s*:\s*(?P<positive>\d+)\s+positive\s*,\s*"
        r"(?P<negative>\d+)\s+negative\s*,\s*and\s*(?P<neutral>\d+)\s+neutral",
        re.IGNORECASE,
    ),
    "strongest_positive": re.compile(
        r"(?P<aspect>[A-Za-z][A-Za-z0-9 /_-]*?)\s+emerged\s+as\s+the\s+clearest\s+strength\s*,\s*"
        r"receiving\s+(?P<positive>\d+)\s+positive\s+mentions?\s+out\s+of\s+(?P<total>\d+)\s*,\s*"
        r"alongside\s+(?P<negative>\d+)\s+negative\s+and\s+(?P<neutral>\d+)\s+neutral",
        re.IGNORECASE,
    ),
    "strongest_negative": re.compile(
        r"(?:In\s+contrast\s*,\s*)?(?P<aspect>[A-Za-z][A-Za-z0-9 /_-]*?)\s+was\s+the\s+main\s+concern\s*,\s*"
        r"with\s+(?P<negative>\d+)\s+negative\s+mentions?\s+out\s+of\s+(?P<total>\d+)\s*,\s*"
        r"compared\s+with\s+(?P<positive>\d+)\s+positive\s+and\s+(?P<neutral>\d+)\s+neutral",
        re.IGNORECASE,
    ),
    "most_divided": re.compile(
        r"(?P<aspect>[A-Za-z][A-Za-z0-9 /_-]*?)\s+feedback\s+was\s+the\s+most\s+divided\s*,\s*"
        r"with\s+(?P<positive>\d+)\s+positive\s*,\s*(?P<negative>\d+)\s+negative\s*,\s*and\s*"
        r"(?P<neutral>\d+)\s+neutral\s+mentions?\s+among\s+(?P<total>\d+)\s+total",
        re.IGNORECASE,
    ),
}


def _claim_matches(report: str) -> list[tuple[re.Match, str | None]]:
    """Return old structured and new natural claims in report order."""
    matches = [(match, None) for match in STRUCTURED_CLAIM_PATTERN.finditer(report)]
    matches.extend((match, None) for match in NATURAL_CLAIM_PATTERN.finditer(report))
    for role, pattern in ANALYTICAL_PATTERNS.items():
        matches.extend((match, role) for match in pattern.finditer(report))
    return sorted(matches, key=lambda item: item[0].start())


@dataclass(frozen=True)
class ClaimCheck:
    aspect: str
    claimed: dict[str, int]
    expected: dict[str, int] | None
    valid: bool
    errors: tuple[str, ...]
    analytical_role: str | None = None


def check_report(
    report: str,
    rows: list[dict[str, Any]],
    required_aspects: list[str] | None = None,
    required_roles: list[str] | None = None,
) -> dict[str, Any]:
    """Compare every parseable numeric claim with the matching source row."""
    source = {str(row["aspect"]).strip().lower(): row for row in rows}
    matches = _claim_matches(report)
    checks = []
    role_claims: dict[str, str] = {}
    for match, role in matches:
        aspect = match.group("aspect").strip().lower()
        claimed = {field: int(match.group(field)) for field in ("total", "positive", "negative", "neutral")}
        row = source.get(aspect)
        errors = []
        if row is None:
            errors.append("aspect not found in source statistics")
            expected = None
        else:
            expected = {field: int(row[field]) for field in claimed}
            errors.extend(
                f"{field}: claimed {claimed[field]}, expected {expected[field]}"
                for field in claimed
                if claimed[field] != expected[field]
            )
        if role:
            role_claims[role] = aspect
        checks.append(ClaimCheck(aspect, claimed, expected, not errors, tuple(errors), role))

    claim_spans = [match.span() for match, _ in matches]
    numeric_matches = re.finditer(r"\d+(?:\.\d+)?%?", report)
    unparsed_numeric_claims = sum(
        not any(start <= match.start() and match.end() <= end for start, end in claim_spans)
        for match in numeric_matches
    )
    claimed_aspects = [check.aspect for check in checks]
    duplicate_aspects = sorted({aspect for aspect in claimed_aspects if claimed_aspects.count(aspect) > 1})
    required = {aspect.strip().lower() for aspect in required_aspects or []}
    missing_aspects = sorted(required - set(claimed_aspects))
    requested_roles = list(required_roles or [])
    missing_roles = sorted(set(requested_roles) - set(role_claims))
    role_errors = []
    if requested_roles and required_aspects:
        selected = [source[aspect.strip().lower()] for aspect in required_aspects]
        expected_roles = dict(zip(requested_roles, [row["aspect"] for row in selected]))
        role_errors = [
            f"{role}: claimed {role_claims[role]}, expected {expected_roles[role]}"
            for role in requested_roles
            if role in role_claims and role_claims[role] != expected_roles[role]
        ]
    remainder = report
    for start, end in reversed(claim_spans):
        remainder = remainder[:start] + remainder[end:]
    unexpected_text = remainder.strip(" \t\r\n.;")
    all_valid = (
        bool(checks)
        and all(check.valid for check in checks)
        and unparsed_numeric_claims == 0
        and not duplicate_aspects
        and not missing_aspects
        and not missing_roles
        and not role_errors
        and not unexpected_text
    )
    return {
        "passed": all_valid,
        "claims_checked": len(checks),
        "valid_claims": sum(check.valid for check in checks),
        "unparsed_numeric_tokens": unparsed_numeric_claims,
        "missing_aspects": missing_aspects,
        "missing_roles": missing_roles,
        "role_errors": role_errors,
        "duplicate_aspects": duplicate_aspects,
        "unexpected_text": unexpected_text,
        "checks": [asdict(check) for check in checks],
    }
