"""Load, validate, and sequence the motion-analytics pattern language.

Ported from rosetta-motion/pipelines/pattern_language.py with adapted
stage vocabulary for the motion-analytics-toolkit domain.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATTERN_LANGUAGE = ROOT / "patterns" / "motion_analytics_pattern_language.v1.json"
DEFAULT_SEQUENCE_OUT = ROOT / "patterns" / "motion_analytics_pattern_sequence.v1.json"

VALID_STAGES = {
    "semantic",
    "perception",
    "analytics",
    "topology",
    "robustness",
    "governance",
    "report",
}
VALID_TIERS = {"core", "adaptable", "experimental"}


def load_pattern_language(path: Path | None = None) -> dict:
    """Load the pattern language JSON payload from disk."""
    p = path or DEFAULT_PATTERN_LANGUAGE
    return json.loads(Path(p).read_text())


def _check_required_pattern_fields(pattern: dict, idx: int) -> list[str]:
    """Validate required pattern keys."""
    req = {
        "id",
        "name",
        "stage",
        "problem",
        "solution",
        "confidence_tier",
        "larger_patterns",
        "smaller_patterns",
        "order_hint",
    }
    missing = sorted(req - set(pattern.keys()))
    return [f"patterns[{idx}] missing fields: {', '.join(missing)}"] if missing else []


def validate_pattern_language_data(data: dict) -> list[str]:
    """Return a list of validation errors for pattern language payload.

    Eight-step validation:
    1. Root structure check
    2. Patterns array check
    3. Required fields per pattern
    4. Duplicate ID detection
    5. Stage enum validation
    6. Tier enum validation
    7. Type checks (lists, ints)
    8. Link integrity (no dangling refs, no self-refs, cycle detection via Kahn's)
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["root payload must be an object"]
    if "patterns" not in data:
        return ["root payload missing 'patterns' key"]
    patterns = data.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        return ["'patterns' must be a non-empty array"]

    by_id: dict[str, dict] = {}
    for i, p in enumerate(patterns):
        if not isinstance(p, dict):
            errors.append(f"patterns[{i}] is not an object")
            continue
        errors.extend(_check_required_pattern_fields(p, i))
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            errors.append(f"patterns[{i}] has invalid id")
            continue
        if pid in by_id:
            errors.append(f"duplicate pattern id: {pid}")
        by_id[pid] = p

        if p.get("stage") not in VALID_STAGES:
            errors.append(f"pattern {pid} has invalid stage: {p.get('stage')}")
        if p.get("confidence_tier") not in VALID_TIERS:
            errors.append(f"pattern {pid} has invalid confidence_tier: {p.get('confidence_tier')}")
        if not isinstance(p.get("larger_patterns"), list):
            errors.append(f"pattern {pid} larger_patterns must be an array")
        if not isinstance(p.get("smaller_patterns"), list):
            errors.append(f"pattern {pid} smaller_patterns must be an array")
        if not isinstance(p.get("order_hint"), int):
            errors.append(f"pattern {pid} order_hint must be an integer")

    if errors:
        return errors

    # Link integrity checks.
    for pid, p in by_id.items():
        for parent in p.get("larger_patterns", []):
            if parent not in by_id:
                errors.append(f"pattern {pid} references unknown larger pattern: {parent}")
            if parent == pid:
                errors.append(f"pattern {pid} cannot reference itself as larger")
        for child in p.get("smaller_patterns", []):
            if child not in by_id:
                errors.append(f"pattern {pid} references unknown smaller pattern: {child}")
            if child == pid:
                errors.append(f"pattern {pid} cannot reference itself as smaller")

    if errors:
        return errors

    # Cycle detection via Kahn's algorithm on larger -> smaller edges.
    in_deg: dict[str, int] = defaultdict(int)
    out: dict[str, list[str]] = defaultdict(list)
    for pid in by_id:
        in_deg[pid] = 0
    for pid, p in by_id.items():
        for child in p.get("smaller_patterns", []):
            out[pid].append(child)
            in_deg[child] += 1
    q: deque[str] = deque(pid for pid, deg in in_deg.items() if deg == 0)
    seen = 0
    while q:
        cur = q.popleft()
        seen += 1
        for nxt in out[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                q.append(nxt)
    if seen != len(by_id):
        errors.append("pattern graph contains at least one cycle")

    return errors


def build_pattern_sequence(data: dict) -> list[dict]:
    """Build a deterministic larger-to-smaller sequence of pattern rows.

    Uses topological sort with ``(order_hint, id)`` as tiebreaker for
    deterministic ordering across runs.
    """
    errs = validate_pattern_language_data(data)
    if errs:
        raise ValueError("invalid pattern language: " + "; ".join(errs))

    patterns = data["patterns"]
    by_id = {p["id"]: p for p in patterns}

    out: dict[str, list[str]] = defaultdict(list)
    in_deg: dict[str, int] = defaultdict(int)
    for pid in by_id:
        in_deg[pid] = 0
    for p in patterns:
        src = p["id"]
        for dst in p["smaller_patterns"]:
            out[src].append(dst)
            in_deg[dst] += 1

    ready = [pid for pid, deg in in_deg.items() if deg == 0]

    def _sort_key(pid: str) -> tuple[int, str]:
        row = by_id[pid]
        return (int(row.get("order_hint", 0)), pid)

    ready.sort(key=_sort_key)
    ordered: list[str] = []
    while ready:
        cur = ready.pop(0)
        ordered.append(cur)
        for nxt in out[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=_sort_key)

    return [
        {
            "index": i,
            "id": pid,
            "name": by_id[pid]["name"],
            "stage": by_id[pid]["stage"],
            "confidence_tier": by_id[pid]["confidence_tier"],
            "order_hint": by_id[pid]["order_hint"],
            "larger_patterns": by_id[pid]["larger_patterns"],
            "smaller_patterns": by_id[pid]["smaller_patterns"],
        }
        for i, pid in enumerate(ordered)
    ]


def write_pattern_sequence(data: dict, out_path: Path | None = None) -> Path:
    """Build and write deterministic sequence artifact."""
    seq = build_pattern_sequence(data)
    out = out_path or DEFAULT_SEQUENCE_OUT
    payload = {
        "version": data.get("version"),
        "source": str(DEFAULT_PATTERN_LANGUAGE),
        "sequence": seq,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    return Path(out)
