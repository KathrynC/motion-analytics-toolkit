"""Coverage analysis for the motion-analytics pattern language.

Provides functions to measure which patterns were exercised in a run,
tier breakdowns, dependency completeness checks, and reporting.
"""
from __future__ import annotations

import json
from pathlib import Path

# Maps pattern IDs to their source module paths in the toolkit.
PATTERN_MODULE_MAP = {
    "grounding_foundation": "motion_analytics.archetypes.base",
    "image_schema_detection": "motion_analytics.core.image_schemas",
    "behavioral_feature_extraction": "motion_analytics.biomechanics.gait",
    "archetype_matching": "motion_analytics.archetypes.persona",
    "lattice_construction": "motion_analytics.semantic_ca.lattice",
    "cluster_identification": "motion_analytics.semantic_ca.emergence",
    "stress_testing": "motion_analytics.scenarios.base",
    "violation_auditing": "motion_analytics.archetypes.violations",
    "coverage_reporting": "motion_analytics.patterns.coverage",
}


def check_dependency_completeness(
    touched: set[str], pattern_data: dict
) -> list[str]:
    """Warn when a touched pattern's prerequisites were not touched.

    Returns a list of human-readable warning strings.
    """
    by_id = {p["id"]: p for p in pattern_data["patterns"]}
    warnings: list[str] = []
    for pid in sorted(touched):
        p = by_id.get(pid)
        if p is None:
            continue
        for parent in p.get("larger_patterns", []):
            if parent not in touched:
                warnings.append(
                    f"pattern '{pid}' was touched but prerequisite '{parent}' was not"
                )
    return warnings


def analyze_coverage(
    run_log: dict[str, dict | None], pattern_data: dict
) -> dict:
    """Analyze pattern coverage from a run log.

    Parameters
    ----------
    run_log : dict
        Mapping of ``{pattern_id: result_dict_or_None}``.  A pattern is
        considered *touched* if its key is present in the log and the value
        is not ``None``.
    pattern_data : dict
        The full pattern language payload (with ``"patterns"`` key).

    Returns
    -------
    dict
        Coverage report with keys: ``total_patterns``, ``touched_patterns``,
        ``missed_patterns``, ``coverage_fraction``, ``core_coverage``,
        ``adaptable_coverage``, ``tier_breakdown``, ``dependency_completeness``,
        ``stage_coverage``.
    """
    patterns = pattern_data["patterns"]
    all_ids = {p["id"] for p in patterns}
    touched = {pid for pid, result in run_log.items() if pid in all_ids and result is not None}
    missed = all_ids - touched

    total = len(all_ids)
    frac = len(touched) / total if total > 0 else 0.0

    # Tier breakdown.
    tier_breakdown: dict[str, dict] = {}
    for tier in ("core", "adaptable", "experimental"):
        tier_ids = {p["id"] for p in patterns if p["confidence_tier"] == tier}
        tier_touched = tier_ids & touched
        tier_missed = tier_ids - touched
        tier_breakdown[tier] = {
            "total": len(tier_ids),
            "touched": len(tier_touched),
            "missed": len(tier_missed),
        }

    core_total = tier_breakdown["core"]["total"]
    core_coverage = tier_breakdown["core"]["touched"] / core_total if core_total > 0 else 0.0
    adapt_total = tier_breakdown["adaptable"]["total"]
    adaptable_coverage = tier_breakdown["adaptable"]["touched"] / adapt_total if adapt_total > 0 else 0.0

    # Stage coverage.
    stage_coverage: dict[str, dict] = {}
    stages = {p["stage"] for p in patterns}
    for stage in sorted(stages):
        stage_ids = {p["id"] for p in patterns if p["stage"] == stage}
        stage_touched = stage_ids & touched
        stage_coverage[stage] = {
            "total": len(stage_ids),
            "touched": len(stage_touched),
        }

    dep_warnings = check_dependency_completeness(touched, pattern_data)

    return {
        "total_patterns": total,
        "touched_patterns": sorted(touched),
        "missed_patterns": sorted(missed),
        "coverage_fraction": round(frac, 4),
        "core_coverage": round(core_coverage, 4),
        "adaptable_coverage": round(adaptable_coverage, 4),
        "tier_breakdown": tier_breakdown,
        "dependency_completeness": dep_warnings,
        "stage_coverage": stage_coverage,
    }


def coverage_report_to_json(report: dict, path: str | Path) -> Path:
    """Write a coverage report dict to disk as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    return p


def summarize_coverage(report: dict) -> str:
    """Return a human-readable summary string from a coverage report."""
    lines = [
        f"Pattern coverage: {report['coverage_fraction']:.0%} "
        f"({len(report['touched_patterns'])}/{report['total_patterns']})",
        f"  Core:      {report['core_coverage']:.0%} "
        f"({report['tier_breakdown']['core']['touched']}/{report['tier_breakdown']['core']['total']})",
        f"  Adaptable: {report['adaptable_coverage']:.0%} "
        f"({report['tier_breakdown']['adaptable']['touched']}/{report['tier_breakdown']['adaptable']['total']})",
    ]
    exp = report["tier_breakdown"].get("experimental", {})
    if exp.get("total", 0) > 0:
        exp_frac = exp["touched"] / exp["total"]
        lines.append(
            f"  Experimental: {exp_frac:.0%} ({exp['touched']}/{exp['total']})"
        )
    if report["dependency_completeness"]:
        lines.append("  Dependency warnings:")
        for w in report["dependency_completeness"]:
            lines.append(f"    - {w}")
    return "\n".join(lines)
