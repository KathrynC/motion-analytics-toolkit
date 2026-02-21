"""Tests for the motion-analytics pattern coverage analysis."""

import json
import pytest
from pathlib import Path

from motion_analytics.patterns.pattern_language import load_pattern_language
from motion_analytics.patterns.coverage import (
    PATTERN_MODULE_MAP,
    analyze_coverage,
    check_dependency_completeness,
    coverage_report_to_json,
    summarize_coverage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def pattern_data():
    return load_pattern_language()


def _all_pattern_ids(data):
    return {p["id"] for p in data["patterns"]}


# ---------------------------------------------------------------------------
# PATTERN_MODULE_MAP
# ---------------------------------------------------------------------------

class TestPatternModuleMap:
    def test_all_patterns_mapped(self, pattern_data):
        ids = _all_pattern_ids(pattern_data)
        assert ids == set(PATTERN_MODULE_MAP.keys())

    def test_map_values_are_strings(self):
        for pid, mod in PATTERN_MODULE_MAP.items():
            assert isinstance(mod, str), f"{pid} mapped to non-string"
            assert "." in mod, f"{pid} module path lacks dots: {mod}"


# ---------------------------------------------------------------------------
# check_dependency_completeness
# ---------------------------------------------------------------------------

class TestDependencyCompleteness:
    def test_no_warnings_when_all_touched(self, pattern_data):
        all_ids = _all_pattern_ids(pattern_data)
        warnings = check_dependency_completeness(all_ids, pattern_data)
        assert warnings == []

    def test_warns_missing_prerequisite(self, pattern_data):
        # Touch a child without its parent.
        warnings = check_dependency_completeness(
            {"image_schema_detection"}, pattern_data
        )
        assert any("grounding_foundation" in w for w in warnings)

    def test_no_warnings_for_root_only(self, pattern_data):
        warnings = check_dependency_completeness(
            {"grounding_foundation"}, pattern_data
        )
        assert warnings == []

    def test_empty_touched(self, pattern_data):
        warnings = check_dependency_completeness(set(), pattern_data)
        assert warnings == []

    def test_unknown_id_ignored(self, pattern_data):
        warnings = check_dependency_completeness(
            {"nonexistent_pattern"}, pattern_data
        )
        assert warnings == []


# ---------------------------------------------------------------------------
# analyze_coverage
# ---------------------------------------------------------------------------

class TestAnalyzeCoverage:
    def test_full_coverage(self, pattern_data):
        all_ids = _all_pattern_ids(pattern_data)
        run_log = {pid: {"status": "ok"} for pid in all_ids}
        report = analyze_coverage(run_log, pattern_data)
        assert report["coverage_fraction"] == 1.0
        assert report["missed_patterns"] == []
        assert report["core_coverage"] == 1.0

    def test_zero_coverage(self, pattern_data):
        report = analyze_coverage({}, pattern_data)
        assert report["coverage_fraction"] == 0.0
        assert len(report["missed_patterns"]) == 9

    def test_none_result_not_touched(self, pattern_data):
        run_log = {"grounding_foundation": None}
        report = analyze_coverage(run_log, pattern_data)
        assert "grounding_foundation" in report["missed_patterns"]

    def test_partial_coverage(self, pattern_data):
        run_log = {
            "grounding_foundation": {"ok": True},
            "image_schema_detection": {"ok": True},
            "behavioral_feature_extraction": {"ok": True},
        }
        report = analyze_coverage(run_log, pattern_data)
        assert report["total_patterns"] == 9
        assert len(report["touched_patterns"]) == 3
        assert len(report["missed_patterns"]) == 6
        assert 0.3 < report["coverage_fraction"] < 0.4

    def test_tier_breakdown_present(self, pattern_data):
        run_log = {"grounding_foundation": {"ok": True}}
        report = analyze_coverage(run_log, pattern_data)
        assert "core" in report["tier_breakdown"]
        assert "adaptable" in report["tier_breakdown"]
        assert "experimental" in report["tier_breakdown"]
        assert report["tier_breakdown"]["core"]["total"] > 0

    def test_stage_coverage_present(self, pattern_data):
        run_log = {"grounding_foundation": {"ok": True}}
        report = analyze_coverage(run_log, pattern_data)
        assert "semantic" in report["stage_coverage"]
        assert report["stage_coverage"]["semantic"]["total"] >= 1

    def test_dependency_warnings_in_report(self, pattern_data):
        run_log = {"image_schema_detection": {"ok": True}}
        report = analyze_coverage(run_log, pattern_data)
        assert len(report["dependency_completeness"]) > 0

    def test_unknown_ids_in_run_log_ignored(self, pattern_data):
        run_log = {"bogus_id": {"ok": True}}
        report = analyze_coverage(run_log, pattern_data)
        assert report["coverage_fraction"] == 0.0


# ---------------------------------------------------------------------------
# coverage_report_to_json
# ---------------------------------------------------------------------------

class TestCoverageReportToJson:
    def test_writes_file(self, tmp_path, pattern_data):
        run_log = {"grounding_foundation": {"ok": True}}
        report = analyze_coverage(run_log, pattern_data)
        out = coverage_report_to_json(report, tmp_path / "report.json")
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["total_patterns"] == 9

    def test_creates_parent_dirs(self, tmp_path, pattern_data):
        report = analyze_coverage({}, pattern_data)
        out = coverage_report_to_json(
            report, tmp_path / "nested" / "dir" / "report.json"
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# summarize_coverage
# ---------------------------------------------------------------------------

class TestSummarizeCoverage:
    def test_summary_is_string(self, pattern_data):
        report = analyze_coverage({}, pattern_data)
        summary = summarize_coverage(report)
        assert isinstance(summary, str)
        assert "0%" in summary

    def test_full_coverage_summary(self, pattern_data):
        all_ids = _all_pattern_ids(pattern_data)
        run_log = {pid: {"ok": True} for pid in all_ids}
        report = analyze_coverage(run_log, pattern_data)
        summary = summarize_coverage(report)
        assert "100%" in summary
        assert "Core" in summary

    def test_dependency_warnings_in_summary(self, pattern_data):
        run_log = {"image_schema_detection": {"ok": True}}
        report = analyze_coverage(run_log, pattern_data)
        summary = summarize_coverage(report)
        assert "Dependency warnings" in summary
