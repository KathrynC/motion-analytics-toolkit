"""Tests for the motion-analytics pattern language validation and sequencing."""

import copy
import json
import pytest
from pathlib import Path

from motion_analytics.patterns.pattern_language import (
    VALID_STAGES,
    VALID_TIERS,
    DEFAULT_PATTERN_LANGUAGE,
    build_pattern_sequence,
    load_pattern_language,
    validate_pattern_language_data,
    write_pattern_sequence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_graph():
    """Two-node valid pattern graph."""
    return {
        "version": "test",
        "patterns": [
            {
                "id": "a",
                "name": "A",
                "stage": "semantic",
                "problem": "p",
                "solution": "s",
                "confidence_tier": "core",
                "larger_patterns": [],
                "smaller_patterns": ["b"],
                "order_hint": 0,
            },
            {
                "id": "b",
                "name": "B",
                "stage": "analytics",
                "problem": "p",
                "solution": "s",
                "confidence_tier": "adaptable",
                "larger_patterns": ["a"],
                "smaller_patterns": [],
                "order_hint": 10,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_load_default(self):
        data = load_pattern_language()
        assert "patterns" in data
        assert len(data["patterns"]) == 9

    def test_load_explicit_path(self):
        data = load_pattern_language(DEFAULT_PATTERN_LANGUAGE)
        assert data["version"] == "v1"

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pattern_language(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Validation — happy path
# ---------------------------------------------------------------------------

class TestValidationHappyPath:
    def test_shipped_graph_validates(self):
        data = load_pattern_language()
        errors = validate_pattern_language_data(data)
        assert errors == []

    def test_minimal_graph_validates(self):
        errors = validate_pattern_language_data(_make_minimal_graph())
        assert errors == []


# ---------------------------------------------------------------------------
# Validation — structural errors
# ---------------------------------------------------------------------------

class TestValidationStructuralErrors:
    def test_non_dict_root(self):
        errors = validate_pattern_language_data([1, 2])
        assert errors == ["root payload must be an object"]

    def test_missing_patterns_key(self):
        errors = validate_pattern_language_data({"version": "v1"})
        assert errors == ["root payload missing 'patterns' key"]

    def test_empty_patterns_array(self):
        errors = validate_pattern_language_data({"patterns": []})
        assert errors == ["'patterns' must be a non-empty array"]

    def test_patterns_not_list(self):
        errors = validate_pattern_language_data({"patterns": "bad"})
        assert errors == ["'patterns' must be a non-empty array"]

    def test_pattern_not_dict(self):
        errors = validate_pattern_language_data({"patterns": ["not_a_dict"]})
        assert any("is not an object" in e for e in errors)


# ---------------------------------------------------------------------------
# Validation — field-level errors
# ---------------------------------------------------------------------------

class TestValidationFieldErrors:
    def test_missing_required_field(self):
        data = _make_minimal_graph()
        del data["patterns"][0]["problem"]
        errors = validate_pattern_language_data(data)
        assert any("missing fields" in e for e in errors)

    def test_invalid_stage(self):
        data = _make_minimal_graph()
        data["patterns"][0]["stage"] = "bogus"
        errors = validate_pattern_language_data(data)
        assert any("invalid stage" in e for e in errors)

    def test_invalid_tier(self):
        data = _make_minimal_graph()
        data["patterns"][0]["confidence_tier"] = "bogus"
        errors = validate_pattern_language_data(data)
        assert any("invalid confidence_tier" in e for e in errors)

    def test_order_hint_not_int(self):
        data = _make_minimal_graph()
        data["patterns"][0]["order_hint"] = "zero"
        errors = validate_pattern_language_data(data)
        assert any("order_hint must be an integer" in e for e in errors)

    def test_larger_patterns_not_list(self):
        data = _make_minimal_graph()
        data["patterns"][0]["larger_patterns"] = "bad"
        errors = validate_pattern_language_data(data)
        assert any("larger_patterns must be an array" in e for e in errors)

    def test_smaller_patterns_not_list(self):
        data = _make_minimal_graph()
        data["patterns"][0]["smaller_patterns"] = "bad"
        errors = validate_pattern_language_data(data)
        assert any("smaller_patterns must be an array" in e for e in errors)

    def test_duplicate_id(self):
        data = _make_minimal_graph()
        data["patterns"][1]["id"] = "a"
        data["patterns"][1]["larger_patterns"] = []
        errors = validate_pattern_language_data(data)
        assert any("duplicate pattern id" in e for e in errors)

    def test_invalid_id_type(self):
        data = _make_minimal_graph()
        data["patterns"][0]["id"] = 123
        errors = validate_pattern_language_data(data)
        assert any("invalid id" in e for e in errors)

    def test_empty_id(self):
        data = _make_minimal_graph()
        data["patterns"][0]["id"] = ""
        errors = validate_pattern_language_data(data)
        assert any("invalid id" in e for e in errors)


# ---------------------------------------------------------------------------
# Validation — link integrity
# ---------------------------------------------------------------------------

class TestValidationLinkIntegrity:
    def test_dangling_larger_ref(self):
        data = _make_minimal_graph()
        data["patterns"][1]["larger_patterns"] = ["nonexistent"]
        errors = validate_pattern_language_data(data)
        assert any("unknown larger pattern" in e for e in errors)

    def test_dangling_smaller_ref(self):
        data = _make_minimal_graph()
        data["patterns"][0]["smaller_patterns"] = ["nonexistent"]
        errors = validate_pattern_language_data(data)
        assert any("unknown smaller pattern" in e for e in errors)

    def test_self_ref_larger(self):
        data = _make_minimal_graph()
        data["patterns"][0]["larger_patterns"] = ["a"]
        errors = validate_pattern_language_data(data)
        assert any("cannot reference itself as larger" in e for e in errors)

    def test_self_ref_smaller(self):
        data = _make_minimal_graph()
        data["patterns"][0]["smaller_patterns"] = ["a"]
        errors = validate_pattern_language_data(data)
        assert any("cannot reference itself as smaller" in e for e in errors)


# ---------------------------------------------------------------------------
# Validation — cycle detection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    def test_cycle_detected(self):
        data = {
            "version": "test",
            "patterns": [
                {
                    "id": "x",
                    "name": "X",
                    "stage": "semantic",
                    "problem": "p",
                    "solution": "s",
                    "confidence_tier": "core",
                    "larger_patterns": [],
                    "smaller_patterns": ["y"],
                    "order_hint": 0,
                },
                {
                    "id": "y",
                    "name": "Y",
                    "stage": "analytics",
                    "problem": "p",
                    "solution": "s",
                    "confidence_tier": "core",
                    "larger_patterns": ["x"],
                    "smaller_patterns": ["x"],
                    "order_hint": 10,
                },
            ],
        }
        errors = validate_pattern_language_data(data)
        assert any("cycle" in e for e in errors)

    def test_no_false_cycle_on_dag(self):
        data = load_pattern_language()
        errors = validate_pattern_language_data(data)
        assert not any("cycle" in e for e in errors)


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------

class TestSequencing:
    def test_sequence_length(self):
        seq = build_pattern_sequence(load_pattern_language())
        assert len(seq) == 9

    def test_sequence_indices_contiguous(self):
        seq = build_pattern_sequence(load_pattern_language())
        assert [s["index"] for s in seq] == list(range(len(seq)))

    def test_sequence_respects_dependencies(self):
        seq = build_pattern_sequence(load_pattern_language())
        order = {s["id"]: s["index"] for s in seq}
        data = load_pattern_language()
        for p in data["patterns"]:
            for child in p["smaller_patterns"]:
                assert order[p["id"]] < order[child], (
                    f"{p['id']} should come before {child}"
                )

    def test_grounding_first(self):
        seq = build_pattern_sequence(load_pattern_language())
        assert seq[0]["id"] == "grounding_foundation"

    def test_coverage_reporting_last(self):
        seq = build_pattern_sequence(load_pattern_language())
        assert seq[-1]["id"] == "coverage_reporting"

    def test_minimal_sequence(self):
        seq = build_pattern_sequence(_make_minimal_graph())
        assert [s["id"] for s in seq] == ["a", "b"]

    def test_invalid_data_raises(self):
        with pytest.raises(ValueError, match="invalid pattern language"):
            build_pattern_sequence({"patterns": []})

    def test_sequence_row_has_expected_keys(self):
        seq = build_pattern_sequence(_make_minimal_graph())
        expected_keys = {
            "index", "id", "name", "stage", "confidence_tier",
            "order_hint", "larger_patterns", "smaller_patterns",
        }
        assert set(seq[0].keys()) == expected_keys


# ---------------------------------------------------------------------------
# Write sequence
# ---------------------------------------------------------------------------

class TestWriteSequence:
    def test_write_creates_file(self, tmp_path):
        out = tmp_path / "seq.json"
        result = write_pattern_sequence(_make_minimal_graph(), out)
        assert result.exists()
        payload = json.loads(result.read_text())
        assert "sequence" in payload
        assert payload["version"] == "test"

    def test_write_default_path(self):
        """Write to default path and verify; clean up after."""
        from motion_analytics.patterns.pattern_language import DEFAULT_SEQUENCE_OUT
        data = load_pattern_language()
        result = write_pattern_sequence(data)
        assert result.exists()
        payload = json.loads(result.read_text())
        assert len(payload["sequence"]) == 9


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_valid_stages_set(self):
        assert "semantic" in VALID_STAGES
        assert "perception" in VALID_STAGES
        assert "report" in VALID_STAGES
        assert len(VALID_STAGES) == 7

    def test_valid_tiers_set(self):
        assert VALID_TIERS == {"core", "adaptable", "experimental"}
