"""Tests for metaphor violation detection (Lakoff ICM framework)."""

import numpy as np
import pytest

from motion_analytics.archetypes.base import GroundingCriterion, ICM
from motion_analytics.archetypes.violations import MetaphorAuditor
from motion_analytics.archetypes.templates import BUILTIN_ARCHETYPES
from motion_analytics.archetypes.persona import PersonaArchetype
from motion_analytics.archetypes.base import ArchetypeLibrary
from motion_analytics.core.schemas import (
    Telemetry, TimeStep, LinkState, JointState, ContactState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(t, x=0.0):
    return TimeStep(
        timestamp=t,
        links={
            'torso': LinkState([x, 0, 0.1], [0, 0, 0, 1], [0.01, 0, 0], [0, 0, 0]),
            'back_leg': LinkState([x - 0.5, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]),
            'front_leg': LinkState([x + 0.5, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]),
        },
        joints={
            'back_leg_joint': JointState(angle=0.3 * np.sin(2 * np.pi * 2 * t), velocity=0.5, torque=0.1),
            'front_leg_joint': JointState(angle=-0.3 * np.sin(2 * np.pi * 2 * t), velocity=0.5, torque=0.1),
        },
        contacts=[
            ContactState('back_leg', 1.0, [0, 0], [0, 0, 0], np.sin(2 * np.pi * 2 * t) > 0),
            ContactState('front_leg', 1.0, [0, 0], [0, 0, 0], np.sin(2 * np.pi * 2 * t) <= 0),
        ],
        com_position=[x, 0.0, 0.1],
        com_velocity=[0.01, 0.0, 0.0],
    )


def _make_telemetry(n=200, sr=240.0, weights=None):
    steps = [_make_ts(i / sr, x=i * 0.001) for i in range(n)]
    meta = {'source': 'test'}
    if weights is not None:
        meta['synapse_weights'] = weights
    return Telemetry(metadata=meta, timesteps=steps, sampling_rate=sr)


# ---------------------------------------------------------------------------
# GroundingCriterion tests
# ---------------------------------------------------------------------------

class TestGroundingCriterion:
    def test_gt_pass(self):
        gc = GroundingCriterion('x', 'gt', 0.5)
        assert gc.check({'x': 0.8}) is True

    def test_gt_fail(self):
        gc = GroundingCriterion('x', 'gt', 0.5)
        assert gc.check({'x': 0.3}) is False

    def test_lt_pass(self):
        gc = GroundingCriterion('x', 'lt', 0.5)
        assert gc.check({'x': 0.2}) is True

    def test_near_pass(self):
        gc = GroundingCriterion('x', 'near', 1.0, tolerance=0.1)
        assert gc.check({'x': 1.05}) is True

    def test_near_fail(self):
        gc = GroundingCriterion('x', 'near', 1.0, tolerance=0.1)
        assert gc.check({'x': 1.5}) is False

    def test_between_pass(self):
        gc = GroundingCriterion('x', 'between', 0.2, tolerance=0.8)
        assert gc.check({'x': 0.5}) is True

    def test_between_fail(self):
        gc = GroundingCriterion('x', 'between', 0.2, tolerance=0.8)
        assert gc.check({'x': 0.1}) is False

    def test_missing_feature(self):
        gc = GroundingCriterion('missing', 'gt', 0.5)
        assert gc.check({'x': 1.0}) is False

    def test_serialization_roundtrip(self):
        gc = GroundingCriterion('x', 'gt', 0.5, rationale='test reason')
        d = gc.to_dict()
        gc2 = GroundingCriterion.from_dict(d)
        assert gc2.feature == 'x'
        assert gc2.predicate == 'gt'
        assert gc2.value == 0.5
        assert gc2.rationale == 'test reason'


# ---------------------------------------------------------------------------
# ICM tests
# ---------------------------------------------------------------------------

class TestICM:
    def test_no_violations(self):
        icm = ICM('test', ['assumption'], [
            GroundingCriterion('x', 'gt', 10.0, rationale='impossible'),
        ])
        violations = icm.check_violations({'x': 0.5})
        assert violations == []

    def test_violation_detected(self):
        icm = ICM('test', ['assumption'], [
            GroundingCriterion('x', 'lt', 1.0, rationale='too small'),
        ])
        violations = icm.check_violations({'x': 0.5})
        assert len(violations) == 1
        assert 'too small' in violations[0]

    def test_serialization_roundtrip(self):
        icm = ICM('test_icm', ['bg1', 'bg2'], [
            GroundingCriterion('x', 'gt', 0.5),
        ])
        d = icm.to_dict()
        icm2 = ICM.from_dict(d)
        assert icm2.name == 'test_icm'
        assert len(icm2.background) == 2
        assert len(icm2.violation_conditions) == 1


# ---------------------------------------------------------------------------
# MetaphorAuditor tests
# ---------------------------------------------------------------------------

class TestMetaphorAuditor:
    def test_audit_returns_all_archetypes(self):
        tel = _make_telemetry(weights=[0.7, -0.7, 0, 0, 0, 0, 0.7, -0.7, 0, 0])
        auditor = MetaphorAuditor(BUILTIN_ARCHETYPES)
        report = auditor.audit(tel)
        assert set(report.keys()) == {'deleuze_fold', 'deleuze_bwo', 'borges_mirror', 'foucault_panopticon'}

    def test_audit_result_structure(self):
        tel = _make_telemetry(weights=[0.7, -0.7, 0, 0, 0, 0, 0.7, -0.7, 0, 0])
        auditor = MetaphorAuditor(BUILTIN_ARCHETYPES)
        report = auditor.audit(tel)
        for name, result in report.items():
            assert 'similarity' in result
            assert 'grounding_pass' in result
            assert 'failed_criteria' in result
            assert 'icm_violated' in result
            assert 'icm_violations' in result
            assert 'verdict' in result
            assert result['verdict'] in ('grounded', 'partial', 'violated')

    def test_verdict_values(self):
        tel = _make_telemetry()
        auditor = MetaphorAuditor(BUILTIN_ARCHETYPES)
        report = auditor.audit(tel)
        verdicts = {r['verdict'] for r in report.values()}
        # At least some verdict should be produced
        assert len(verdicts) >= 1

    def test_archetype_with_no_criteria(self):
        lib = ArchetypeLibrary([
            PersonaArchetype('bare', weight_vector=np.ones(6)),
        ])
        tel = _make_telemetry(weights=[1, 1, 1, 1, 1, 1])
        auditor = MetaphorAuditor(lib)
        report = auditor.audit(tel)
        assert report['bare']['grounding_pass'] is True
        assert report['bare']['icm_violated'] is False
        assert report['bare']['verdict'] == 'grounded'

    def test_similarity_computed(self):
        tel = _make_telemetry(weights=[0.7, -0.7, 0, 0, 0, 0, 0.7, -0.7, 0, 0])
        auditor = MetaphorAuditor(BUILTIN_ARCHETYPES)
        report = auditor.audit(tel)
        # deleuze_fold should have high similarity since weights match
        assert report['deleuze_fold']['similarity'] > 0.8

    def test_layer_warnings_present(self):
        tel = _make_telemetry(weights=[0.7, -0.7, 0, 0, 0, 0, 0.7, -0.7, 0, 0])
        auditor = MetaphorAuditor(BUILTIN_ARCHETYPES)
        report = auditor.audit(tel)
        for name, result in report.items():
            assert 'layer_warnings' in result
            assert isinstance(result['layer_warnings'], list)

    def test_layer_warnings_for_linking_criteria(self):
        """Archetypes that use linking features (phase_lock) in grounding should get warnings."""
        lib = ArchetypeLibrary([
            PersonaArchetype(
                'test_arch',
                weight_vector=np.ones(6),
                grounding_criteria=[
                    GroundingCriterion('phase_lock', 'gt', 0.5, rationale='linking feature in grounding'),
                ],
            ),
        ])
        tel = _make_telemetry(weights=[1, 1, 1, 1, 1, 1])
        auditor = MetaphorAuditor(lib)
        report = auditor.audit(tel)
        assert len(report['test_arch']['layer_warnings']) >= 1
        assert 'phase_lock' in report['test_arch']['layer_warnings'][0]
