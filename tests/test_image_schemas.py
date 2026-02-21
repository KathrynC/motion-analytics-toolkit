"""Tests for image schema detectors (Lakoff experiential structures)."""

import numpy as np
import pytest

from motion_analytics.core.image_schemas import ImageSchema, ImageSchemaDetector
from motion_analytics.core.schemas import (
    Telemetry, TimeStep, LinkState, JointState, ContactState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(t, x=0.0, y=0.0, z=0.1, back_angle=0.0, front_angle=0.0,
             back_contact=True, front_contact=False):
    contacts = []
    if back_contact:
        contacts.append(ContactState('back_leg', 1.0, [0, 0], [0, 0, 0], True))
    if front_contact:
        contacts.append(ContactState('front_leg', 1.0, [0, 0], [0, 0, 0], True))
    return TimeStep(
        timestamp=t,
        links={
            'torso': LinkState([x, y, z], [0, 0, 0, 1], [0.01, 0, 0], [0, 0, 0]),
            'back_leg': LinkState([x - 0.5, y, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]),
            'front_leg': LinkState([x + 0.5, y, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]),
        },
        joints={
            'back_leg_joint': JointState(angle=back_angle, velocity=0.5, torque=0.1),
            'front_leg_joint': JointState(angle=front_angle, velocity=0.5, torque=0.2),
        },
        contacts=contacts,
        com_position=[x, y, z],
        com_velocity=[0.01, 0.0, 0.0],
    )


def _make_telemetry(n=200, sr=240.0):
    steps = []
    for i in range(n):
        t = i / sr
        x = i * 0.001
        freq = 2.0
        phase = 2 * np.pi * freq * t
        back_angle = 0.3 * np.sin(phase)
        front_angle = 0.3 * np.sin(phase + np.pi)
        back_contact = np.sin(phase) > 0
        front_contact = np.sin(phase + np.pi) > 0
        steps.append(_make_ts(
            t, x=x,
            back_angle=back_angle, front_angle=front_angle,
            back_contact=back_contact, front_contact=front_contact,
        ))
    return Telemetry(metadata={}, timesteps=steps, sampling_rate=sr)


# ---------------------------------------------------------------------------
# ImageSchema dataclass tests
# ---------------------------------------------------------------------------

class TestImageSchemaDataclass:
    def test_create(self):
        s = ImageSchema('PATH', {'length': 1.5})
        assert s.name == 'PATH'
        assert s.metrics['length'] == 1.5

    def test_empty_metrics(self):
        s = ImageSchema('CYCLE')
        assert s.metrics == {}


# ---------------------------------------------------------------------------
# PATH schema tests
# ---------------------------------------------------------------------------

class TestPathSchema:
    def test_detects_forward_motion(self):
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_path(tel)
        assert schema.name == 'PATH'
        assert schema.metrics['displacement'] > 0
        assert schema.metrics['path_length'] > 0

    def test_straightness_bounded(self):
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_path(tel)
        assert 0.0 <= schema.metrics['straightness'] <= 1.0 + 1e-6

    def test_curvature_integral_nonnegative(self):
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_path(tel)
        assert schema.metrics['curvature_integral'] >= 0.0


# ---------------------------------------------------------------------------
# CYCLE schema tests
# ---------------------------------------------------------------------------

class TestCycleSchema:
    def test_detects_frequency(self):
        tel = _make_telemetry(n=500)
        schema = ImageSchemaDetector().detect_cycle(tel)
        assert schema.name == 'CYCLE'
        # Should detect ~2 Hz dominant frequency
        assert schema.metrics['dominant_frequency'] > 0

    def test_cycle_count_positive(self):
        tel = _make_telemetry(n=500)
        schema = ImageSchemaDetector().detect_cycle(tel)
        assert schema.metrics['cycle_count'] > 0

    def test_regularity_bounded(self):
        tel = _make_telemetry(n=500)
        schema = ImageSchemaDetector().detect_cycle(tel)
        assert 0.0 <= schema.metrics['regularity'] <= 1.0


# ---------------------------------------------------------------------------
# CONTACT schema tests
# ---------------------------------------------------------------------------

class TestContactSchema:
    def test_contact_fraction_range(self):
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_contact(tel)
        assert schema.name == 'CONTACT'
        assert 0.0 <= schema.metrics['contact_fraction'] <= 1.0

    def test_contact_transitions_nonnegative(self):
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_contact(tel)
        assert schema.metrics['contact_transitions'] >= 0

    def test_contact_symmetry_range(self):
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_contact(tel)
        assert 0.0 <= schema.metrics['contact_symmetry'] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# BALANCE schema tests
# ---------------------------------------------------------------------------

class TestBalanceSchema:
    def test_balance_metrics_exist(self):
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_balance(tel)
        assert schema.name == 'BALANCE'
        assert 'com_height_variance' in schema.metrics
        assert 'lateral_sway' in schema.metrics
        assert 'vertical_oscillation' in schema.metrics

    def test_variance_nonnegative(self):
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_balance(tel)
        assert schema.metrics['com_height_variance'] >= 0.0
        assert schema.metrics['lateral_sway'] >= 0.0
        assert schema.metrics['vertical_oscillation'] >= 0.0


# ---------------------------------------------------------------------------
# FORCE schema tests
# ---------------------------------------------------------------------------

class TestForceSchema:
    def test_force_metrics_exist(self):
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_force(tel)
        assert schema.name == 'FORCE'
        assert 'peak_torque' in schema.metrics
        assert 'mean_torque' in schema.metrics
        assert 'torque_asymmetry' in schema.metrics

    def test_torque_nonnegative(self):
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_force(tel)
        assert schema.metrics['peak_torque'] >= 0.0
        assert schema.metrics['mean_torque'] >= 0.0

    def test_asymmetry_detects_unequal_torques(self):
        # Our fixture has torque=0.1 for back and torque=0.2 for front
        tel = _make_telemetry()
        schema = ImageSchemaDetector().detect_force(tel)
        assert schema.metrics['torque_asymmetry'] > 0.0


# ---------------------------------------------------------------------------
# detect_all integration test
# ---------------------------------------------------------------------------

class TestDetectAll:
    def test_returns_all_five_schemas(self):
        tel = _make_telemetry()
        schemas = ImageSchemaDetector().detect_all(tel)
        assert set(schemas.keys()) == {'PATH', 'CYCLE', 'CONTACT', 'BALANCE', 'FORCE'}
        for name, schema in schemas.items():
            assert isinstance(schema, ImageSchema)
            assert schema.name == name
