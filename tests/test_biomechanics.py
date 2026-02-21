"""Tests for biomechanics module: gait analysis and energetics."""

import numpy as np
import pytest

from motion_analytics.core.schemas import (
    Telemetry, TimeStep, LinkState, JointState, ContactState,
)
from motion_analytics.biomechanics.gait import GaitAnalyzer
from motion_analytics.biomechanics.energetics import EnergeticsAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ts(t, x, back_angle, front_angle, back_on, front_on, vx=0.01):
    contacts = []
    if back_on:
        contacts.append(ContactState(
            link_name='back_leg', normal_force=1.0,
            friction_force=[0.0, 0.0], contact_point=[0.0, 0.0, 0.0],
            is_in_contact=True,
        ))
    if front_on:
        contacts.append(ContactState(
            link_name='front_leg', normal_force=1.0,
            friction_force=[0.0, 0.0], contact_point=[0.0, 0.0, 0.0],
            is_in_contact=True,
        ))
    return TimeStep(
        timestamp=t,
        links={
            'torso': LinkState([x, 0, 0.1], [0, 0, 0, 1], [vx, 0, 0], [0, 0, 0]),
            'back_leg': LinkState([x - 0.5, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]),
            'front_leg': LinkState([x + 0.5, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]),
        },
        joints={
            'back_leg_joint': JointState(angle=back_angle, velocity=0.5, torque=0.1),
            'front_leg_joint': JointState(angle=front_angle, velocity=0.5, torque=0.1),
        },
        contacts=contacts,
        com_position=[x, 0.0, 0.1],
        com_velocity=[vx, 0.0, 0.0],
    )


def _alternating_telemetry(n=200, sr=240.0):
    """Telemetry with alternating back/front contact — idealized gait."""
    steps = []
    for i in range(n):
        t = i / sr
        x = i * 0.001
        freq = 2.0
        phase = 2 * np.pi * freq * t
        back_angle = 0.3 * np.sin(phase)
        front_angle = 0.3 * np.sin(phase + np.pi)
        back_on = np.sin(phase) > 0
        front_on = np.sin(phase + np.pi) > 0
        steps.append(_make_ts(t, x, back_angle, front_angle, back_on, front_on))
    return Telemetry(metadata={}, timesteps=steps, sampling_rate=sr)


def _stationary_telemetry(n=100, sr=240.0):
    """Telemetry with no motion and constant double support."""
    steps = []
    for i in range(n):
        steps.append(_make_ts(i / sr, 0.0, 0.0, 0.0, True, True, vx=0.0))
    return Telemetry(metadata={}, timesteps=steps, sampling_rate=sr)


# ---------------------------------------------------------------------------
# GaitAnalyzer tests
# ---------------------------------------------------------------------------

class TestGaitAnalyzer:
    def test_returns_all_keys(self):
        tel = _alternating_telemetry()
        result = GaitAnalyzer().analyze(tel)
        assert 'gait_phases' in result
        assert 'symmetry' in result
        assert 'step_characteristics' in result

    def test_duty_factors_sum_reasonable(self):
        result = GaitAnalyzer().analyze(_alternating_telemetry())
        gp = result['gait_phases']
        # Duty factors should be in [0, 1]
        assert 0.0 <= gp['duty_factor_back'] <= 1.0
        assert 0.0 <= gp['duty_factor_front'] <= 1.0

    def test_double_support_stationary(self):
        result = GaitAnalyzer().analyze(_stationary_telemetry())
        assert result['gait_phases']['double_support'] == 1.0

    def test_phase_lock_range(self):
        result = GaitAnalyzer().analyze(_alternating_telemetry())
        plv = result['symmetry']['phase_lock']
        assert 0.0 <= plv <= 1.0

    def test_symmetry_index_bounded(self):
        result = GaitAnalyzer().analyze(_alternating_telemetry())
        si = result['symmetry']['symmetry_index']
        # Should be within ±200 (percentage-based)
        assert -200 <= si <= 200

    def test_step_count_positive(self):
        result = GaitAnalyzer().analyze(_alternating_telemetry())
        assert result['step_characteristics']['num_strides'] >= 0


# ---------------------------------------------------------------------------
# EnergeticsAnalyzer tests
# ---------------------------------------------------------------------------

class TestEnergeticsAnalyzer:
    def test_returns_all_keys(self):
        result = EnergeticsAnalyzer().analyze(_alternating_telemetry())
        assert 'total_work' in result
        assert 'cost_of_transport' in result
        assert 'efficiency' in result
        assert 'peak_power' in result
        assert 'joint_work' in result

    def test_total_work_nonnegative(self):
        result = EnergeticsAnalyzer().analyze(_alternating_telemetry())
        assert result['total_work'] >= 0.0

    def test_cost_of_transport_nonnegative(self):
        result = EnergeticsAnalyzer().analyze(_alternating_telemetry())
        assert result['cost_of_transport'] >= 0

    def test_energy_variance_keys(self):
        result = EnergeticsAnalyzer().analyze(_alternating_telemetry())
        ev = result['energy_variance']
        assert 'kinetic' in ev
        assert 'potential' in ev
        assert 'total' in ev

    def test_joint_work_per_joint(self):
        result = EnergeticsAnalyzer().analyze(_alternating_telemetry())
        jw = result['joint_work']
        assert 'back_leg_joint' in jw
        assert 'front_leg_joint' in jw

    def test_stationary_low_work(self):
        result = EnergeticsAnalyzer().analyze(_stationary_telemetry())
        # Stationary robot should have near-zero work
        assert result['total_work'] < 1.0
