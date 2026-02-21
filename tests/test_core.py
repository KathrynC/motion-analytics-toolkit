"""Tests for core module: schemas, base classes, signal processing."""

import numpy as np
import pytest
import json
import tempfile
from pathlib import Path

from motion_analytics.core.schemas import LinkState, JointState, ContactState, TimeStep, Telemetry
from motion_analytics.core.base import MotionAnalyzer, CompositeAnalyzer
from motion_analytics.core.signal import (
    compute_phase_difference,
    compute_phase_locking_value,
    compute_spectral_arc_length,
    compute_dimensionless_jerk,
    detect_peaks_with_prominence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_link(x=0.0, y=0.0, z=0.0):
    return LinkState(
        position=[x, y, z],
        orientation=[0.0, 0.0, 0.0, 1.0],
        linear_velocity=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
    )


def make_timestep(t, x=0.0, y=0.0, z=0.1, back_angle=0.0, front_angle=0.0,
                  back_contact=False, front_contact=False):
    contacts = []
    if back_contact:
        contacts.append(ContactState(
            link_name='back_leg', normal_force=1.0,
            friction_force=[0.0, 0.0], contact_point=[0.0, 0.0, 0.0],
            is_in_contact=True,
        ))
    if front_contact:
        contacts.append(ContactState(
            link_name='front_leg', normal_force=1.0,
            friction_force=[0.0, 0.0], contact_point=[0.0, 0.0, 0.0],
            is_in_contact=True,
        ))
    return TimeStep(
        timestamp=t,
        links={
            'torso': make_link(x, y, z),
            'back_leg': make_link(x - 0.5, y, 0.0),
            'front_leg': make_link(x + 0.5, y, 0.0),
        },
        joints={
            'back_leg_joint': JointState(angle=back_angle, velocity=0.0, torque=0.1),
            'front_leg_joint': JointState(angle=front_angle, velocity=0.0, torque=0.1),
        },
        contacts=contacts,
        com_position=[x, y, z],
        com_velocity=[0.01, 0.0, 0.0],
    )


def make_telemetry(n_steps=100, sampling_rate=240.0):
    """Create a simple telemetry object with sinusoidal leg motion."""
    timesteps = []
    for i in range(n_steps):
        t = i / sampling_rate
        x = i * 0.001  # slow forward motion
        back_angle = 0.3 * np.sin(2 * np.pi * 2.0 * t)  # 2 Hz
        front_angle = 0.3 * np.sin(2 * np.pi * 2.0 * t + np.pi)  # antiphase
        back_contact = np.sin(2 * np.pi * 2.0 * t) > 0
        front_contact = np.sin(2 * np.pi * 2.0 * t + np.pi) > 0
        timesteps.append(make_timestep(
            t, x=x, back_angle=back_angle, front_angle=front_angle,
            back_contact=back_contact, front_contact=front_contact,
        ))
    return Telemetry(
        metadata={'source': 'test', 'synapse_weights': [0.5, -0.5, 0.3, -0.3, 0.1, -0.1]},
        timesteps=timesteps,
        sampling_rate=sampling_rate,
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_link_state_creation(self):
        ls = LinkState(position=[1, 2, 3], orientation=[0, 0, 0, 1],
                       linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        assert ls.position == [1, 2, 3]

    def test_joint_state_defaults(self):
        js = JointState(angle=0.5, velocity=1.0)
        assert js.torque == 0.0
        assert js.force == 0.0

    def test_contact_state(self):
        cs = ContactState(link_name='back_leg', normal_force=10.0,
                          friction_force=[1.0, 0.0], contact_point=[0, 0, 0],
                          is_in_contact=True)
        assert cs.is_in_contact

    def test_timestep_has_all_fields(self):
        ts = make_timestep(0.0)
        assert ts.timestamp == 0.0
        assert 'torso' in ts.links
        assert 'back_leg_joint' in ts.joints

    def test_telemetry_metadata(self):
        tel = make_telemetry(10)
        assert tel.metadata['source'] == 'test'
        assert len(tel.timesteps) == 10
        assert tel.sampling_rate == 240.0


# ---------------------------------------------------------------------------
# Base analyzer tests
# ---------------------------------------------------------------------------

class TestBaseAnalyzer:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            MotionAnalyzer()

    def test_concrete_analyzer(self):
        class DummyAnalyzer(MotionAnalyzer):
            def analyze(self, telemetry):
                self.results = {'value': 42}
                return self.results

        a = DummyAnalyzer()
        tel = make_telemetry(10)
        result = a.analyze(tel)
        assert result['value'] == 42

    def test_composite_analyzer(self):
        class ConstAnalyzer(MotionAnalyzer):
            def __init__(self, val):
                super().__init__()
                self._val = val
            def analyze(self, telemetry):
                self.results = {'v': self._val}
                return self.results

        comp = CompositeAnalyzer({'a': ConstAnalyzer(1), 'b': ConstAnalyzer(2)})
        result = comp.analyze(make_telemetry(10))
        assert result['a']['v'] == 1
        assert result['b']['v'] == 2

    def test_json_compatible(self):
        class NumpyAnalyzer(MotionAnalyzer):
            def analyze(self, telemetry):
                self.results = {'arr': np.array([1, 2, 3]), 'f': np.float64(3.14)}
                return self.results

        a = NumpyAnalyzer()
        a.analyze(make_telemetry(10))
        j = a.to_json_compatible()
        assert j['arr'] == [1, 2, 3]
        assert isinstance(j['f'], float)

    def test_save_results(self):
        class SimpleAnalyzer(MotionAnalyzer):
            def analyze(self, telemetry):
                self.results = {'x': 1}
                return self.results

        a = SimpleAnalyzer()
        a.analyze(make_telemetry(10))
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            a.save_results(f.name)
            f.seek(0)
        data = json.loads(Path(f.name).read_text())
        assert data['x'] == 1


# ---------------------------------------------------------------------------
# Signal processing tests
# ---------------------------------------------------------------------------

class TestSignal:
    def test_phase_difference_zero_for_identical(self):
        sig = np.sin(np.linspace(0, 4 * np.pi, 500))
        pd = compute_phase_difference(sig, sig)
        assert abs(pd) < 0.01

    def test_phase_difference_pi_for_antiphase(self):
        t = np.linspace(0, 4 * np.pi, 500)
        sig1 = np.sin(t)
        sig2 = np.sin(t + np.pi)
        pd = compute_phase_difference(sig1, sig2)
        assert abs(abs(pd) - np.pi) < 0.1

    def test_plv_identical_signals(self):
        sig = np.sin(np.linspace(0, 4 * np.pi, 500))
        plv = compute_phase_locking_value(sig, sig)
        assert plv > 0.95

    def test_plv_range(self):
        sig1 = np.sin(np.linspace(0, 4 * np.pi, 500))
        sig2 = np.sin(np.linspace(0, 6 * np.pi, 500))
        plv = compute_phase_locking_value(sig1, sig2)
        assert 0.0 <= plv <= 1.0

    def test_spectral_arc_length_nonnegative(self):
        t = np.linspace(0, 1, 240)
        smooth = np.sin(2 * np.pi * t)
        sal = compute_spectral_arc_length(smooth, 240.0)
        assert sal >= 0.0

    def test_dimensionless_jerk_smooth(self):
        t = np.linspace(0, 1, 240)
        position = 0.5 * t  # constant velocity — minimal jerk
        jerk = compute_dimensionless_jerk(position, 240.0)
        assert jerk >= 0.0

    def test_dimensionless_jerk_zero_amplitude(self):
        position = np.zeros(100)
        jerk = compute_dimensionless_jerk(position, 240.0)
        assert jerk == 0.0

    def test_detect_peaks(self):
        t = np.linspace(0, 2 * np.pi, 500)
        data = np.sin(t)
        peaks, props = detect_peaks_with_prominence(data, 500.0 / (2 * np.pi))
        assert len(peaks) >= 1

    def test_detect_peaks_prominence_filter(self):
        t = np.linspace(0, 4 * np.pi, 1000)
        data = np.sin(t) + 0.1 * np.sin(10 * t)
        peaks, _ = detect_peaks_with_prominence(data, 1000.0 / (4 * np.pi), prominence=0.5)
        # Should find roughly 2 large peaks, not the small ones
        assert len(peaks) >= 1
        assert len(peaks) <= 4
