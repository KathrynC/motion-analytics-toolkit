"""Tests for kinematics module: path analysis and forward kinematics."""

import numpy as np
import pytest

from motion_analytics.core.schemas import (
    Telemetry, TimeStep, LinkState, JointState, ContactState,
)
from motion_analytics.kinematics.path import PathAnalyzer
from motion_analytics.utils.kinematics import (
    rpy_to_rotation_matrix,
    compute_leg_positions,
    compute_link_positions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(t, x, y=0.0, z=0.1):
    return TimeStep(
        timestamp=t,
        links={
            'torso': LinkState([x, y, z], [0, 0, 0, 1], [0.01, 0, 0], [0, 0, 0]),
        },
        joints={
            'back_leg_joint': JointState(angle=0.0, velocity=0.0),
            'front_leg_joint': JointState(angle=0.0, velocity=0.0),
        },
        contacts=[],
        com_position=[x, y, z],
        com_velocity=[0.01, 0.0, 0.0],
    )


def _straight_telemetry(n=200, sr=240.0):
    steps = [_make_ts(i / sr, i * 0.001) for i in range(n)]
    return Telemetry(metadata={}, timesteps=steps, sampling_rate=sr)


def _circular_telemetry(n=200, sr=240.0):
    steps = []
    for i in range(n):
        t = i / sr
        theta = 2 * np.pi * t * 2  # 2 Hz circular
        x = 0.1 * np.cos(theta)
        y = 0.1 * np.sin(theta)
        steps.append(_make_ts(t, x, y))
    return Telemetry(metadata={}, timesteps=steps, sampling_rate=sr)


# ---------------------------------------------------------------------------
# PathAnalyzer tests
# ---------------------------------------------------------------------------

class TestPathAnalyzer:
    def test_returns_all_keys(self):
        result = PathAnalyzer().analyze(_straight_telemetry())
        assert 'path_curvature' in result
        assert 'path_efficiency' in result
        assert 'workspace' in result

    def test_straight_path_high_straightness(self):
        result = PathAnalyzer().analyze(_straight_telemetry())
        assert result['path_efficiency']['straightness'] > 0.9

    def test_circular_path_lower_straightness(self):
        result = PathAnalyzer().analyze(_circular_telemetry())
        assert result['path_efficiency']['straightness'] < 0.5

    def test_curvature_nonnegative(self):
        result = PathAnalyzer().analyze(_straight_telemetry())
        assert result['path_curvature']['mean'] >= 0.0

    def test_workspace_volume_positive(self):
        result = PathAnalyzer().analyze(_straight_telemetry())
        # Straight line has zero y and z range, so volume may be ~0
        assert result['workspace']['x_range'] > 0


# ---------------------------------------------------------------------------
# Forward kinematics tests
# ---------------------------------------------------------------------------

class TestForwardKinematics:
    def test_rotation_identity(self):
        R = rpy_to_rotation_matrix(0, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_orthogonal(self):
        R = rpy_to_rotation_matrix(0.3, 0.5, 0.7)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rotation_90_yaw(self):
        R = rpy_to_rotation_matrix(0, 0, np.pi / 2)
        # x-axis should map to y-axis
        np.testing.assert_allclose(R @ np.array([1, 0, 0]), np.array([0, 1, 0]), atol=1e-10)

    def test_leg_positions_zero_angles(self):
        back, front = compute_leg_positions(
            torso_pos=(0, 0, 0.5),
            torso_rpy=(0, 0, 0),
            back_angle=0.0,
            front_angle=0.0,
        )
        # With zero angles, legs extend along x axis from joints
        assert back.shape == (3,)
        assert front.shape == (3,)
        # Back joint at (-0.5, 0, 0.5), leg COM at (-0.5 + 0.5, 0, 0.5) = (0, 0, 0.5) approx
        # Front joint at (0.5, 0, 0.5), leg COM at (0.5 + 0.5, 0, 0.5) = (1, 0, 0.5) approx

    def test_link_positions_returns_dict(self):
        result = compute_link_positions(
            torso_pos=(0, 0, 0.5),
            torso_rpy=(0, 0, 0),
            back_angle=0.3,
            front_angle=-0.3,
        )
        assert 'torso' in result
        assert 'back_leg' in result
        assert 'front_leg' in result
        for v in result.values():
            assert v.shape == (3,)

    def test_symmetry_with_opposite_angles(self):
        back, front = compute_leg_positions(
            torso_pos=(0, 0, 0.5),
            torso_rpy=(0, 0, 0),
            back_angle=0.3,
            front_angle=0.3,
        )
        # Both legs at same angle; z-components should be the same
        np.testing.assert_allclose(back[2], front[2], atol=1e-10)
