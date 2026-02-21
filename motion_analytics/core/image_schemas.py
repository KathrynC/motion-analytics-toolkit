"""Image schema detectors grounded in Lakoff's experiential structures.

Each detector extracts a pre-conceptual schema (PATH, CYCLE, CONTACT, BALANCE, FORCE)
from telemetry data as a first-class structure with named scalar metrics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict

from .schemas import Telemetry


@dataclass
class ImageSchema:
    """A detected image schema with its quantified metrics."""
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)


class ImageSchemaDetector:
    """Detect Lakoff image schemas in motion telemetry.

    Five schemas from embodied cognition:
      PATH   — source-path-goal structure of trajectories
      CYCLE  — periodic repetition in joint motion
      CONTACT — ground interaction patterns
      BALANCE — postural stability of the center of mass
      FORCE  — torque and effort dynamics at joints
    """

    def detect_all(self, telemetry: Telemetry) -> Dict[str, ImageSchema]:
        """Run all five schema detectors."""
        return {
            'PATH': self.detect_path(telemetry),
            'CYCLE': self.detect_cycle(telemetry),
            'CONTACT': self.detect_contact(telemetry),
            'BALANCE': self.detect_balance(telemetry),
            'FORCE': self.detect_force(telemetry),
        }

    # ------------------------------------------------------------------
    # PATH — source-path-goal
    # ------------------------------------------------------------------
    def detect_path(self, telemetry: Telemetry) -> ImageSchema:
        positions = np.array([ts.com_position for ts in telemetry.timesteps])
        start = positions[0]
        end = positions[-1]

        displacement = float(np.linalg.norm(end - start))
        segments = np.diff(positions[:, :2], axis=0)
        path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
        straightness = displacement / (path_length + 1e-10)

        # Curvature integral (2D)
        dx = np.gradient(positions[:, 0])
        dy = np.gradient(positions[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = (dx**2 + dy**2 + 1e-10) ** 1.5
        curvature = np.abs(dx * ddy - dy * ddx) / denom
        dt = 1.0 / telemetry.sampling_rate
        curvature_integral = float(np.trapezoid(curvature, dx=dt))

        return ImageSchema('PATH', {
            'path_length': path_length,
            'displacement': displacement,
            'straightness': float(straightness),
            'curvature_integral': curvature_integral,
        })

    # ------------------------------------------------------------------
    # CYCLE — periodic repetition
    # ------------------------------------------------------------------
    def detect_cycle(self, telemetry: Telemetry) -> ImageSchema:
        # Use joint angles for periodicity detection
        joint_names = list(telemetry.timesteps[0].joints.keys())
        if not joint_names:
            return ImageSchema('CYCLE', {
                'dominant_frequency': 0.0,
                'cycle_count': 0.0,
                'regularity': 0.0,
            })

        # Take the first joint's angle signal
        angles = np.array([ts.joints[joint_names[0]].angle
                           for ts in telemetry.timesteps])
        angles = angles - np.mean(angles)

        n = len(angles)
        sr = telemetry.sampling_rate
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        spectrum = np.abs(np.fft.rfft(angles))

        # Ignore DC
        spectrum[0] = 0.0
        if len(spectrum) < 2:
            return ImageSchema('CYCLE', {
                'dominant_frequency': 0.0,
                'cycle_count': 0.0,
                'regularity': 0.0,
            })

        peak_idx = int(np.argmax(spectrum))
        dominant_freq = float(freqs[peak_idx])
        duration = n / sr
        cycle_count = dominant_freq * duration

        # Regularity: fraction of spectral energy in the dominant peak (±1 bin)
        lo = max(1, peak_idx - 1)
        hi = min(len(spectrum), peak_idx + 2)
        peak_energy = float(np.sum(spectrum[lo:hi] ** 2))
        total_energy = float(np.sum(spectrum[1:] ** 2)) + 1e-10
        regularity = peak_energy / total_energy

        return ImageSchema('CYCLE', {
            'dominant_frequency': dominant_freq,
            'cycle_count': float(cycle_count),
            'regularity': float(regularity),
        })

    # ------------------------------------------------------------------
    # CONTACT — ground interaction
    # ------------------------------------------------------------------
    def detect_contact(self, telemetry: Telemetry) -> ImageSchema:
        n = len(telemetry.timesteps)
        contact_counts = []
        link_contacts: Dict[str, int] = {}

        for ts in telemetry.timesteps:
            active = [c for c in ts.contacts if c.is_in_contact]
            contact_counts.append(len(active))
            for c in active:
                link_contacts[c.link_name] = link_contacts.get(c.link_name, 0) + 1

        contact_array = np.array(contact_counts, dtype=float)
        contact_fraction = float(np.mean(contact_array > 0))

        # Transitions: timesteps where contact count changes
        transitions = int(np.sum(np.abs(np.diff(contact_array)) > 0))

        # Symmetry: how evenly distributed contacts are across links
        if link_contacts:
            counts = np.array(list(link_contacts.values()), dtype=float)
            total = counts.sum() + 1e-10
            probs = counts / total
            max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            contact_symmetry = entropy / (max_entropy + 1e-10)
        else:
            contact_symmetry = 0.0

        return ImageSchema('CONTACT', {
            'contact_fraction': contact_fraction,
            'contact_transitions': float(transitions),
            'contact_symmetry': float(contact_symmetry),
        })

    # ------------------------------------------------------------------
    # BALANCE — postural stability
    # ------------------------------------------------------------------
    def detect_balance(self, telemetry: Telemetry) -> ImageSchema:
        positions = np.array([ts.com_position for ts in telemetry.timesteps])

        height = positions[:, 2]
        com_height_variance = float(np.var(height))

        # Lateral sway: std of horizontal displacement from mean
        lateral = positions[:, 1]  # y-axis
        lateral_sway = float(np.std(lateral))

        # Vertical oscillation: peak-to-peak of height
        vertical_oscillation = float(np.ptp(height))

        return ImageSchema('BALANCE', {
            'com_height_variance': com_height_variance,
            'lateral_sway': lateral_sway,
            'vertical_oscillation': vertical_oscillation,
        })

    # ------------------------------------------------------------------
    # FORCE — effort dynamics
    # ------------------------------------------------------------------
    def detect_force(self, telemetry: Telemetry) -> ImageSchema:
        joint_names = list(telemetry.timesteps[0].joints.keys())
        if not joint_names:
            return ImageSchema('FORCE', {
                'peak_torque': 0.0,
                'mean_torque': 0.0,
                'torque_asymmetry': 0.0,
            })

        all_torques = {}
        for jn in joint_names:
            all_torques[jn] = np.array([ts.joints[jn].torque
                                        for ts in telemetry.timesteps])

        # Aggregate across joints
        torque_magnitudes = np.concatenate([np.abs(t) for t in all_torques.values()])
        peak_torque = float(np.max(torque_magnitudes))
        mean_torque = float(np.mean(torque_magnitudes))

        # Asymmetry: difference in mean torque between joints
        if len(joint_names) >= 2:
            means = [float(np.mean(np.abs(all_torques[jn]))) for jn in joint_names]
            torque_asymmetry = float(max(means) - min(means))
        else:
            torque_asymmetry = 0.0

        return ImageSchema('FORCE', {
            'peak_torque': peak_torque,
            'mean_torque': mean_torque,
            'torque_asymmetry': torque_asymmetry,
        })
