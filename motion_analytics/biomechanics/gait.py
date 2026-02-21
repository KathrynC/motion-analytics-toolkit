"""Gait phase detection and symmetry analysis."""

import numpy as np
from typing import Dict
from ..core.base import MotionAnalyzer
from ..core.schemas import Telemetry
from ..core.signal import compute_phase_locking_value

class GaitAnalyzer(MotionAnalyzer):
    """Analyze gait phases, symmetry, and step characteristics."""
    
    def analyze(self, telemetry: Telemetry) -> Dict:
        # Extract contact states
        back_contact = np.array([
            any(c.is_in_contact for c in ts.contacts if c.link_name == 'back_leg')
            for ts in telemetry.timesteps
        ], dtype=bool)
        front_contact = np.array([
            any(c.is_in_contact for c in ts.contacts if c.link_name == 'front_leg')
            for ts in telemetry.timesteps
        ], dtype=bool)
        
        # 1. Gait phases
        duty_back = np.mean(back_contact)
        duty_front = np.mean(front_contact)
        double_support = np.mean(back_contact & front_contact)
        single_support = np.mean(back_contact ^ front_contact)
        flight = np.mean(~back_contact & ~front_contact)
        
        # 2. Symmetry analysis
        # Extract joint angles
        back_angle = np.array([
            ts.joints.get('back_leg_joint', ts.joints.get('joint_0')).angle
            for ts in telemetry.timesteps
        ]) if 'back_leg_joint' in telemetry.timesteps[0].joints else np.zeros(len(telemetry.timesteps))
        front_angle = np.array([
            ts.joints.get('front_leg_joint', ts.joints.get('joint_1')).angle
            for ts in telemetry.timesteps
        ]) if 'front_leg_joint' in telemetry.timesteps[0].joints else np.zeros(len(telemetry.timesteps))
        
        phase_lock = compute_phase_locking_value(back_angle, front_angle)
        symmetry_index = 100 * (duty_back - duty_front) / ((duty_back + duty_front)/2 + 1e-10)
        
        # 3. Step characteristics
        back_strikes = np.where(np.diff(back_contact.astype(int)) == 1)[0]
        if len(back_strikes) > 1:
            step_times = np.diff(back_strikes) / telemetry.sampling_rate
            avg_step_time = np.mean(step_times)
            torso_vel = np.array([ts.com_velocity for ts in telemetry.timesteps])
            avg_speed = np.mean(np.linalg.norm(torso_vel, axis=1))
            step_length = avg_speed * avg_step_time
            step_frequency = 1 / avg_step_time
            step_time_cv = np.std(step_times) / avg_step_time
        else:
            step_length = step_frequency = step_time_cv = 0.0
        
        results = {
            'gait_phases': {
                'duty_factor_back': float(duty_back),
                'duty_factor_front': float(duty_front),
                'double_support': float(double_support),
                'single_support': float(single_support),
                'flight': float(flight)
            },
            'symmetry': {
                'phase_lock': float(phase_lock),
                'symmetry_index': float(symmetry_index),
                'temporal_symmetry': float(1 - abs(symmetry_index)/100)
            },
            'step_characteristics': {
                'step_length': float(step_length),
                'step_frequency': float(step_frequency),
                'step_time_variability': float(step_time_cv),
                'num_strides': len(back_strikes)
            }
        }
        self.results = results
        return results
