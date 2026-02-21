"""Mechanical work, power, and efficiency analysis."""

import numpy as np
from ..core.base import MotionAnalyzer
from ..core.schemas import Telemetry

class EnergeticsAnalyzer(MotionAnalyzer):
    """Compute mechanical work, power, and efficiency."""
    
    def analyze(self, telemetry: Telemetry) -> Dict:
        mass = 3.0  # kg (from ER spec)
        g = 9.8     # m/s²
        
        positions = np.array([ts.com_position for ts in telemetry.timesteps])
        velocities = np.array([ts.com_velocity for ts in telemetry.timesteps])
        
        # Kinetic energy
        kinetic = 0.5 * mass * np.sum(velocities**2, axis=1)
        
        # Potential energy
        height = positions[:, 2]
        potential = mass * g * height
        
        # Total mechanical energy
        total_energy = kinetic + potential
        
        dt = 1 / telemetry.sampling_rate
        power = np.gradient(total_energy, dt)
        work = np.trapz(np.abs(power), dx=dt)
        
        # Displacement
        displacement = np.linalg.norm(positions[-1] - positions[0])
        
        # Cost of transport
        cot = work / (mass * g * displacement) if displacement > 0 else np.inf
        efficiency = 1 / cot if cot > 0 else 0
        
        # Joint work (if torque available)
        joint_work = {}
        for joint_name in telemetry.timesteps[0].joints:
            torque = np.array([ts.joints[joint_name].torque for ts in telemetry.timesteps])
            velocity = np.array([ts.joints[joint_name].velocity for ts in telemetry.timesteps])
            power_joint = torque * velocity
            work_joint = np.trapz(np.abs(power_joint), dx=dt)
            joint_work[joint_name] = float(work_joint)
        
        results = {
            'total_work': float(work),
            'cost_of_transport': float(cot),
            'efficiency': float(efficiency),
            'peak_power': float(np.max(np.abs(power))),
            'mean_power': float(np.mean(np.abs(power))),
            'energy_variance': {
                'kinetic': float(np.std(kinetic)),
                'potential': float(np.std(potential)),
                'total': float(np.std(total_energy))
            },
            'joint_work': joint_work
        }
        self.results = results
        return results
