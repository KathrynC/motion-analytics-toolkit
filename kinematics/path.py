"""Path analysis for motion trajectories."""

import numpy as np
from typing import Dict, List, Tuple
from ..core.base import MotionAnalyzer
from ..core.schemas import Telemetry

class PathAnalyzer(MotionAnalyzer):
    """Analyze the path taken by the center of mass."""
    
    def analyze(self, telemetry: Telemetry) -> Dict:
        """
        Compute path-based metrics.
        
        Returns:
            Dictionary with keys:
            - path_curvature: mean, max, integral, complexity
            - path_efficiency: straightness, transmission ratio
            - workspace: volume, ranges, explored ratio
        """
        # Extract COM positions
        positions = np.array([ts.com_position for ts in telemetry.timesteps])
        
        # 1. Path Curvature
        dx = np.gradient(positions[:, 0])
        dy = np.gradient(positions[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        denominator = (dx**2 + dy**2 + 1e-10)**1.5
        curvature = np.abs(dx * ddy - dy * ddx) / denominator
        
        path_curvature = {
            'mean': float(np.mean(curvature)),
            'max': float(np.max(curvature)),
            'integral': float(np.trapz(curvature, dx=1/telemetry.sampling_rate)),
            'complexity': float(np.std(curvature))
        }
        
        # 2. Path Efficiency
        start_pos = positions[0]
        end_pos = positions[-1]
        straight_distance = np.linalg.norm(end_pos - start_pos)
        
        # Actual path length (2D)
        path_2d = positions[:, :2]
        path_length = np.sum(np.linalg.norm(np.diff(path_2d, axis=0), axis=1))
        
        # Joint motion (proxy for mechanical work)
        total_joint_motion = 0.0
        for ts in telemetry.timesteps:
            for joint in ts.joints.values():
                total_joint_motion += abs(joint.angle)
        
        path_efficiency = {
            'straightness': float(straight_distance / (path_length + 1e-10)),
            'transmission_ratio': float(path_length / (total_joint_motion + 1e-10)),
            'curvilinear_ratio': float(path_length / straight_distance)
        }
        
        # 3. Workspace Utilization
        all_positions = positions.flatten()
        workspace = {
            'x_range': float(np.ptp(positions[:, 0])),
            'y_range': float(np.ptp(positions[:, 1])),
            'z_range': float(np.ptp(positions[:, 2])),
            'volume': float(np.prod(np.ptp(positions, axis=0))),
            'explored_ratio': float(len(np.unique(positions, axis=0)) / len(positions))
        }
        
        self.results = {
            'path_curvature': path_curvature,
            'path_efficiency': path_efficiency,
            'workspace': workspace
        }
        
        return self.results
