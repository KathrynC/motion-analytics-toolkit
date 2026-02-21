"""Core data structures for motion analytics toolkit.
Inspired by the telemetry format from Evolutionary-Robotics project.
"""

from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

@dataclass
class LinkState:
    """State of a rigid body link at a timestep."""
    position: List[float]      # [x, y, z] in meters
    orientation: List[float]   # [x, y, z, w] quaternion
    linear_velocity: List[float]
    angular_velocity: List[float]
    
@dataclass
class JointState:
    """State of a joint at a timestep."""
    angle: float                # radians
    velocity: float             # rad/s
    torque: float = 0.0         # N·m (if available)
    force: float = 0.0          # N (if available, for prismatic joints)

@dataclass
class ContactState:
    """Ground contact information for a link."""
    link_name: str
    normal_force: float         # N
    friction_force: List[float] # [fx, fy]
    contact_point: List[float]  # [x, y, z]
    is_in_contact: bool

@dataclass
class TimeStep:
    """Single timestep of motion data."""
    timestamp: float            # seconds since start
    links: Dict[str, LinkState]  # e.g., {"torso": LinkState, "back_leg": LinkState}
    joints: Dict[str, JointState] # e.g., {"back_leg_joint": JointState}
    contacts: List[ContactState]
    com_position: List[float]   # center of mass [x, y, z]
    com_velocity: List[float]
    
@dataclass
class Telemetry:
    """Complete telemetry for a single trial."""
    metadata: Dict[str, Any]     # simulator info, parameters, date
    timesteps: List[TimeStep]
    sampling_rate: float         # Hz (240 in your ER project)
    
    def to_array(self, field: str) -> np.ndarray:
        """Extract a specific field across all timesteps as numpy array."""
        # Implementation will depend on field structure
        pass
