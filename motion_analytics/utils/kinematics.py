"""Kinematics utilities for the 3‑link robot used in Evolutionary‑Robotics."""

import numpy as np
from typing import Dict, Tuple

def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll‑pitch‑yaw (ZYX order) to a 3x3 rotation matrix.

    Args:
        roll: Rotation around x‑axis (radians)
        pitch: Rotation around y‑axis (radians)
        yaw:   Rotation around z‑axis (radians)

    Returns:
        3x3 rotation matrix.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rotation matrix: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    return R


def compute_leg_positions(
    torso_pos: Tuple[float, float, float],
    torso_rpy: Tuple[float, float, float],
    back_angle: float,
    front_angle: float,
    joint_offset_back: Tuple[float, float, float] = (-0.5, 0.0, 0.0),
    joint_offset_front: Tuple[float, float, float] = (0.5, 0.0, 0.0),
    leg_length: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the positions of the back leg and front leg centers of mass
    for the 3‑link robot used in the Evolutionary‑Robotics project.

    Assumptions:
        - Torso is 1 m long, legs are 1 m long.
        - Joints are located at the ends of the torso, with offsets given.
        - Leg links are rigid, with COM at half the leg length from the joint.
        - Joint axes are the y‑axis (so rotation occurs in the x‑z plane).
        - Leg initial orientation is along the torso's x‑axis.

    Args:
        torso_pos:   (x, y, z) position of the torso center.
        torso_rpy:   (roll, pitch, yaw) orientation of the torso.
        back_angle:  Joint angle of the back leg (radians).
        front_angle: Joint angle of the front leg (radians).
        joint_offset_back:  (dx, dy, dz) from torso center to back joint.
        joint_offset_front: (dx, dy, dz) from torso center to front joint.
        leg_length:  Total length of each leg (distance from joint to tip).

    Returns:
        Tuple of two numpy arrays: (back_leg_com, front_leg_com), each (3,).
    """
    torso_pos = np.array(torso_pos)
    joint_offset_back = np.array(joint_offset_back)
    joint_offset_front = np.array(joint_offset_front)

    # Rotation matrix of the torso
    R_torso = rpy_to_rotation_matrix(*torso_rpy)

    # Joint positions in world frame
    back_joint_world = torso_pos + R_torso @ joint_offset_back
    front_joint_world = torso_pos + R_torso @ joint_offset_front

    # Torso axes (columns of R_torso)
    torso_x = R_torso[:, 0]
    torso_y = R_torso[:, 1]   # axis of rotation
    torso_z = R_torso[:, 2]

    # Function to rotate a vector around y‑axis by given angle
    def rotate_around_y(vec: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a vector around the y‑axis by angle (radians)."""
        # Rotation matrix about y‑axis
        ca, sa = np.cos(angle), np.sin(angle)
        # Since vec is expressed in the torso frame, we rotate its components.
        # Alternatively, we can use the torso_x and torso_z directly.
        # New direction = cos(angle)*torso_x + sin(angle)*torso_z   (positive angle lifts leg)
        return ca * vec + sa * torso_z

    # Directions from each joint to the leg COM
    back_dir = rotate_around_y(torso_x, back_angle)
    front_dir = rotate_around_y(torso_x, front_angle)

    # Leg COM is at half the leg length from the joint along that direction
    back_com = back_joint_world + (leg_length / 2.0) * back_dir
    front_com = front_joint_world + (leg_length / 2.0) * front_dir

    return back_com, front_com


def compute_link_positions(
    torso_pos: Tuple[float, float, float],
    torso_rpy: Tuple[float, float, float],
    back_angle: float,
    front_angle: float,
    link_lengths: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # torso, back leg, front leg
) -> Dict[str, np.ndarray]:
    """
    Compute positions of all three links: torso center, back leg COM, front leg COM.

    Returns a dictionary with keys 'torso', 'back_leg', 'front_leg', each a (3,) numpy array.
    """
    # Torso center is given
    torso_center = np.array(torso_pos)

    # Leg COMs
    back_com, front_com = compute_leg_positions(
        torso_pos, torso_rpy, back_angle, front_angle,
        leg_length=link_lengths[1]  # assuming leg length is the same for both
    )

    return {
        'torso': torso_center,
        'back_leg': back_com,
        'front_leg': front_com
    }
