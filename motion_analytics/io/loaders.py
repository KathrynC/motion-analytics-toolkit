"""Telemetry loaders for various data formats."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from ..core.schemas import Telemetry, TimeStep, LinkState, JointState, ContactState
from ..utils.kinematics import compute_leg_positions


def load_er_telemetry(gait_name: str, telemetry_dir: Path) -> Telemetry:
    """
    Load a gait's telemetry from Evolutionary-Robotics JSONL format.
    
    The expected file structure:
        telemetry_dir / gait_name / telemetry.jsonl
    where each line is a JSON record with keys: t, base, rpy, contacts, link_contacts,
    vel, ang_vel, joints.

    Args:
        gait_name: Name of the gait (e.g., "43_hidden_cpg_champion")
        telemetry_dir: Path to directory containing subfolders for each gait
        
    Returns:
        Telemetry object in standardized format
    """
    jsonl_path = telemetry_dir / gait_name / "telemetry.jsonl"
    if not jsonl_path.exists():
        # Try alternative: maybe telemetry is directly in telemetry_dir
        jsonl_path = telemetry_dir / f"{gait_name}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Telemetry file not found: {jsonl_path}")
    
    timesteps = []
    sampling_rate = 240.0  # ER project uses 240 Hz
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            
            t = record['t']
            base = record['base']
            rpy = record['rpy']
            vel = record['vel']
            ang_vel = record['ang_vel']
            
            # Joint data
            joints_dict = {}
            for jd in record.get('joints', []):
                idx = jd['j']
                if idx == 0:
                    name = 'back_leg_joint'
                elif idx == 1:
                    name = 'front_leg_joint'
                else:
                    name = f'joint_{idx}'
                joints_dict[name] = JointState(
                    angle=jd.get('pos', 0.0),
                    velocity=jd.get('vel', 0.0),
                    torque=jd.get('tau', 0.0)
                )
            
            # Link states — torso directly from base; legs via forward kinematics
            torso_pos = (base['x'], base['y'], base['z'])
            torso_rpy = (rpy['r'], rpy['p'], rpy['y'])
            back_angle = joints_dict.get('back_leg_joint',
                                         JointState(angle=0.0, velocity=0.0)).angle
            front_angle = joints_dict.get('front_leg_joint',
                                          JointState(angle=0.0, velocity=0.0)).angle

            back_com, front_com = compute_leg_positions(
                torso_pos, torso_rpy, back_angle, front_angle)

            links = {
                'torso': LinkState(
                    position=list(torso_pos),
                    orientation=[rpy['r'], rpy['p'], rpy['y']],  # keep as RPY for now
                    linear_velocity=[vel['vx'], vel['vy'], vel['vz']],
                    angular_velocity=[ang_vel['wx'], ang_vel['wy'], ang_vel['wz']]
                ),
                'back_leg': LinkState(
                    position=back_com.tolist(),
                    orientation=[0.0, 0.0, 0.0, 1.0],
                    linear_velocity=[0.0, 0.0, 0.0],
                    angular_velocity=[0.0, 0.0, 0.0]
                ),
                'front_leg': LinkState(
                    position=front_com.tolist(),
                    orientation=[0.0, 0.0, 0.0, 1.0],
                    linear_velocity=[0.0, 0.0, 0.0],
                    angular_velocity=[0.0, 0.0, 0.0]
                )
            }
            
            # Contact states from link_contacts
            contacts = []
            link_contacts = record.get('link_contacts', [False, False, False])
            link_names = ['torso', 'back_leg', 'front_leg']
            for i, is_contact in enumerate(link_contacts):
                if is_contact:
                    contacts.append(ContactState(
                        link_name=link_names[i],
                        normal_force=1.0,  # binary indicator
                        friction_force=[0.0, 0.0],
                        contact_point=[0.0, 0.0, 0.0],
                        is_in_contact=True
                    ))
            
            timestep = TimeStep(
                timestamp=t / sampling_rate,
                links=links,
                joints=joints_dict,
                contacts=contacts,
                com_position=[base['x'], base['y'], base['z']],
                com_velocity=[vel['vx'], vel['vy'], vel['vz']]
            )
            timesteps.append(timestep)
    
    return Telemetry(
        metadata={
            'source': 'Evolutionary-Robotics',
            'gait_name': gait_name,
            'sampling_rate': sampling_rate,
            'original_file': str(jsonl_path)
        },
        timesteps=timesteps,
        sampling_rate=sampling_rate
    )


def load_json_telemetry(json_path: Path, sampling_rate: float = 240.0) -> Telemetry:
    """
    Load telemetry from a single JSON file that contains a list of timestep records.
    
    Expected format: a list of dicts, each dict having keys like:
        'timestamp', 'links', 'joints', 'contacts', 'com_position', 'com_velocity'
    This matches the internal Telemetry structure (but as a dict, not dataclass).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Assume data is a dict with 'metadata' and 'timesteps' keys, or just a list of timesteps.
    if isinstance(data, dict) and 'timesteps' in data:
        metadata = data.get('metadata', {})
        timesteps_data = data['timesteps']
        sr = data.get('sampling_rate', sampling_rate)
    elif isinstance(data, list):
        metadata = {}
        timesteps_data = data
        sr = sampling_rate
    else:
        raise ValueError("Unsupported JSON format for telemetry")
    
    # Convert each timestep dict to a TimeStep dataclass
    timesteps = []
    for ts_data in timesteps_data:
        # Convert links
        links = {}
        for name, link_dict in ts_data.get('links', {}).items():
            links[name] = LinkState(**link_dict)
        # Convert joints
        joints = {}
        for name, joint_dict in ts_data.get('joints', {}).items():
            joints[name] = JointState(**joint_dict)
        # Convert contacts
        contacts = [ContactState(**c) for c in ts_data.get('contacts', [])]
        # Create timestep
        timesteps.append(TimeStep(
            timestamp=ts_data['timestamp'],
            links=links,
            joints=joints,
            contacts=contacts,
            com_position=ts_data['com_position'],
            com_velocity=ts_data['com_velocity']
        ))
    
    return Telemetry(
        metadata=metadata,
        timesteps=timesteps,
        sampling_rate=sr
    )


def load_telemetry(path: Union[str, Path], format: Optional[str] = None, **kwargs) -> Telemetry:
    """
    Generic telemetry loader that auto-detects format based on file extension or explicit format.
    
    Args:
        path: Path to telemetry file or directory (for ER format).
        format: One of 'er' (Evolutionary-Robotics folder), 'json', or None (auto-detect).
        **kwargs: Additional arguments passed to the specific loader.
    
    Returns:
        Telemetry object.
    """
    path = Path(path)
    if format is None:
        # Auto-detect
        if path.is_dir():
            # Assume it's an ER gait folder containing telemetry.jsonl
            # In this case, the caller should provide gait_name separately.
            # For auto-detection, we'll raise an error asking for explicit format.
            raise ValueError("For a directory, please specify format='er' and provide gait_name via kwargs.")
        elif path.suffix == '.jsonl':
            format = 'er_jsonl'
        elif path.suffix == '.json':
            format = 'json'
        else:
            raise ValueError(f"Unrecognized file extension: {path.suffix}")
    
    if format == 'er':
        # Evolutionary-Robotics folder format
        gait_name = kwargs.get('gait_name')
        if gait_name is None:
            raise ValueError("gait_name must be provided for ER format")
        return load_er_telemetry(gait_name, path)
    elif format == 'er_jsonl':
        # Direct JSONL file (not in a folder)
        # In this case, the file contains one JSON object per line as in ER format.
        # We'll treat it similarly but without the folder structure.
        return load_er_telemetry_from_file(path)
    elif format == 'json':
        return load_json_telemetry(path, sampling_rate=kwargs.get('sampling_rate', 240.0))
    else:
        raise ValueError(f"Unknown format: {format}")


def load_er_telemetry_from_file(jsonl_path: Path) -> Telemetry:
    """
    Load ER telemetry from a single JSONL file (not inside a gait folder).
    Assumes the file name (without extension) is the gait name.
    """
    gait_name = jsonl_path.stem
    # Create a temporary directory path that is just the parent
    return load_er_telemetry(gait_name, jsonl_path.parent)


def load_motion_dictionary(dict_path: Path) -> List[Dict[str, Any]]:
    """
    Load a motion dictionary JSON file (e.g., motion_gait_dictionary_v2.json).
    
    Returns a list of entries.
    """
    with open(dict_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'entries' in data:
        return data['entries']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unsupported motion dictionary format")
