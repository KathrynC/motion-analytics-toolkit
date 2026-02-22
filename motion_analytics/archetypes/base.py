"""Base classes for archetype definition and similarity measurement."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import json
from pathlib import Path

from ..core.schemas import Telemetry
from ..core.signal import compute_phase_locking_value


# ------------------------------------------------------------------
# Feature layer classification (Lakoff Maxim 7)
# ------------------------------------------------------------------

# Maps canonical feature names to their Lakoff layer.
# Grounded = directly observable sensorimotor features.
# Linking = cross-domain abstractions requiring interpretation.
FEATURE_LAYERS: Dict[str, str] = {
    # --- Grounded (sensorimotor, directly observable) ---
    # From path/kinematics
    'straightness': 'grounded',
    'curvature_complexity': 'grounded',
    'workspace_volume': 'grounded',
    'mean_speed': 'grounded',
    'speed_cv': 'grounded',
    'dx': 'grounded',
    'dy': 'grounded',
    'displacement': 'grounded',
    'yaw_net_rad': 'grounded',
    'yaw_degrees': 'grounded',
    'contact_entropy_bits': 'grounded',
    'duty_factor_asymmetry': 'grounded',
    'flight_fraction': 'grounded',
    'cost_of_transport': 'grounded',
    'peak_power': 'grounded',
    'efficiency': 'grounded',
    # From image schemas (promoted top-level)
    'cycle_count': 'grounded',
    'cycle_regularity': 'grounded',
    'dominant_frequency': 'grounded',
    'contact_fraction': 'grounded',
    'contact_transitions': 'grounded',
    'lateral_sway': 'grounded',
    'vertical_oscillation': 'grounded',
    'torque_asymmetry': 'grounded',
    # All schema.* prefixed features are grounded
    # (handled by prefix rule in get_feature_layer())

    # --- Linking (cross-domain abstraction) ---
    'phase_lock': 'linking',
    'symmetry_index': 'linking',
    'phase_lock_score': 'linking',
}


def get_feature_layer(feature_name: str) -> str:
    """Return 'grounded' or 'linking' for a feature name.

    All schema.* prefixed features are grounded.
    Aliases inherit from their canonical feature.
    Unknown features default to 'linking' (conservative: must be grounded to be grounded).
    """
    if feature_name.startswith('schema.'):
        return 'grounded'
    canonical = FEATURE_ALIASES.get(feature_name, feature_name)
    return FEATURE_LAYERS.get(canonical, 'linking')


# ------------------------------------------------------------------
# Lakoff grounding structures
# ------------------------------------------------------------------

@dataclass
class GroundingCriterion:
    """A testable predicate that anchors a label to observable features.

    Following Lakoff Maxim 7 (ground first, link second): every metaphorical
    label must be grounded in sensorimotor observables before cross-domain
    linking is permitted.
    """
    feature: str           # behavioral feature key, e.g. 'phase_lock'
    predicate: str         # 'gt', 'lt', 'between', 'near'
    value: float           # threshold or target
    tolerance: float = 0.0 # for 'near' predicate
    rationale: str = ""    # why this criterion grounds the label
    layer: str = ""        # 'grounded', 'linking', or '' (unconstrained)

    def check(self, features: Dict[str, float]) -> bool:
        """Return True if the criterion is satisfied by the given features."""
        actual = features.get(self.feature)
        if actual is None:
            return False
        if self.predicate == 'gt':
            return actual > self.value
        elif self.predicate == 'lt':
            return actual < self.value
        elif self.predicate == 'near':
            return abs(actual - self.value) <= self.tolerance
        elif self.predicate == 'between':
            # value encodes low bound, tolerance encodes high bound
            return self.value <= actual <= self.tolerance
        return False

    def layer_warning(self) -> str:
        """Return warning if this criterion claims grounded layer but references a linking feature."""
        if self.layer == 'grounded' and get_feature_layer(self.feature) == 'linking':
            return (f"Grounding criterion on '{self.feature}' claims grounded layer "
                    f"but feature is classified as linking")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'feature': self.feature,
            'predicate': self.predicate,
            'value': self.value,
            'tolerance': self.tolerance,
            'rationale': self.rationale,
        }
        if self.layer:
            d['layer'] = self.layer
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundingCriterion':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ICM:
    """Idealized Cognitive Model — background assumptions a label presupposes.

    When violation_conditions are met, the label's ICM breaks and the label
    should not be applied (or should be flagged as metaphor violation).
    """
    name: str
    background: List[str]                          # prose assumptions
    violation_conditions: List[GroundingCriterion]  # when the label breaks

    def check_violations(self, features: Dict[str, float]) -> List[str]:
        """Return list of violated condition descriptions (empty = ICM intact)."""
        violations = []
        for cond in self.violation_conditions:
            if cond.check(features):
                violations.append(
                    f"{cond.feature} {cond.predicate} {cond.value}: {cond.rationale}"
                )
        return violations

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'background': self.background,
            'violation_conditions': [c.to_dict() for c in self.violation_conditions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ICM':
        return cls(
            name=data['name'],
            background=data['background'],
            violation_conditions=[
                GroundingCriterion.from_dict(c) for c in data['violation_conditions']
            ],
        )


# ------------------------------------------------------------------
# Archetype base class
# ------------------------------------------------------------------

class Archetype(ABC):
    """
    Abstract base class representing a conceptual archetype (persona, philosophical idea).

    Subclasses must implement the `similarity_to` method, which quantifies how close
    a given motion (as telemetry or feature vector) is to this archetype.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        grounding_criteria: Optional[List[GroundingCriterion]] = None,
        icm: Optional[ICM] = None,
    ):
        self.name = name
        self.description = description
        self.grounding_criteria = grounding_criteria or []
        self.icm = icm

    @abstractmethod
    def similarity_to(self, telemetry: Telemetry) -> float:
        """
        Return a similarity score (0 to 1) between the given motion and this archetype.
        """
        pass

    def check_grounding(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Test all grounding criteria against extracted features.

        Returns:
            (all_pass, list_of_failure_descriptions)
        """
        failures = []
        for gc in self.grounding_criteria:
            if not gc.check(features):
                failures.append(
                    f"{gc.feature} failed {gc.predicate} {gc.value}: {gc.rationale}"
                )
        return (len(failures) == 0, failures)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON export)."""
        d = {
            'name': self.name,
            'description': self.description,
            'type': self.__class__.__name__,
        }
        if self.grounding_criteria:
            d['grounding_criteria'] = [gc.to_dict() for gc in self.grounding_criteria]
        if self.icm:
            d['icm'] = self.icm.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Archetype':
        """Deserialize from dictionary (to be overridden by subclasses)."""
        raise NotImplementedError


class ArchetypeLibrary:
    """
    Collection of archetypes with methods to load/save and query.
    """
    
    def __init__(self, archetypes: Optional[List[Archetype]] = None):
        self.archetypes = archetypes or []
        self._name_index = {a.name: a for a in self.archetypes}
    
    def add(self, archetype: Archetype):
        self.archetypes.append(archetype)
        self._name_index[archetype.name] = archetype
    
    def get(self, name: str) -> Optional[Archetype]:
        return self._name_index.get(name)
    
    def best_match(self, telemetry: Telemetry) -> tuple[Archetype, float]:
        """
        Find the archetype with highest similarity to the given motion.
        Returns (archetype, score).
        """
        best_score = -1.0
        best_arch = None
        for arch in self.archetypes:
            score = arch.similarity_to(telemetry)
            if score > best_score:
                best_score = score
                best_arch = arch
        return best_arch, best_score
    
    def similarity_vector(self, telemetry: Telemetry) -> Dict[str, float]:
        """Return a dict mapping archetype names to similarity scores."""
        return {a.name: a.similarity_to(telemetry) for a in self.archetypes}
    
    def save(self, path: Path):
        """Save library to JSON file."""
        data = [a.to_dict() for a in self.archetypes]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ArchetypeLibrary':
        """Load library from JSON file.

        Currently supports PersonaArchetype entries. Unknown types are skipped.
        """
        from .persona import PersonaArchetype

        with open(path, 'r') as f:
            data = json.load(f)
        lib = cls()
        for item in data:
            if item.get('type') == 'PersonaArchetype':
                lib.add(PersonaArchetype.from_dict(item))
        return lib


# ----------------------------------------------------------------------
# Feature extraction helpers (used by concrete archetypes)
# ----------------------------------------------------------------------

# Aliases mapping Beer-analytics / JSON variant names to canonical toolkit keys.
# When a feature appears under multiple spellings (e.g. 'path_straightness' in
# the Beer system vs 'straightness' in the toolkit), FEATURE_ALIASES maps the
# variant(s) to the canonical key that extract_behavioral_features() produces.
FEATURE_ALIASES: Dict[str, str] = {
    'path_straightness': 'straightness',
    'contact_entropy': 'contact_entropy_bits',
    'entropy': 'contact_entropy_bits',
    'phase_lock_score': 'phase_lock',
}


def extract_behavioral_features(telemetry: Telemetry) -> Dict[str, float]:
    """
    Compute scalar features that characterise a motion.

    Returns a flat dict containing:
      - The 10 canonical toolkit features (straightness, curvature_complexity, …)
      - Beer-compatible features (mean_speed, speed_cv, dx, dy, displacement,
        yaw_net_rad, yaw_degrees, contact_entropy_bits)
      - Image schema metrics (prefixed as schema.NAME.metric and 8 promoted top-level)
      - Spelling aliases so that grounding criteria using Beer-style names
        (path_straightness, contact_entropy, entropy, phase_lock_score)
        resolve correctly.
    """
    from ..kinematics.path import PathAnalyzer
    from ..biomechanics.gait import GaitAnalyzer
    from ..biomechanics.energetics import EnergeticsAnalyzer
    from ..core.image_schemas import ImageSchemaDetector

    path = PathAnalyzer().analyze(telemetry)
    gait = GaitAnalyzer().analyze(telemetry)
    energy = EnergeticsAnalyzer().analyze(telemetry)

    # Image schema detection (makes pattern language arrow real)
    schemas = ImageSchemaDetector().detect_all(telemetry)

    # --- 10 canonical toolkit features (unchanged) -----------------------
    features: Dict[str, float] = {
        'straightness': path['path_efficiency']['straightness'],
        'curvature_complexity': path['path_curvature']['complexity'],
        'workspace_volume': path['workspace']['volume'],
        'phase_lock': gait['symmetry']['phase_lock'],
        'symmetry_index': gait['symmetry']['symmetry_index'],
        'duty_factor_asymmetry': abs(gait['gait_phases']['duty_factor_back'] -
                                     gait['gait_phases']['duty_factor_front']),
        'flight_fraction': gait['gait_phases']['flight'],
        'cost_of_transport': energy['cost_of_transport'],
        'peak_power': energy['peak_power'],
        'efficiency': energy['efficiency'],
    }

    # --- Beer-compatible features ----------------------------------------
    positions = np.array([ts.com_position for ts in telemetry.timesteps])
    velocities = np.array([ts.com_velocity for ts in telemetry.timesteps])
    dt = 1.0 / telemetry.sampling_rate

    # Speed
    speed = np.linalg.norm(velocities, axis=1)
    mean_speed = float(np.mean(speed))
    features['mean_speed'] = mean_speed
    features['speed_cv'] = float(np.std(speed) / (mean_speed + 1e-10))

    # Displacement components
    features['dx'] = float(positions[-1, 0] - positions[0, 0])
    features['dy'] = float(positions[-1, 1] - positions[0, 1])
    features['displacement'] = float(np.linalg.norm(positions[-1] - positions[0]))

    # Yaw from body orientation quaternions
    # Extract yaw from torso quaternion at each timestep
    yaw_vals = []
    for ts in telemetry.timesteps:
        torso = ts.links.get('torso')
        if torso is not None:
            qx, qy, qz, qw = torso.orientation
            # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
            yaw = float(np.arctan2(2 * (qw * qz + qx * qy),
                                   1 - 2 * (qy**2 + qz**2)))
            yaw_vals.append(yaw)
    if len(yaw_vals) >= 2:
        yaw_net = yaw_vals[-1] - yaw_vals[0]
    else:
        yaw_net = 0.0
    features['yaw_net_rad'] = float(yaw_net)
    features['yaw_degrees'] = float(np.degrees(yaw_net))

    # Contact entropy (Shannon entropy of the 4 contact-state distribution)
    back_contact = np.array([
        any(c.is_in_contact for c in ts.contacts if c.link_name == 'back_leg')
        for ts in telemetry.timesteps
    ], dtype=int)
    front_contact = np.array([
        any(c.is_in_contact for c in ts.contacts if c.link_name == 'front_leg')
        for ts in telemetry.timesteps
    ], dtype=int)
    # 4 states: 00, 01, 10, 11
    state_codes = back_contact * 2 + front_contact
    counts = np.bincount(state_codes, minlength=4).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    features['contact_entropy_bits'] = float(-np.sum(probs * np.log2(probs)))

    # --- Image schema metrics (prefixed + promoted top-level) -----------
    for schema_name, schema in schemas.items():
        for metric_name, value in schema.metrics.items():
            features[f'schema.{schema_name}.{metric_name}'] = float(value)

    # Promote 8 previously-unused schema metrics as top-level features
    features['cycle_count'] = float(schemas['CYCLE'].metrics.get('cycle_count', 0.0))
    features['cycle_regularity'] = float(schemas['CYCLE'].metrics.get('regularity', 0.0))
    features['dominant_frequency'] = float(schemas['CYCLE'].metrics.get('dominant_frequency', 0.0))
    features['contact_fraction'] = float(schemas['CONTACT'].metrics.get('contact_fraction', 0.0))
    features['contact_transitions'] = float(schemas['CONTACT'].metrics.get('contact_transitions', 0.0))
    features['lateral_sway'] = float(schemas['BALANCE'].metrics.get('lateral_sway', 0.0))
    features['vertical_oscillation'] = float(schemas['BALANCE'].metrics.get('vertical_oscillation', 0.0))
    features['torque_asymmetry'] = float(schemas['FORCE'].metrics.get('torque_asymmetry', 0.0))

    # --- Spelling aliases ------------------------------------------------
    for alias, canonical in FEATURE_ALIASES.items():
        features[alias] = features[canonical]

    return features
