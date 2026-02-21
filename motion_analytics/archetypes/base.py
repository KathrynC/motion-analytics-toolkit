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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature': self.feature,
            'predicate': self.predicate,
            'value': self.value,
            'tolerance': self.tolerance,
            'rationale': self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundingCriterion':
        return cls(**data)


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
        """Load library from JSON file (requires custom deserialization)."""
        with open(path, 'r') as f:
            data = json.load(f)
        # This is a stub – subclasses should implement from_dict properly.
        lib = cls()
        for item in data:
            if item['type'] == 'PersonaArchetype':
                # We'll need PersonaArchetype imported here, but we'll handle later.
                # For now, just skip.
                pass
        return lib


# ----------------------------------------------------------------------
# Feature extraction helpers (used by concrete archetypes)
# ----------------------------------------------------------------------

def extract_behavioral_features(telemetry: Telemetry) -> Dict[str, float]:
    """
    Compute a set of scalar features that characterise a motion.
    These are used for similarity measurement when weights are not available.
    """
    # Use existing analyzers (they return dictionaries)
    from ..kinematics.path import PathAnalyzer
    from ..biomechanics.gait import GaitAnalyzer
    from ..biomechanics.energetics import EnergeticsAnalyzer
    
    path = PathAnalyzer().analyze(telemetry)
    gait = GaitAnalyzer().analyze(telemetry)
    energy = EnergeticsAnalyzer().analyze(telemetry)
    
    features = {
        # Path features
        'straightness': path['path_efficiency']['straightness'],
        'curvature_complexity': path['path_curvature']['complexity'],
        'workspace_volume': path['workspace']['volume'],
        
        # Gait features
        'phase_lock': gait['symmetry']['phase_lock'],
        'symmetry_index': gait['symmetry']['symmetry_index'],
        'duty_factor_asymmetry': abs(gait['gait_phases']['duty_factor_back'] -
                                     gait['gait_phases']['duty_factor_front']),
        'flight_fraction': gait['gait_phases']['flight'],
        
        # Energy features
        'cost_of_transport': energy['cost_of_transport'],
        'peak_power': energy['peak_power'],
        'efficiency': energy['efficiency'],
    }
    return features
