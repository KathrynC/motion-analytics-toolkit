"""Base classes for archetype definition and similarity measurement."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import numpy as np
import json
from pathlib import Path

from ..core.schemas import Telemetry
from ..core.signal import compute_phase_locking_value


class Archetype(ABC):
    """
    Abstract base class representing a conceptual archetype (persona, philosophical idea).
    
    Subclasses must implement the `similarity_to` method, which quantifies how close
    a given motion (as telemetry or feature vector) is to this archetype.
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def similarity_to(self, telemetry: Telemetry) -> float:
        """
        Return a similarity score (0 to 1) between the given motion and this archetype.
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON export)."""
        return {
            'name': self.name,
            'description': self.description,
            'type': self.__class__.__name__
        }
    
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
