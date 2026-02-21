"""High-level analyzer that matches a motion against an archetype library."""

from typing import Dict, Optional
import numpy as np

from .base import ArchetypeLibrary
from .persona import PersonaArchetype
from ..core.base import MotionAnalyzer
from ..core.schemas import Telemetry


class StructuralTransferAnalyzer(MotionAnalyzer):
    """
    Analyzer that, given a motion (telemetry), returns its similarity to
    a library of archetypes. This operationalizes the structural transfer hypothesis.
    """
    
    def __init__(self, library: ArchetypeLibrary, config: Optional[Dict] = None):
        super().__init__(config)
        self.library = library
    
    def analyze(self, telemetry: Telemetry) -> Dict:
        """
        Return a dictionary with:
          - 'best_match': {'name': str, 'score': float}
          - 'similarities': dict mapping archetype names to scores
          - 'all_scores': list of (name, score)
        """
        sim_vector = self.library.similarity_vector(telemetry)
        best_name, best_score = max(sim_vector.items(), key=lambda x: x[1])
        
        # Also compute weight-space similarity if weights available (for debug)
        weight_sim = None
        weights = telemetry.metadata.get('synapse_weights')
        if weights is not None:
            # Find archetype with closest weight vector
            best_weight_arch = None
            best_weight_score = -1
            for arch in self.library.archetypes:
                if isinstance(arch, PersonaArchetype) and arch.weight_vector is not None:
                    w = np.array(weights).flatten()
                    ref = arch.weight_vector.flatten()
                    min_len = min(len(w), len(ref))
                    w = w[:min_len]
                    ref = ref[:min_len]
                    norm = np.linalg.norm(w) * np.linalg.norm(ref)
                    if norm == 0:
                        continue
                    cos = np.dot(w, ref) / norm
                    score = (cos + 1) / 2
                    if score > best_weight_score:
                        best_weight_score = score
                        best_weight_arch = arch.name
            if best_weight_arch:
                weight_sim = {'name': best_weight_arch, 'score': best_weight_score}
        
        self.results = {
            'best_match': {'name': best_name, 'score': best_score},
            'similarities': sim_vector,
            'all_scores': sorted(sim_vector.items(), key=lambda x: -x[1]),
            'weight_space_match': weight_sim
        }
        return self.results
