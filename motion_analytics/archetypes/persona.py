"""Archetypes based on persona weight vectors from your PERSONAS.md data."""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Any

from .base import Archetype, ArchetypeLibrary, extract_behavioral_features
from ..core.schemas import Telemetry


class PersonaArchetype(Archetype):
    """
    Archetype defined by a specific weight vector (or set of vectors)
    that produce motions characteristic of a persona (e.g., 'deleuze_fold').
    Similarity can be computed either in weight space (if weights are available)
    or in behavioral feature space.
    """
    
    def __init__(
        self,
        name: str,
        weight_vector: Optional[np.ndarray] = None,
        feature_vector: Optional[np.ndarray] = None,
        description: str = ""
    ):
        super().__init__(name, description)
        self.weight_vector = weight_vector  # shape (6,) or (10,) for crosswired
        self.feature_vector = feature_vector  # precomputed behavioral features
        
    def similarity_to(self, telemetry: Telemetry) -> float:
        """
        Compute similarity between this archetype and the given motion.
        
        Strategy:
          - If telemetry contains the weight vector (in metadata), use cosine similarity in weight space.
          - Otherwise, extract behavioral features from telemetry and compare to stored feature vector.
        """
        # Try to get weights from telemetry metadata (if this run was generated from a known weight set)
        weights = telemetry.metadata.get('synapse_weights')
        if weights is not None and self.weight_vector is not None:
            # Weight-space similarity (cosine)
            w = np.array(weights).flatten()
            ref = self.weight_vector.flatten()
            # Pad/crop if dimensions differ (6 vs 10)
            min_len = min(len(w), len(ref))
            w = w[:min_len]
            ref = ref[:min_len]
            norm_product = np.linalg.norm(w) * np.linalg.norm(ref)
            if norm_product == 0:
                return 0.0
            cos_sim = np.dot(w, ref) / norm_product
            return float((cos_sim + 1) / 2)  # map from [-1,1] to [0,1]
        
        # Otherwise, use behavioral feature space
        if self.feature_vector is None:
            # This should not happen; archetype should have features precomputed.
            return 0.0
        
        # Extract features from telemetry
        feats = extract_behavioral_features(telemetry)
        vec = np.array([feats[k] for k in self._feature_keys()])
        # Normalize vectors (using precomputed norm if stored)
        # Simple cosine similarity
        norm_product = np.linalg.norm(vec) * np.linalg.norm(self.feature_vector)
        if norm_product == 0:
            return 0.0
        cos_sim = np.dot(vec, self.feature_vector) / norm_product
        return float((cos_sim + 1) / 2)
    
    def _feature_keys(self) -> list:
        """Ordered list of feature keys used in the feature vector."""
        return [
            'straightness', 'curvature_complexity', 'workspace_volume',
            'phase_lock', 'symmetry_index', 'duty_factor_asymmetry', 'flight_fraction',
            'cost_of_transport', 'peak_power', 'efficiency'
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data['weight_vector'] = self.weight_vector.tolist() if self.weight_vector is not None else None
        data['feature_vector'] = self.feature_vector.tolist() if self.feature_vector is not None else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonaArchetype':
        wv = np.array(data['weight_vector']) if data.get('weight_vector') else None
        fv = np.array(data['feature_vector']) if data.get('feature_vector') else None
        return cls(
            name=data['name'],
            weight_vector=wv,
            feature_vector=fv,
            description=data.get('description', '')
        )


def load_persona_library(json_path: Path) -> ArchetypeLibrary:
    """
    Load a persona library from a JSON file containing entries for each persona gait.
    
    Expected format:
    [
        {
            "name": "deleuze_fold",
            "weight_vector": [0.7, -0.3, ...],
            "description": "Fold concept: self-feedback, etc.",
            "feature_vector": [...]   (optional, will be computed if missing)
        },
        ...
    ]
    
    If feature_vector is missing, it will be computed on the fly (requires telemetry).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lib = ArchetypeLibrary()
    for item in data:
        arch = PersonaArchetype.from_dict(item)
        lib.add(arch)
    return lib
