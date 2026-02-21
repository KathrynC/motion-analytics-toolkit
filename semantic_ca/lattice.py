"""Lattice (graph) construction over motion dictionary entries."""

import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

# Type alias for a similarity function: takes two feature vectors -> float in [0,1]
SimilarityMetric = Callable[[np.ndarray, np.ndarray], float]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity normalized to [0,1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (np.dot(a, b) / (norm_a * norm_b) + 1) / 2  # map from [-1,1] to [0,1]

def euclidean_similarity(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> float:
    """Convert Euclidean distance to similarity: exp(-dist/scale)."""
    dist = np.linalg.norm(a - b)
    return float(np.exp(-dist / scale))

@dataclass
class Lattice:
    """
    A graph where nodes are dictionary entries and edges represent behavioral similarity.
    """
    entries: List[Dict[str, Any]]
    node_ids: List[str]  # unique identifiers (e.g., index or label+seed)
    feature_matrix: np.ndarray  # shape (n_nodes, n_features)
    adjacency: np.ndarray  # binary adjacency matrix (n_nodes x n_nodes)
    similarity_matrix: np.ndarray  # float matrix of similarity scores
    
    @classmethod
    def from_dictionary(
        cls,
        dict_path: Path,
        feature_extractor: Callable[[Dict], np.ndarray],
        similarity_metric: SimilarityMetric = cosine_similarity,
        graph_type: str = 'threshold',
        threshold: float = 0.7,
        k: int = 5
    ) -> 'Lattice':
        """
        Build a lattice from a motion dictionary JSON file.
        
        Args:
            dict_path: Path to JSON file (e.g., motion_gait_dictionary_v2.json).
            feature_extractor: Function that takes an entry dict and returns a feature vector.
            similarity_metric: Function to compute similarity between feature vectors.
            graph_type: 'threshold' or 'knn'.
            threshold: Similarity threshold for graph_type='threshold'.
            k: Number of neighbors for graph_type='knn'.
        """
        with open(dict_path, 'r') as f:
            data = json.load(f)
        
        # If the dictionary is a list of entries, use it directly.
        # If it's a dict with an 'entries' key, adjust.
        if isinstance(data, dict) and 'entries' in data:
            entries = data['entries']
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError("Unsupported dictionary format")
        
        # Extract features
        features = []
        node_ids = []
        for i, entry in enumerate(entries):
            feat = feature_extractor(entry)
            if feat is not None:
                features.append(feat)
                node_ids.append(entry.get('id', str(i)))
        
        feature_matrix = np.array(features)
        n = len(features)
        
        # Compute similarity matrix
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                s = similarity_metric(feature_matrix[i], feature_matrix[j])
                sim[i, j] = sim[j, i] = s
        
        # Build adjacency
        if graph_type == 'threshold':
            adj = (sim >= threshold).astype(int)
        elif graph_type == 'knn':
            adj = np.zeros((n, n))
            for i in range(n):
                # Find k nearest neighbors (excluding self)
                neighbors = np.argsort(-sim[i])[1:k+1]  # descending
                adj[i, neighbors] = 1
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        
        return cls(
            entries=entries,
            node_ids=node_ids,
            feature_matrix=feature_matrix,
            adjacency=adj,
            similarity_matrix=sim
        )
    
    def neighbors(self, node_idx: int) -> List[int]:
        """Return indices of neighbors of a given node."""
        return list(np.where(self.adjacency[node_idx] > 0)[0])
    
    def subgraph(self, node_indices: List[int]) -> 'Lattice':
        """Extract a subgraph containing only the specified nodes."""
        indices = np.array(node_indices)
        new_entries = [self.entries[i] for i in indices]
        new_ids = [self.node_ids[i] for i in indices]
        new_feat = self.feature_matrix[indices]
        new_adj = self.adjacency[np.ix_(indices, indices)]
        new_sim = self.similarity_matrix[np.ix_(indices, indices)]
        return Lattice(
            entries=new_entries,
            node_ids=new_ids,
            feature_matrix=new_feat,
            adjacency=new_adj,
            similarity_matrix=new_sim
        )


# Feature extractors
def extract_beer_features(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Extract a feature vector from Beer analytics (Outcome, Contact, Coordination, Rotation Axis).
    Expected fields in entry: 'beer_analytics' dict with keys.
    """
    beer = entry.get('beer_analytics', {})
    if not beer:
        return None
    
    # Flatten relevant metrics
    feat = []
    # Outcome
    outcome = beer.get('outcome', {})
    feat.extend([
        outcome.get('displacement', 0),
        outcome.get('speed_mean', 0),
        outcome.get('efficiency', 0),
    ])
    # Contact
    contact = beer.get('contact', {})
    feat.extend([
        contact.get('entropy', 0),
        contact.get('duty_cycle_back', 0),
        contact.get('duty_cycle_front', 0),
    ])
    # Coordination
    coord = beer.get('coordination', {})
    feat.extend([
        coord.get('phase_lock', 0),
        coord.get('freq_back', 0),
        coord.get('freq_front', 0),
    ])
    # Rotation axis
    rot = beer.get('rotation_axis', {})
    feat.extend([
        rot.get('roll_dominance', 0),
        rot.get('pitch_dominance', 0),
        rot.get('yaw_dominance', 0),
        rot.get('axis_switching_rate', 0),
    ])
    return np.array(feat)


def extract_label_embedding(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    If labels are available as embeddings (e.g., from a language model),
    you could use them. Placeholder.
    """
    # This would require storing label embeddings in the dictionary.
    return None
