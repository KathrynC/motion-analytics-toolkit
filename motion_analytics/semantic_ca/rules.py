"""Local update rules for semantic cellular automata."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import random

from .lattice import Lattice


class Rule(ABC):
    """Abstract base class for a local update rule."""
    
    @abstractmethod
    def apply(self, lattice: Lattice, node_idx: int) -> Dict[str, Any]:
        """
        Compute the new state for a given node based on its neighbors.
        Returns a dictionary of updated attributes (e.g., {'label': 'new_label'}).
        """
        pass


class MajorityVoteRule(Rule):
    """
    Update a node's label to the majority label among its neighbors.
    If there's a tie, keep the current label (or optionally pick randomly).
    """
    
    def __init__(self, label_key: str = 'label', tie_behavior: str = 'keep'):
        self.label_key = label_key
        self.tie_behavior = tie_behavior  # 'keep' or 'random'
    
    def apply(self, lattice: Lattice, node_idx: int) -> Dict[str, Any]:
        neighbors = lattice.neighbors(node_idx)
        if not neighbors:
            return {}  # no change
        
        # Get labels of neighbors
        neighbor_labels = [
            lattice.entries[n].get(self.label_key, 'unknown')
            for n in neighbors
        ]
        # Count
        from collections import Counter
        counts = Counter(neighbor_labels)
        max_count = max(counts.values())
        most_common = [label for label, cnt in counts.items() if cnt == max_count]
        
        current_label = lattice.entries[node_idx].get(self.label_key, 'unknown')
        if len(most_common) == 1:
            new_label = most_common[0]
        else:
            # Tie
            if self.tie_behavior == 'keep':
                new_label = current_label
            else:  # random
                new_label = random.choice(most_common)
        
        if new_label != current_label:
            return {self.label_key: new_label}
        return {}


class PropagationRule(Rule):
    """
    Propagate a property (e.g., confidence, archetype score) by averaging neighbors.
    """
    
    def __init__(self, property_key: str, alpha: float = 0.5):
        """
        Args:
            property_key: Name of the property to propagate.
            alpha: Weight of neighbor average vs current value.
        """
        self.property_key = property_key
        self.alpha = alpha
    
    def apply(self, lattice: Lattice, node_idx: int) -> Dict[str, Any]:
        neighbors = lattice.neighbors(node_idx)
        if not neighbors:
            return {}
        
        current = lattice.entries[node_idx].get(self.property_key, 0.0)
        neighbor_vals = [
            lattice.entries[n].get(self.property_key, current)
            for n in neighbors
        ]
        neighbor_avg = np.mean(neighbor_vals)
        new_val = (1 - self.alpha) * current + self.alpha * neighbor_avg
        return {self.property_key: float(new_val)}


class MutationRule(Rule):
    """
    Randomly mutate a node's feature vector towards/away from neighbors.
    This simulates exploration of the behavior space.
    """
    
    def __init__(self, feature_std: float = 0.1, mutation_prob: float = 0.01):
        self.feature_std = feature_std
        self.mutation_prob = mutation_prob
    
    def apply(self, lattice: Lattice, node_idx: int) -> Dict[str, Any]:
        if random.random() > self.mutation_prob:
            return {}
        
        # Mutate by adding Gaussian noise to the feature vector
        noise = np.random.normal(0, self.feature_std, size=lattice.feature_matrix.shape[1])
        # We don't store features back in entries; this would require a way to update.
        # For now, we just return a marker.
        return {'mutated': True}


class CompositeRule(Rule):
    """Apply multiple rules in sequence."""
    
    def __init__(self, rules: List[Rule]):
        self.rules = rules
    
    def apply(self, lattice: Lattice, node_idx: int) -> Dict[str, Any]:
        updates = {}
        for rule in self.rules:
            new_updates = rule.apply(lattice, node_idx)
            updates.update(new_updates)
        return updates
