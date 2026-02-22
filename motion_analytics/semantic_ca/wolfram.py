"""Classify semantic CA dynamics into Wolfram classes (1-4).

Wolfram's four classes of CA behavior:
  Class 1 (Fixed):    All cells converge to a single state.
  Class 2 (Periodic): Cells settle into stable or oscillating patterns.
  Class 3 (Chaotic):  Cells evolve pseudo-randomly with no stable structure.
  Class 4 (Complex):  Localized structures emerge and interact (edge of chaos).

The classifier evolves a lattice under a rule for N generations, measuring
label entropy, cluster count, boundary activity, and order parameter at each
step. These trajectories are then scored against characteristic signatures
of each Wolfram class.
"""

import copy
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .lattice import Lattice
from .rules import Rule


@dataclass
class WolframEvidence:
    """Metrics collected during CA evolution for classification."""
    label_entropy: np.ndarray       # entropy at each generation
    cluster_counts: np.ndarray      # number of label-based clusters at each generation
    boundary_counts: np.ndarray     # number of boundary nodes at each generation
    order_parameter: np.ndarray     # fraction in majority label at each generation
    generations: int

    # Derived statistics (computed by compute_derived)
    entropy_decay_rate: float = 0.0
    entropy_final: float = 0.0
    entropy_variance_tail: float = 0.0
    cluster_stability: float = 0.0
    boundary_activity: float = 0.0
    order_convergence: float = 0.0

    def compute_derived(self):
        """Compute derived statistics from raw trajectory arrays."""
        n = self.generations
        if n < 2:
            return

        self.entropy_final = float(self.label_entropy[-1])
        e0 = float(self.label_entropy[0])

        # Entropy decay rate: how much entropy dropped relative to initial
        if e0 > 1e-10:
            self.entropy_decay_rate = float(1.0 - self.entropy_final / e0)
        else:
            # Started at zero entropy — already fixed
            self.entropy_decay_rate = 1.0

        # Tail variance: variance of entropy in the last quarter of generations
        tail_start = max(1, 3 * n // 4)
        tail = self.label_entropy[tail_start:]
        self.entropy_variance_tail = float(np.var(tail)) if len(tail) > 1 else 0.0

        # Cluster stability: 1 - (std of cluster counts in tail / mean)
        tail_clusters = self.cluster_counts[tail_start:]
        mean_c = float(np.mean(tail_clusters))
        if mean_c > 1e-10:
            cv = float(np.std(tail_clusters)) / mean_c
            self.cluster_stability = float(max(0.0, 1.0 - cv))
        else:
            self.cluster_stability = 1.0

        # Boundary activity: fraction of generations (in tail) with nonzero boundaries
        tail_boundaries = self.boundary_counts[tail_start:]
        total_nodes = max(1, int(np.max(self.boundary_counts)) + 1) if len(self.boundary_counts) > 0 else 1
        if len(tail_boundaries) > 0:
            # Normalized: mean boundary fraction
            mean_b = float(np.mean(tail_boundaries))
            self.boundary_activity = mean_b / max(total_nodes, 1)
        else:
            self.boundary_activity = 0.0

        # Order convergence: final order parameter
        self.order_convergence = float(self.order_parameter[-1])


class WolframClassifier:
    """Classify semantic CA rule dynamics into Wolfram classes.

    Run a rule on a lattice for N generations, measuring:
    - Label entropy trajectory (do labels converge?)
    - Cluster count trajectory (do clusters stabilize?)
    - Boundary activity (are boundaries static or dynamic?)
    - Order parameter (does one label dominate?)
    """

    def __init__(self, generations: int = 100, label_key: str = 'label'):
        self.generations = generations
        self.label_key = label_key

    def _label_entropy(self, labels: List[str]) -> float:
        """Shannon entropy of label distribution."""
        if not labels:
            return 0.0
        counts = Counter(labels)
        n = len(labels)
        probs = np.array([c / n for c in counts.values()])
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def _order_parameter(self, labels: List[str]) -> float:
        """Fraction of nodes with the majority label."""
        if not labels:
            return 0.0
        counts = Counter(labels)
        return float(max(counts.values()) / len(labels))

    def _label_clusters(self, lattice: Lattice) -> int:
        """Count connected components where adjacent nodes share a label.

        A label-cluster is a maximal connected set of nodes that all have the
        same label AND are connected through adjacency edges.
        """
        n = len(lattice.entries)
        if n == 0:
            return 0
        visited = [False] * n
        cluster_count = 0
        for start in range(n):
            if visited[start]:
                continue
            visited[start] = True
            cluster_count += 1
            label = lattice.entries[start].get(self.label_key, '')
            stack = [start]
            while stack:
                node = stack.pop()
                for nb in lattice.neighbors(node):
                    if not visited[nb] and lattice.entries[nb].get(self.label_key, '') == label:
                        visited[nb] = True
                        stack.append(nb)
        return cluster_count

    def _boundary_count(self, lattice: Lattice) -> int:
        """Count nodes that have at least one neighbor with a different label."""
        count = 0
        for i in range(len(lattice.entries)):
            my_label = lattice.entries[i].get(self.label_key, '')
            for nb in lattice.neighbors(i):
                if lattice.entries[nb].get(self.label_key, '') != my_label:
                    count += 1
                    break
        return count

    def evolve_and_measure(self, lattice: Lattice, rule: Rule) -> WolframEvidence:
        """Run the CA and collect trajectory data.

        The lattice entries are deep-copied so the original is not modified.
        Updates are applied synchronously: all new states are computed before
        any are applied.
        """
        # Deep-copy entries so we don't mutate the caller's lattice
        lattice = Lattice(
            entries=[copy.deepcopy(e) for e in lattice.entries],
            node_ids=list(lattice.node_ids),
            feature_matrix=lattice.feature_matrix.copy(),
            adjacency=lattice.adjacency.copy(),
            similarity_matrix=lattice.similarity_matrix.copy(),
            feature_layers=dict(lattice.feature_layers),
        )

        gens = self.generations
        entropy_arr = np.zeros(gens)
        cluster_arr = np.zeros(gens, dtype=int)
        boundary_arr = np.zeros(gens, dtype=int)
        order_arr = np.zeros(gens)

        for g in range(gens):
            labels = [e.get(self.label_key, '') for e in lattice.entries]
            entropy_arr[g] = self._label_entropy(labels)
            cluster_arr[g] = self._label_clusters(lattice)
            boundary_arr[g] = self._boundary_count(lattice)
            order_arr[g] = self._order_parameter(labels)

            # Synchronous update: compute all updates, then apply
            updates = []
            for i in range(len(lattice.entries)):
                upd = rule.apply(lattice, i)
                updates.append(upd)
            for i, upd in enumerate(updates):
                lattice.entries[i].update(upd)

        evidence = WolframEvidence(
            label_entropy=entropy_arr,
            cluster_counts=cluster_arr,
            boundary_counts=boundary_arr,
            order_parameter=order_arr,
            generations=gens,
        )
        evidence.compute_derived()
        return evidence

    def classify(self, evidence: WolframEvidence) -> Tuple[int, Dict[str, float]]:
        """Return (wolfram_class, confidence_scores).

        Scores range from 0.0 to 1.0 for each class.
        """
        e = evidence
        scores = {'class_1': 0.0, 'class_2': 0.0, 'class_3': 0.0, 'class_4': 0.0}

        e0 = float(e.label_entropy[0]) if e.generations > 0 else 0.0
        e_ratio = e.entropy_final / max(e0, 1e-10)

        # Class 1 indicators: entropy collapses, order → 1
        if e.entropy_decay_rate > 0.8:
            scores['class_1'] += 0.4
        if e_ratio < 0.1:
            scores['class_1'] += 0.3
        if e.order_convergence > 0.9:
            scores['class_1'] += 0.3

        # Class 2 indicators: entropy stable, low tail variance, clusters stable
        if e.entropy_variance_tail < 0.01:
            scores['class_2'] += 0.3
        if e.cluster_stability > 0.9:
            scores['class_2'] += 0.4
        if 0.1 < e_ratio < 0.9:
            scores['class_2'] += 0.3

        # Class 3 indicators: high entropy, high variance, unstable clusters
        if e.entropy_variance_tail > 0.05:
            scores['class_3'] += 0.3
        if e.cluster_stability < 0.5:
            scores['class_3'] += 0.3
        if e_ratio > 0.7:
            scores['class_3'] += 0.4

        # Class 4 indicators: intermediate entropy, structured boundaries
        if 0.3 < e.cluster_stability < 0.85:
            scores['class_4'] += 0.3
        if 0.02 < e.entropy_variance_tail < 0.05:
            scores['class_4'] += 0.3
        if 0.2 < e.boundary_activity < 0.7:
            scores['class_4'] += 0.4

        best = max(scores, key=scores.get)
        wolfram_class = int(best[-1])
        return wolfram_class, scores

    def classify_lattice(self, lattice: Lattice, rule: Rule) -> Tuple[int, WolframEvidence]:
        """Convenience: evolve + classify in one call."""
        evidence = self.evolve_and_measure(lattice, rule)
        wolfram_class, _scores = self.classify(evidence)
        return wolfram_class, evidence
