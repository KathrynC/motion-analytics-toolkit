"""Detect emergent structures in the lattice: clusters, boundaries, phase transitions."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
import networkx as nx

from .lattice import Lattice


class ClusterDetector:
    """Find clusters (families) of similar motions in the lattice."""
    
    def __init__(self, lattice: Lattice):
        self.lattice = lattice
    
    def find_clusters_hierarchical(self, threshold: float = 0.5, criterion: str = 'distance') -> List[List[int]]:
        """Hierarchical clustering based on feature vectors."""
        # Convert similarity to distance: d = 1 - sim
        dist_matrix = 1 - self.lattice.similarity_matrix
        # pdist expects condensed distance matrix
        condensed = pdist(self.lattice.feature_matrix, metric='euclidean')  # or use dist_matrix upper triangular
        Z = linkage(condensed, method='average')
        labels = fcluster(Z, t=threshold, criterion=criterion)
        clusters = {}
        for idx, lab in enumerate(labels):
            clusters.setdefault(lab, []).append(idx)
        return list(clusters.values())
    
    def find_clusters_dbscan(self, eps: float = 0.5, min_samples: int = 2) -> List[List[int]]:
        """DBSCAN clustering on feature vectors."""
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(1 - self.lattice.similarity_matrix)
        clusters = {}
        for idx, lab in enumerate(labels):
            if lab != -1:  # ignore noise
                clusters.setdefault(lab, []).append(idx)
        return list(clusters.values())
    
    def find_connected_components(self) -> List[List[int]]:
        """Find connected components in the adjacency graph."""
        G = nx.from_numpy_array(self.lattice.adjacency)
        components = [list(comp) for comp in nx.connected_components(G)]
        return components


class BoundaryDetector:
    """Identify nodes that lie on boundaries between clusters."""
    
    def __init__(self, lattice: Lattice):
        self.lattice = lattice
    
    def find_boundary_nodes(self, clusters: List[List[int]]) -> List[int]:
        """
        Nodes that have neighbors in a different cluster.
        """
        # Create a mapping node -> cluster_id
        node_to_cluster = {}
        for cid, nodes in enumerate(clusters):
            for n in nodes:
                node_to_cluster[n] = cid
        
        boundaries = []
        for node in range(len(self.lattice.entries)):
            my_cluster = node_to_cluster.get(node)
            if my_cluster is None:
                continue
            for nb in self.lattice.neighbors(node):
                if node_to_cluster.get(nb, -1) != my_cluster:
                    boundaries.append(node)
                    break
        return boundaries


class PhaseTransitionDetector:
    """
    Detect potential phase transitions: regions where small changes in feature space
    lead to large changes in labels or cluster membership.
    """
    
    def __init__(self, lattice: Lattice):
        self.lattice = lattice
    
    def find_transition_zones(self, cluster_labels: List[int], window: float = 0.1) -> List[int]:
        """
        Identify nodes that are close to nodes of a different cluster in feature space,
        even if not directly connected in the graph.
        """
        n = len(self.lattice.entries)
        # Compute pairwise distances
        dist = 1 - self.lattice.similarity_matrix
        transition_nodes = []
        for i in range(n):
            # Find nodes with different cluster label within distance window
            for j in range(n):
                if i != j and cluster_labels[j] != cluster_labels[i] and dist[i, j] < window:
                    transition_nodes.append(i)
                    break
        return transition_nodes
    
    def compute_order_parameter(self, cluster_labels: List[int]) -> float:
        """
        Compute a global measure of order, e.g., fraction of nodes with same label as majority.
        This can track phase transitions over time.
        """
        if not cluster_labels:
            return 0.0
        from collections import Counter
        counts = Counter(cluster_labels)
        majority = max(counts.values())
        return majority / len(cluster_labels)
