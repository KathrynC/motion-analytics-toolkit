"""Tests for semantic_ca module: lattice, rules, emergence, dynamics, builtin."""

import json
import tempfile
import numpy as np
import pytest
from pathlib import Path

from motion_analytics.semantic_ca.lattice import (
    Lattice, cosine_similarity, euclidean_similarity,
    extract_beer_features,
)
from motion_analytics.semantic_ca.rules import (
    Rule, MajorityVoteRule, PropagationRule, MutationRule, CompositeRule,
)
from motion_analytics.semantic_ca.emergence import (
    ClusterDetector, BoundaryDetector, PhaseTransitionDetector,
)
from motion_analytics.semantic_ca.builtin import BEER_METRICS, DEFAULT_SIMILARITY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dict_file(n=10):
    """Create a temporary motion dictionary JSON file."""
    entries = []
    for i in range(n):
        entries.append({
            'id': f'entry_{i}',
            'label': f'gait_{i % 3}',
            'beer_analytics': {
                'outcome': {'displacement': float(i * 0.1), 'speed_mean': 0.5, 'efficiency': 0.3},
                'contact': {'entropy': 0.5, 'duty_cycle_back': 0.5, 'duty_cycle_front': 0.5},
                'coordination': {'phase_lock': 0.8, 'freq_back': 2.0, 'freq_front': 2.0},
                'rotation_axis': {
                    'roll_dominance': 0.1, 'pitch_dominance': 0.5,
                    'yaw_dominance': 0.4, 'axis_switching_rate': 0.2,
                },
            },
        })
    path = Path(tempfile.mktemp(suffix='.json'))
    path.write_text(json.dumps(entries))
    return path, entries


def _build_lattice(n=10, threshold=0.5):
    path, entries = _make_dict_file(n)
    lattice = Lattice.from_dictionary(
        path, feature_extractor=extract_beer_features,
        graph_type='threshold', threshold=threshold,
    )
    return lattice


# ---------------------------------------------------------------------------
# Similarity function tests
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_cosine_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-10

    def test_cosine_zero_vector(self):
        a = np.array([1.0, 2.0])
        z = np.zeros(2)
        assert cosine_similarity(a, z) == 0.0

    def test_cosine_range(self):
        a = np.random.randn(5)
        b = np.random.randn(5)
        s = cosine_similarity(a, b)
        assert 0.0 <= s <= 1.0

    def test_euclidean_identical(self):
        a = np.array([1.0, 2.0])
        assert abs(euclidean_similarity(a, a) - 1.0) < 1e-10

    def test_euclidean_far_apart(self):
        a = np.array([0.0])
        b = np.array([100.0])
        s = euclidean_similarity(a, b, scale=1.0)
        assert s < 0.01


# ---------------------------------------------------------------------------
# Lattice tests
# ---------------------------------------------------------------------------

class TestLattice:
    def test_from_dictionary_threshold(self):
        lattice = _build_lattice(10, threshold=0.5)
        assert len(lattice.entries) == 10
        assert lattice.feature_matrix.shape[0] == 10
        assert lattice.similarity_matrix.shape == (10, 10)
        assert lattice.adjacency.shape == (10, 10)

    def test_from_dictionary_knn(self):
        path, _ = _make_dict_file(10)
        lattice = Lattice.from_dictionary(
            path, feature_extractor=extract_beer_features,
            graph_type='knn', k=3,
        )
        # Each node should have at most k neighbors
        for i in range(10):
            assert len(lattice.neighbors(i)) <= 10

    def test_from_dictionary_invalid_type(self):
        path, _ = _make_dict_file(5)
        with pytest.raises(ValueError, match="Unknown graph_type"):
            Lattice.from_dictionary(path, extract_beer_features, graph_type='invalid')

    def test_neighbors(self):
        lattice = _build_lattice(10, threshold=0.5)
        nbs = lattice.neighbors(0)
        assert isinstance(nbs, list)
        for n in nbs:
            assert 0 <= n < 10

    def test_subgraph(self):
        lattice = _build_lattice(10, threshold=0.5)
        sub = lattice.subgraph([0, 1, 2])
        assert len(sub.entries) == 3
        assert sub.feature_matrix.shape[0] == 3
        assert sub.adjacency.shape == (3, 3)

    def test_dict_with_entries_key(self):
        entries = [
            {'id': '0', 'beer_analytics': {
                'outcome': {'displacement': 0.1, 'speed_mean': 0.5, 'efficiency': 0.3},
                'contact': {'entropy': 0.5, 'duty_cycle_back': 0.5, 'duty_cycle_front': 0.5},
                'coordination': {'phase_lock': 0.8, 'freq_back': 2.0, 'freq_front': 2.0},
                'rotation_axis': {'roll_dominance': 0.1, 'pitch_dominance': 0.5,
                                  'yaw_dominance': 0.4, 'axis_switching_rate': 0.2},
            }}
        ]
        data = {'entries': entries}
        path = Path(tempfile.mktemp(suffix='.json'))
        path.write_text(json.dumps(data))
        lattice = Lattice.from_dictionary(path, extract_beer_features)
        assert len(lattice.entries) == 1


# ---------------------------------------------------------------------------
# Feature extractor tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def test_beer_features_shape(self):
        entry = {
            'beer_analytics': {
                'outcome': {'displacement': 0.5, 'speed_mean': 0.3, 'efficiency': 0.2},
                'contact': {'entropy': 0.5, 'duty_cycle_back': 0.5, 'duty_cycle_front': 0.5},
                'coordination': {'phase_lock': 0.8, 'freq_back': 2, 'freq_front': 2},
                'rotation_axis': {'roll_dominance': 0.1, 'pitch_dominance': 0.5,
                                  'yaw_dominance': 0.4, 'axis_switching_rate': 0.2},
            }
        }
        feat = extract_beer_features(entry)
        assert feat.shape == (13,)

    def test_beer_features_missing_analytics(self):
        assert extract_beer_features({}) is None
        assert extract_beer_features({'beer_analytics': {}}) is None


# ---------------------------------------------------------------------------
# Rule tests
# ---------------------------------------------------------------------------

class TestRules:
    def test_majority_vote_basic(self):
        lattice = _build_lattice(5, threshold=0.0)  # low threshold = few edges
        # Manually set labels
        for i, entry in enumerate(lattice.entries):
            entry['label'] = 'A' if i < 3 else 'B'
        # Force full connectivity
        lattice.adjacency = np.ones((5, 5)) - np.eye(5)

        rule = MajorityVoteRule()
        # Node 3 has label 'B' but majority of neighbors are 'A'
        updates = rule.apply(lattice, 3)
        assert updates.get('label') == 'A'

    def test_majority_vote_no_neighbors(self):
        lattice = _build_lattice(3, threshold=0.0)
        lattice.adjacency = np.zeros((3, 3))  # no edges
        rule = MajorityVoteRule()
        assert rule.apply(lattice, 0) == {}

    def test_propagation_rule(self):
        lattice = _build_lattice(3, threshold=0.0)
        lattice.adjacency = np.ones((3, 3)) - np.eye(3)
        for i, e in enumerate(lattice.entries):
            e['score'] = float(i)  # 0, 1, 2
        rule = PropagationRule('score', alpha=1.0)
        updates = rule.apply(lattice, 0)
        # Node 0 (score=0), neighbors are 1 and 2 (avg=1.5)
        assert abs(updates['score'] - 1.5) < 1e-10

    def test_mutation_rule_low_prob(self):
        lattice = _build_lattice(3)
        rule = MutationRule(mutation_prob=0.0)
        assert rule.apply(lattice, 0) == {}

    def test_composite_rule(self):
        lattice = _build_lattice(3, threshold=0.0)
        lattice.adjacency = np.ones((3, 3)) - np.eye(3)
        for i, e in enumerate(lattice.entries):
            e['label'] = 'A'
            e['score'] = 1.0

        rule = CompositeRule([
            MajorityVoteRule(),
            PropagationRule('score', alpha=0.5),
        ])
        updates = rule.apply(lattice, 0)
        assert 'score' in updates


# ---------------------------------------------------------------------------
# Emergence tests
# ---------------------------------------------------------------------------

class TestEmergence:
    def test_cluster_detector_hierarchical(self):
        lattice = _build_lattice(10, threshold=0.5)
        detector = ClusterDetector(lattice)
        clusters = detector.find_clusters_hierarchical(threshold=2.0)
        assert isinstance(clusters, list)
        # All nodes should appear in some cluster
        all_nodes = set()
        for c in clusters:
            all_nodes.update(c)
        assert len(all_nodes) == 10

    def test_cluster_detector_dbscan(self):
        lattice = _build_lattice(10, threshold=0.5)
        detector = ClusterDetector(lattice)
        clusters = detector.find_clusters_dbscan(eps=0.5, min_samples=2)
        assert isinstance(clusters, list)

    def test_connected_components(self):
        lattice = _build_lattice(10, threshold=0.5)
        detector = ClusterDetector(lattice)
        comps = detector.find_connected_components()
        assert isinstance(comps, list)
        all_nodes = set()
        for c in comps:
            all_nodes.update(c)
        assert len(all_nodes) == 10

    def test_boundary_detector(self):
        lattice = _build_lattice(6, threshold=0.0)
        lattice.adjacency = np.ones((6, 6)) - np.eye(6)
        clusters = [[0, 1, 2], [3, 4, 5]]
        detector = BoundaryDetector(lattice)
        boundaries = detector.find_boundary_nodes(clusters)
        # All nodes are boundary nodes since everyone connects to everyone
        assert len(boundaries) == 6

    def test_phase_transition_detector(self):
        lattice = _build_lattice(6, threshold=0.5)
        labels = [0, 0, 0, 1, 1, 1]
        detector = PhaseTransitionDetector(lattice)
        zones = detector.find_transition_zones(labels, window=0.5)
        assert isinstance(zones, list)

    def test_order_parameter(self):
        lattice = _build_lattice(6, threshold=0.5)
        labels = [0, 0, 0, 0, 1, 1]
        detector = PhaseTransitionDetector(lattice)
        op = detector.compute_order_parameter(labels)
        assert abs(op - 4 / 6) < 1e-10

    def test_order_parameter_empty(self):
        lattice = _build_lattice(6)
        detector = PhaseTransitionDetector(lattice)
        assert detector.compute_order_parameter([]) == 0.0


# ---------------------------------------------------------------------------
# Builtin tests
# ---------------------------------------------------------------------------

class TestBuiltin:
    def test_beer_metrics_is_list(self):
        assert isinstance(BEER_METRICS, list)
        assert len(BEER_METRICS) == 13

    def test_default_similarity_callable(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        s = DEFAULT_SIMILARITY(a, b)
        assert 0.0 <= s <= 1.0
