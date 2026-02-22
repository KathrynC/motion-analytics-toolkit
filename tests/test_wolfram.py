"""Tests for Wolfram CA classifier (Classes 1-4)."""

import numpy as np
import pytest

from motion_analytics.semantic_ca.lattice import Lattice
from motion_analytics.semantic_ca.rules import (
    MajorityVoteRule, PropagationRule, CompositeRule, Rule,
)
from motion_analytics.semantic_ca.wolfram import WolframClassifier, WolframEvidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lattice(n_nodes, labels, adjacency=None):
    """Build a small lattice with given labels and adjacency."""
    entries = [{'label': lbl} for lbl in labels]
    node_ids = [str(i) for i in range(n_nodes)]
    feat = np.random.default_rng(42).random((n_nodes, 3))
    if adjacency is None:
        # Default: ring topology (each node connected to its neighbors)
        adj = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            adj[i, (i + 1) % n_nodes] = 1
            adj[(i + 1) % n_nodes, i] = 1
        adjacency = adj
    sim = np.eye(n_nodes)
    return Lattice(
        entries=entries,
        node_ids=node_ids,
        feature_matrix=feat,
        adjacency=adjacency,
        similarity_matrix=sim,
    )


def _make_homogeneous_lattice(n=10, label='A'):
    """All nodes have the same label — should converge to Class 1."""
    return _make_lattice(n, [label] * n)


def _make_two_cluster_lattice():
    """Two disconnected clusters with different labels — stable Class 2."""
    n = 10
    labels = ['A'] * 5 + ['B'] * 5
    adj = np.zeros((n, n))
    # Cluster 1: nodes 0-4 fully connected
    for i in range(5):
        for j in range(i + 1, 5):
            adj[i, j] = adj[j, i] = 1
    # Cluster 2: nodes 5-9 fully connected
    for i in range(5, 10):
        for j in range(i + 1, 10):
            adj[i, j] = adj[j, i] = 1
    return _make_lattice(n, labels, adj)


class RandomLabelRule(Rule):
    """Rule that assigns random labels — should produce Class 3 (chaotic)."""

    def __init__(self, labels=('A', 'B', 'C'), seed=42):
        self._labels = labels
        self._rng = np.random.default_rng(seed)

    def apply(self, lattice, node_idx):
        new_label = self._labels[self._rng.integers(0, len(self._labels))]
        return {'label': new_label}


class FlipRule(Rule):
    """Rule that deterministically flips between two labels — produces Class 2."""

    def __init__(self, labels=('A', 'B')):
        self._labels = labels

    def apply(self, lattice, node_idx):
        current = lattice.entries[node_idx].get('label', self._labels[0])
        if current == self._labels[0]:
            return {'label': self._labels[1]}
        return {'label': self._labels[0]}


# ---------------------------------------------------------------------------
# WolframEvidence tests
# ---------------------------------------------------------------------------

class TestWolframEvidence:
    def test_dataclass_creation(self):
        e = WolframEvidence(
            label_entropy=np.array([1.0, 0.5, 0.1]),
            cluster_counts=np.array([3, 2, 1]),
            boundary_counts=np.array([4, 2, 0]),
            order_parameter=np.array([0.4, 0.6, 0.9]),
            generations=3,
        )
        assert e.generations == 3
        assert len(e.label_entropy) == 3

    def test_derived_statistics(self):
        e = WolframEvidence(
            label_entropy=np.array([2.0, 1.5, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0]),
            cluster_counts=np.array([5, 4, 3, 2, 1, 1, 1, 1]),
            boundary_counts=np.array([6, 4, 3, 2, 1, 0, 0, 0]),
            order_parameter=np.array([0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.0, 1.0]),
            generations=8,
        )
        e.compute_derived()
        assert e.entropy_final == 0.0
        assert e.entropy_decay_rate == 1.0  # full decay
        assert e.order_convergence == 1.0
        assert e.entropy_variance_tail == 0.0  # all zeros in tail


# ---------------------------------------------------------------------------
# WolframClassifier tests
# ---------------------------------------------------------------------------

class TestWolframClassifier:
    def test_class_1_unanimous_rule(self):
        """MajorityVoteRule on homogeneous lattice → Class 1."""
        lattice = _make_homogeneous_lattice(10, 'A')
        rule = MajorityVoteRule()
        wc = WolframClassifier(generations=20)
        cls, evidence = wc.classify_lattice(lattice, rule)
        assert cls == 1

    def test_class_2_oscillating(self):
        """FlipRule deterministically alternates labels → Class 2."""
        lattice = _make_lattice(8, ['A', 'B'] * 4)
        rule = FlipRule()
        wc = WolframClassifier(generations=50)
        cls, evidence = wc.classify_lattice(lattice, rule)
        assert cls == 2

    def test_class_3_random_rule(self):
        """RandomLabelRule assigns random labels → Class 3."""
        # Need enough nodes that random assignment produces high, variable entropy
        n = 30
        lattice = _make_lattice(n, ['A'] * 10 + ['B'] * 10 + ['C'] * 10)
        rule = RandomLabelRule(labels=('A', 'B', 'C'), seed=99)
        wc = WolframClassifier(generations=100)
        cls, evidence = wc.classify_lattice(lattice, rule)
        assert cls == 3

    def test_evolve_returns_evidence(self):
        lattice = _make_homogeneous_lattice(6)
        rule = MajorityVoteRule()
        wc = WolframClassifier(generations=10)
        evidence = wc.evolve_and_measure(lattice, rule)
        assert evidence.generations == 10
        assert len(evidence.label_entropy) == 10
        assert len(evidence.cluster_counts) == 10
        assert len(evidence.boundary_counts) == 10
        assert len(evidence.order_parameter) == 10

    def test_classify_returns_int_1_through_4(self):
        lattice = _make_homogeneous_lattice(6)
        rule = MajorityVoteRule()
        wc = WolframClassifier(generations=10)
        evidence = wc.evolve_and_measure(lattice, rule)
        cls, scores = wc.classify(evidence)
        assert cls in (1, 2, 3, 4)
        assert set(scores.keys()) == {'class_1', 'class_2', 'class_3', 'class_4'}

    def test_classify_lattice_convenience(self):
        lattice = _make_homogeneous_lattice(6)
        rule = MajorityVoteRule()
        wc = WolframClassifier(generations=10)
        cls, evidence = wc.classify_lattice(lattice, rule)
        assert cls in (1, 2, 3, 4)
        assert isinstance(evidence, WolframEvidence)

    def test_generations_parameter(self):
        lattice = _make_homogeneous_lattice(4)
        rule = MajorityVoteRule()
        wc = WolframClassifier(generations=5)
        evidence = wc.evolve_and_measure(lattice, rule)
        assert evidence.generations == 5
        assert len(evidence.label_entropy) == 5

    def test_does_not_mutate_original_lattice(self):
        lattice = _make_lattice(4, ['A', 'B', 'A', 'B'])
        original_labels = [e['label'] for e in lattice.entries]
        rule = MajorityVoteRule()
        wc = WolframClassifier(generations=20)
        wc.classify_lattice(lattice, rule)
        after_labels = [e['label'] for e in lattice.entries]
        assert original_labels == after_labels


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestWolframIntegration:
    def test_majority_vote_on_homogeneous_lattice(self):
        """All same label → Class 1."""
        lattice = _make_homogeneous_lattice(10, 'X')
        wc = WolframClassifier(generations=30)
        cls, evidence = wc.classify_lattice(lattice, MajorityVoteRule())
        assert cls == 1
        # Entropy should be 0 from start (all same label)
        assert evidence.entropy_final == 0.0

    def test_majority_vote_on_two_cluster_lattice(self):
        """Two disconnected clusters with different labels → Class 2."""
        lattice = _make_two_cluster_lattice()
        wc = WolframClassifier(generations=30)
        cls, evidence = wc.classify_lattice(lattice, MajorityVoteRule())
        assert cls == 2
        # Both clusters are internally unanimous, so entropy stays constant
        assert evidence.cluster_stability > 0.8

    def test_propagation_rule_convergence(self):
        """PropagationRule on numeric property → tracks entropy decline."""
        lattice = _make_lattice(8, ['A'] * 4 + ['B'] * 4)
        # Add a numeric property for propagation
        for i, e in enumerate(lattice.entries):
            e['score'] = float(i) / 8.0
        rule = PropagationRule('score', alpha=0.5)
        wc = WolframClassifier(generations=30)
        evidence = wc.evolve_and_measure(lattice, rule)
        # PropagationRule doesn't change labels, so label entropy should be constant
        assert evidence.label_entropy[0] == evidence.label_entropy[-1]

    def test_composite_rule_classification(self):
        """CompositeRule combining MajorityVote + Propagation."""
        lattice = _make_lattice(8, ['A', 'B'] * 4)
        for i, e in enumerate(lattice.entries):
            e['score'] = 0.5
        rule = CompositeRule([
            MajorityVoteRule(),
            PropagationRule('score', alpha=0.3),
        ])
        wc = WolframClassifier(generations=30)
        cls, evidence = wc.classify_lattice(lattice, rule)
        assert cls in (1, 2, 3, 4)

    def test_empty_lattice_handles_gracefully(self):
        """Empty lattice should not crash."""
        lattice = Lattice(
            entries=[],
            node_ids=[],
            feature_matrix=np.empty((0, 0)),
            adjacency=np.empty((0, 0)),
            similarity_matrix=np.empty((0, 0)),
        )
        wc = WolframClassifier(generations=5)
        evidence = wc.evolve_and_measure(lattice, MajorityVoteRule())
        assert evidence.generations == 5
        cls, scores = wc.classify(evidence)
        assert cls in (1, 2, 3, 4)
