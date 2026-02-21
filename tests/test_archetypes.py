"""Tests for archetypes module: base, persona, structural_transfer, templates."""

import json
import tempfile
import numpy as np
import pytest
from pathlib import Path

from motion_analytics.archetypes.base import Archetype, ArchetypeLibrary
from motion_analytics.archetypes.persona import PersonaArchetype, load_persona_library
from motion_analytics.archetypes.structural_transfer import StructuralTransferAnalyzer
from motion_analytics.archetypes.templates import BUILTIN_ARCHETYPES
from motion_analytics.core.schemas import (
    Telemetry, TimeStep, LinkState, JointState, ContactState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(t, x=0.0):
    return TimeStep(
        timestamp=t,
        links={
            'torso': LinkState([x, 0, 0.1], [0, 0, 0, 1], [0.01, 0, 0], [0, 0, 0]),
            'back_leg': LinkState([x - 0.5, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]),
            'front_leg': LinkState([x + 0.5, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0]),
        },
        joints={
            'back_leg_joint': JointState(angle=0.3 * np.sin(2 * np.pi * 2 * t), velocity=0.5, torque=0.1),
            'front_leg_joint': JointState(angle=-0.3 * np.sin(2 * np.pi * 2 * t), velocity=0.5, torque=0.1),
        },
        contacts=[
            ContactState('back_leg', 1.0, [0, 0], [0, 0, 0], np.sin(2 * np.pi * 2 * t) > 0),
            ContactState('front_leg', 1.0, [0, 0], [0, 0, 0], np.sin(2 * np.pi * 2 * t) <= 0),
        ],
        com_position=[x, 0.0, 0.1],
        com_velocity=[0.01, 0.0, 0.0],
    )


def _make_telemetry(n=200, sr=240.0, weights=None):
    steps = [_make_ts(i / sr, x=i * 0.001) for i in range(n)]
    meta = {'source': 'test'}
    if weights is not None:
        meta['synapse_weights'] = weights
    return Telemetry(metadata=meta, timesteps=steps, sampling_rate=sr)


# ---------------------------------------------------------------------------
# Archetype base tests
# ---------------------------------------------------------------------------

class TestArchetypeBase:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Archetype('test')

    def test_archetype_library_add(self):
        lib = ArchetypeLibrary()
        pa = PersonaArchetype('test', weight_vector=np.zeros(6))
        lib.add(pa)
        assert lib.get('test') is pa
        assert lib.get('nonexistent') is None

    def test_archetype_library_save_load(self):
        lib = ArchetypeLibrary([
            PersonaArchetype('a', weight_vector=np.array([1.0, 2.0])),
        ])
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)
        lib.save(path)
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]['name'] == 'a'


# ---------------------------------------------------------------------------
# PersonaArchetype tests
# ---------------------------------------------------------------------------

class TestPersonaArchetype:
    def test_weight_space_similarity(self):
        pa = PersonaArchetype('test', weight_vector=np.array([0.5, -0.5, 0.3, -0.3, 0.1, -0.1]))
        tel = _make_telemetry(weights=[0.5, -0.5, 0.3, -0.3, 0.1, -0.1])
        score = pa.similarity_to(tel)
        # Identical weights should give perfect similarity
        assert abs(score - 1.0) < 1e-6

    def test_weight_space_dissimilar(self):
        pa = PersonaArchetype('test', weight_vector=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        tel = _make_telemetry(weights=[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        score = pa.similarity_to(tel)
        # Opposite weights — similarity should be 0
        assert abs(score - 0.0) < 1e-6

    def test_no_weights_no_features_returns_zero(self):
        pa = PersonaArchetype('test')  # no weight, no feature vector
        tel = _make_telemetry()  # no weights in metadata
        score = pa.similarity_to(tel)
        assert score == 0.0

    def test_from_dict_roundtrip(self):
        pa = PersonaArchetype('fold', weight_vector=np.array([0.7, -0.7]),
                              feature_vector=np.array([1.0, 2.0, 3.0]),
                              description='test desc')
        d = pa.to_dict()
        pa2 = PersonaArchetype.from_dict(d)
        assert pa2.name == 'fold'
        np.testing.assert_allclose(pa2.weight_vector, [0.7, -0.7])
        np.testing.assert_allclose(pa2.feature_vector, [1.0, 2.0, 3.0])

    def test_to_dict_keys(self):
        pa = PersonaArchetype('x', weight_vector=np.ones(6))
        d = pa.to_dict()
        assert 'name' in d
        assert 'type' in d
        assert 'weight_vector' in d


# ---------------------------------------------------------------------------
# Load persona library
# ---------------------------------------------------------------------------

class TestLoadPersonaLibrary:
    def test_load_from_json(self):
        data = [
            {'name': 'a', 'weight_vector': [1, 2, 3], 'description': 'test'},
            {'name': 'b', 'weight_vector': [4, 5, 6]},
        ]
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        lib = load_persona_library(path)
        assert len(lib.archetypes) == 2
        assert lib.get('a').name == 'a'


# ---------------------------------------------------------------------------
# Builtin archetypes
# ---------------------------------------------------------------------------

class TestBuiltinArchetypes:
    def test_library_populated(self):
        assert len(BUILTIN_ARCHETYPES.archetypes) >= 4

    def test_deleuze_fold_exists(self):
        arch = BUILTIN_ARCHETYPES.get('deleuze_fold')
        assert arch is not None
        assert arch.weight_vector is not None

    def test_similarity_vector(self):
        tel = _make_telemetry(weights=[0.7, -0.7, 0.0, 0.0, 0.0, 0.0, 0.7, -0.7, 0.0, 0.0])
        sv = BUILTIN_ARCHETYPES.similarity_vector(tel)
        assert 'deleuze_fold' in sv
        assert all(0 <= v <= 1 for v in sv.values())

    def test_best_match(self):
        tel = _make_telemetry(weights=[0.7, -0.7, 0.0, 0.0, 0.0, 0.0, 0.7, -0.7, 0.0, 0.0])
        arch, score = BUILTIN_ARCHETYPES.best_match(tel)
        assert arch is not None
        assert 0 <= score <= 1


# ---------------------------------------------------------------------------
# StructuralTransferAnalyzer tests
# ---------------------------------------------------------------------------

class TestStructuralTransfer:
    def test_analyze_returns_keys(self):
        tel = _make_telemetry(weights=[0.5, -0.5, 0.3, -0.3, 0.1, -0.1, 0, 0, 0, 0])
        analyzer = StructuralTransferAnalyzer(BUILTIN_ARCHETYPES)
        result = analyzer.analyze(tel)
        assert 'best_match' in result
        assert 'similarities' in result
        assert 'all_scores' in result

    def test_weight_space_match(self):
        tel = _make_telemetry(weights=[0.7, -0.7, 0.0, 0.0, 0.0, 0.0, 0.7, -0.7, 0.0, 0.0])
        analyzer = StructuralTransferAnalyzer(BUILTIN_ARCHETYPES)
        result = analyzer.analyze(tel)
        assert result['weight_space_match'] is not None
        # Deleuze fold should be close
        assert result['weight_space_match']['name'] == 'deleuze_fold'
