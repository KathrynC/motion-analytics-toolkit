"""Tests for weight-space scenario suites."""

import numpy as np
import pytest

from motion_analytics.scenarios.base import (
    Scenario, ScenarioSuite, SimulatorInterface, StressTest,
)
from motion_analytics.scenarios.weight_suites import (
    SYNAPSE_NAMES_6, SYNAPSE_NAMES_10,
    _weight_vector_to_baseline,
    weight_perturbation_suite,
    synapse_ablation_suite,
    archetype_boundary_suite,
    archetype_interpolation_suite,
    cross_topology_suite,
    comprehensive_weight_suite,
    suite_from_archetype,
    interpolation_from_library,
)
from motion_analytics.archetypes.base import ArchetypeLibrary
from motion_analytics.archetypes.persona import PersonaArchetype


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_6 = [0.5, -0.3, 0.8, 0.0, -0.6, 0.2]
SAMPLE_10 = [0.5, -0.3, 0.8, 0.0, -0.6, 0.2, 0.1, -0.1, 0.15, -0.05]


# ---------------------------------------------------------------------------
# _weight_vector_to_baseline
# ---------------------------------------------------------------------------

class TestWeightVectorToBaseline:
    def test_6_element(self):
        bl = _weight_vector_to_baseline(SAMPLE_6)
        assert len(bl) == 6
        assert set(bl.keys()) == set(SYNAPSE_NAMES_6)
        assert bl['w03'] == pytest.approx(0.5)

    def test_10_element(self):
        bl = _weight_vector_to_baseline(SAMPLE_10)
        assert len(bl) == 10
        assert set(bl.keys()) == set(SYNAPSE_NAMES_10)
        assert bl['w44'] == pytest.approx(-0.05)

    def test_crosswired_flag_pads(self):
        bl = _weight_vector_to_baseline(SAMPLE_6, include_crosswired=True)
        assert len(bl) == 10
        assert bl['w33'] == 0.0
        assert bl['w44'] == 0.0
        assert bl['w03'] == pytest.approx(0.5)

    def test_numpy_input(self):
        arr = np.array(SAMPLE_6)
        bl = _weight_vector_to_baseline(arr)
        assert len(bl) == 6
        assert bl['w24'] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Suite 1 — Weight Perturbation
# ---------------------------------------------------------------------------

class TestWeightPerturbationSuite:
    def test_default_count(self):
        suite = weight_perturbation_suite(SAMPLE_6)
        # 6 synapses x 3 epsilons x 2 directions = 36
        assert len(suite) == 36

    def test_custom_epsilons(self):
        suite = weight_perturbation_suite(SAMPLE_6, epsilons=(0.01, 0.5))
        # 6 x 2 x 2 = 24
        assert len(suite) == 24

    def test_synapse_subset(self):
        suite = weight_perturbation_suite(SAMPLE_6, synapse_subset=['w03', 'w14'])
        # 2 x 3 x 2 = 12
        assert len(suite) == 12

    def test_shift_mode(self):
        suite = weight_perturbation_suite(SAMPLE_6)
        for scen in suite.scenarios:
            assert all(m['mode'] == 'shift' for m in scen.modifications)

    def test_param_values(self):
        suite = weight_perturbation_suite(SAMPLE_6, epsilons=(0.1,))
        params = suite.get_param_sets()
        # First scenario should be w03+0.1: baseline w03=0.5, shift +0.1 => 0.6
        assert params[0]['w03'] == pytest.approx(0.6)
        # Second scenario should be w03-0.1: baseline w03=0.5, shift -0.1 => 0.4
        assert params[1]['w03'] == pytest.approx(0.4)

    def test_naming(self):
        suite = weight_perturbation_suite(SAMPLE_6, epsilons=(0.05,))
        names = [s.name for s in suite.scenarios]
        assert 'w03+0.05' in names
        assert 'w03-0.05' in names


# ---------------------------------------------------------------------------
# Suite 2 — Synapse Ablation
# ---------------------------------------------------------------------------

class TestSynapseAblationSuite:
    def test_count_with_pairs(self):
        suite = synapse_ablation_suite(SAMPLE_6)
        # 6 singles + C(6,2)=15 pairs = 21
        assert len(suite) == 21

    def test_singles_only(self):
        suite = synapse_ablation_suite(SAMPLE_6, include_pairs=False)
        assert len(suite) == 6

    def test_set_mode(self):
        suite = synapse_ablation_suite(SAMPLE_6)
        for scen in suite.scenarios:
            assert all(m['mode'] == 'set' for m in scen.modifications)

    def test_actual_zeroing(self):
        suite = synapse_ablation_suite(SAMPLE_6)
        params = suite.get_param_sets()
        # First scenario: ablate_w03 → w03 should be 0.0
        assert params[0]['w03'] == 0.0
        # Other weights should remain at baseline
        assert params[0]['w04'] == pytest.approx(-0.3)

    def test_pair_zeroing(self):
        suite = synapse_ablation_suite(SAMPLE_6)
        # Find a pair scenario
        pair_scenarios = [s for s in suite.scenarios if '+' in s.name]
        assert len(pair_scenarios) == 15
        # Check that the pair scenario has 2 modifications
        assert len(pair_scenarios[0].modifications) == 2


# ---------------------------------------------------------------------------
# Suite 3 — Archetype Boundary
# ---------------------------------------------------------------------------

class TestArchetypeBoundarySuite:
    def test_nonzero_weights(self):
        # 5 nonzero weights x 3 directions x 4 fractions = 60
        # 1 zero weight (w14=0.0): only saturate x 4 = 4
        # Total = 64
        suite = archetype_boundary_suite(SAMPLE_6)
        assert len(suite) == 64

    def test_all_zero_fewer_scenarios(self):
        # All zeros: only saturate direction for each (toward +1.0)
        suite = archetype_boundary_suite([0.0] * 6)
        # 6 weights x 1 direction (saturate) x 4 fractions = 24
        assert len(suite) == 24

    def test_set_mode(self):
        suite = archetype_boundary_suite(SAMPLE_6)
        for scen in suite.scenarios:
            assert all(m['mode'] == 'set' for m in scen.modifications)

    def test_naming_with_archetype(self):
        suite = archetype_boundary_suite(SAMPLE_6, archetype_name="crab")
        assert suite.scenarios[0].name.startswith("crab_")
        assert "toward_zero" in suite.scenarios[0].name or \
               "sign_flip" in suite.scenarios[0].name or \
               "saturate" in suite.scenarios[0].name

    def test_naming_without_archetype(self):
        suite = archetype_boundary_suite(SAMPLE_6)
        # Should not start with underscore
        assert not suite.scenarios[0].name.startswith("_")


# ---------------------------------------------------------------------------
# Suite 4 — Archetype Interpolation
# ---------------------------------------------------------------------------

class TestArchetypeInterpolationSuite:
    def test_default_count(self):
        suite = archetype_interpolation_suite(SAMPLE_6, [0.0] * 6)
        assert len(suite) == 11

    def test_endpoints_correct(self):
        a = [1.0] * 6
        b = [0.0] * 6
        suite = archetype_interpolation_suite(a, b, steps=11)
        params = suite.get_param_sets()
        # t=0 → all weights = a
        for syn in SYNAPSE_NAMES_6:
            assert params[0][syn] == pytest.approx(1.0)
        # t=1 → all weights = b
        for syn in SYNAPSE_NAMES_6:
            assert params[-1][syn] == pytest.approx(0.0)

    def test_midpoint(self):
        a = [1.0] * 6
        b = [0.0] * 6
        suite = archetype_interpolation_suite(a, b, steps=11)
        params = suite.get_param_sets()
        # step 5 = t=0.5
        for syn in SYNAPSE_NAMES_6:
            assert params[5][syn] == pytest.approx(0.5)

    def test_mixed_dimensions(self):
        a6 = [0.5] * 6
        b10 = [0.3] * 10
        suite = archetype_interpolation_suite(a6, b10)
        params = suite.get_param_sets()
        # Should have 10 keys
        assert len(params[0]) == 10
        # t=0: first 6 = 0.5, last 4 = 0.0 (zero-padded a)
        assert params[0]['w03'] == pytest.approx(0.5)
        assert params[0]['w33'] == pytest.approx(0.0)

    def test_naming(self):
        suite = archetype_interpolation_suite(
            SAMPLE_6, [0.0] * 6, label_a="fold", label_b="mirror"
        )
        assert "fold" in suite.scenarios[0].name
        assert "mirror" in suite.scenarios[0].name


# ---------------------------------------------------------------------------
# Suite 5 — Cross-Topology
# ---------------------------------------------------------------------------

class TestCrossTopologySuite:
    def test_default_count(self):
        suite = cross_topology_suite(SAMPLE_6)
        # 5 patterns x 4 strengths x 2 signs = 40
        assert len(suite) == 40

    def test_10_element_baseline(self):
        suite = cross_topology_suite(SAMPLE_6)
        assert len(suite.baseline) == 10

    def test_core_weights_preserved(self):
        suite = cross_topology_suite(SAMPLE_6)
        # Baseline should have original core weights
        assert suite.baseline['w03'] == pytest.approx(0.5)
        assert suite.baseline['w24'] == pytest.approx(0.2)

    def test_crosswired_modified(self):
        suite = cross_topology_suite(SAMPLE_6)
        # Find a self_back scenario — should modify w33
        self_back = [s for s in suite.scenarios if s.name.startswith('self_back')]
        assert len(self_back) == 8  # 4 strengths x 2 signs
        targets = {m['target'] for s in self_back for m in s.modifications}
        assert targets == {'w33'}

    def test_custom_strengths(self):
        suite = cross_topology_suite(SAMPLE_6, crosswire_strengths=(0.2, 0.9))
        # 5 patterns x 2 strengths x 2 signs = 20
        assert len(suite) == 20


# ---------------------------------------------------------------------------
# Comprehensive Composite
# ---------------------------------------------------------------------------

class TestComprehensiveWeightSuite:
    def test_combines_all(self):
        suite = comprehensive_weight_suite(SAMPLE_6)
        # 36 + 21 + 64 + 40 = 161
        expected = (
            len(weight_perturbation_suite(SAMPLE_6))
            + len(synapse_ablation_suite(SAMPLE_6))
            + len(archetype_boundary_suite(SAMPLE_6))
            + len(cross_topology_suite(SAMPLE_6))
        )
        assert len(suite) == expected

    def test_10_element_baseline(self):
        suite = comprehensive_weight_suite(SAMPLE_6)
        assert len(suite.baseline) == 10


# ---------------------------------------------------------------------------
# Integration with StressTest
# ---------------------------------------------------------------------------

class WeightSimulator(SimulatorInterface):
    """Dummy simulator that sums weights as a single output metric."""

    def run(self, params):
        total = sum(v for k, v in params.items() if k.startswith('w'))
        return {'total_weight': total, 'abs_weight': abs(total)}

    def get_baseline_params(self):
        return _weight_vector_to_baseline(SAMPLE_6)


class TestWeightSuiteWithStressTest:
    def test_integration(self):
        sim = WeightSimulator()
        suite = weight_perturbation_suite(SAMPLE_6, epsilons=(0.1,))
        st = StressTest(sim)
        summary = st.run_suite(suite, motion_id='test_gait')
        assert summary.motion_id == 'test_gait'
        assert len(summary.results) == 12  # 6 x 1 x 2
        assert all(r.success for r in summary.results)
        assert 'total_weight' in summary.baseline_outputs


# ---------------------------------------------------------------------------
# Archetype-to-suite bridge
# ---------------------------------------------------------------------------

def _make_library():
    """Build a small test library with two archetypes."""
    lib = ArchetypeLibrary()
    lib.add(PersonaArchetype('crab', weight_vector=np.array(SAMPLE_6)))
    lib.add(PersonaArchetype('spinner', weight_vector=np.array(SAMPLE_10)))
    lib.add(PersonaArchetype('empty', weight_vector=None))
    return lib


class TestSuiteFromArchetype:
    def test_comprehensive(self):
        lib = _make_library()
        suite = suite_from_archetype(lib, 'crab')
        expected = len(comprehensive_weight_suite(SAMPLE_6))
        assert len(suite) == expected

    def test_perturbation(self):
        lib = _make_library()
        suite = suite_from_archetype(lib, 'crab', suite_type='perturbation')
        assert len(suite) == 36

    def test_boundary_gets_archetype_name(self):
        lib = _make_library()
        suite = suite_from_archetype(lib, 'crab', suite_type='boundary')
        assert suite.scenarios[0].name.startswith('crab_')

    def test_unknown_archetype_raises(self):
        lib = _make_library()
        with pytest.raises(KeyError, match='nonexistent'):
            suite_from_archetype(lib, 'nonexistent')

    def test_no_weight_vector_raises(self):
        lib = _make_library()
        with pytest.raises(ValueError, match='no weight_vector'):
            suite_from_archetype(lib, 'empty')

    def test_unknown_suite_type_raises(self):
        lib = _make_library()
        with pytest.raises(ValueError, match='Unknown suite_type'):
            suite_from_archetype(lib, 'crab', suite_type='bogus')

    def test_kwargs_forwarded(self):
        lib = _make_library()
        suite = suite_from_archetype(lib, 'crab', suite_type='perturbation',
                                     epsilons=(0.01,))
        assert len(suite) == 12  # 6 x 1 x 2


class TestInterpolationFromLibrary:
    def test_default_steps(self):
        lib = _make_library()
        suite = interpolation_from_library(lib, 'crab', 'spinner')
        assert len(suite) == 11

    def test_naming_uses_archetype_names(self):
        lib = _make_library()
        suite = interpolation_from_library(lib, 'crab', 'spinner')
        assert 'crab' in suite.scenarios[0].name
        assert 'spinner' in suite.scenarios[0].name

    def test_custom_steps(self):
        lib = _make_library()
        suite = interpolation_from_library(lib, 'crab', 'spinner', steps=5)
        assert len(suite) == 5

    def test_missing_archetype_raises(self):
        lib = _make_library()
        with pytest.raises(KeyError):
            interpolation_from_library(lib, 'crab', 'ghost')
