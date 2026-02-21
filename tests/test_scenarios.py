"""Tests for scenarios module: base, io, builtin, robustness, vulnerability."""

import json
import tempfile
import numpy as np
import pytest
from pathlib import Path

from motion_analytics.scenarios.base import (
    Scenario, ScenarioSuite, SimulatorInterface,
    StressResult, StressSummary, StressTest,
)
from motion_analytics.scenarios.io import load_scenarios, save_results, load_cramer_results
from motion_analytics.scenarios.builtin import (
    gravity_suite, friction_suite, force_suite, comprehensive_suite,
)
from motion_analytics.scenarios.robustness import Robustness, Regret
from motion_analytics.scenarios.vulnerability import VulnerabilityProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummySimulator(SimulatorInterface):
    """Simple simulator that returns params scaled by a factor."""

    def run(self, params):
        g = params.get('gravity_scale', 1.0)
        f = params.get('friction_scale', 1.0)
        return {
            'dx': 1.0 * g * f,
            'speed': 0.5 * g,
            'efficiency': 0.8 / (g + 0.01),
        }

    def get_baseline_params(self):
        return {'gravity_scale': 1.0, 'friction_scale': 1.0, 'max_force_scale': 1.0}


def _make_summary():
    baseline_params = {'gravity_scale': 1.0, 'friction_scale': 1.0}
    baseline_outputs = {'dx': 1.0, 'speed': 0.5}
    results = [
        StressResult('gravity_1.3x', {'gravity_scale': 1.3}, {'dx': 0.9, 'speed': 0.45},
                     baseline_outputs, True),
        StressResult('gravity_2.0x', {'gravity_scale': 2.0}, {'dx': 0.5, 'speed': 0.2},
                     baseline_outputs, True),
        StressResult('failed', {}, {}, baseline_outputs, False),
    ]
    return StressSummary('test_motion', baseline_params, baseline_outputs, results)


# ---------------------------------------------------------------------------
# Scenario base tests
# ---------------------------------------------------------------------------

class TestScenarioBase:
    def test_scenario_creation(self):
        s = Scenario('test', [{'mode': 'scale', 'target': 'gravity_scale', 'value': 1.2}])
        assert s.name == 'test'

    def test_scenario_addition(self):
        s1 = Scenario('a', [{'mode': 'scale', 'target': 'gravity_scale', 'value': 1.2}])
        s2 = Scenario('b', [{'mode': 'shift', 'target': 'friction_scale', 'value': 0.1}])
        combined = s1 + s2
        assert 'a+b' == combined.name
        assert len(combined.modifications) == 2

    def test_suite_get_param_sets(self):
        baseline = {'gravity_scale': 1.0, 'friction_scale': 1.0}
        suite = ScenarioSuite('test', [
            Scenario('g1.2', [{'mode': 'scale', 'target': 'gravity_scale', 'value': 1.2}]),
            Scenario('f+0.1', [{'mode': 'shift', 'target': 'friction_scale', 'value': 0.1}]),
            Scenario('set_g', [{'mode': 'set', 'target': 'gravity_scale', 'value': 0.5}]),
        ], baseline)
        params = suite.get_param_sets()
        assert len(params) == 3
        assert abs(params[0]['gravity_scale'] - 1.2) < 1e-10
        assert abs(params[1]['friction_scale'] - 1.1) < 1e-10
        assert abs(params[2]['gravity_scale'] - 0.5) < 1e-10

    def test_suite_len(self):
        suite = ScenarioSuite('x', [Scenario('a', [])], {})
        assert len(suite) == 1

    def test_suite_invalid_mode(self):
        suite = ScenarioSuite('x', [
            Scenario('bad', [{'mode': 'invalid', 'target': 'x', 'value': 1}]),
        ], {'x': 1.0})
        with pytest.raises(ValueError, match="Unknown mode"):
            suite.get_param_sets()


class TestStressTest:
    def test_run_suite(self):
        sim = DummySimulator()
        suite = gravity_suite()
        st = StressTest(sim)
        summary = st.run_suite(suite, motion_id='dummy')
        assert summary.motion_id == 'dummy'
        assert len(summary.results) == len(suite)
        assert all(r.success for r in summary.results)

    def test_summary_output_matrix(self):
        summary = _make_summary()
        mx = summary.get_output_matrix('dx')
        assert len(mx) == 3
        assert mx[0] == 0.9


# ---------------------------------------------------------------------------
# Builtin suites tests
# ---------------------------------------------------------------------------

class TestBuiltinSuites:
    def test_gravity_suite_count(self):
        s = gravity_suite()
        assert len(s) == 7

    def test_friction_suite_count(self):
        s = friction_suite()
        assert len(s) == 7

    def test_force_suite_count(self):
        s = force_suite()
        assert len(s) == 7

    def test_comprehensive_suite_count(self):
        s = comprehensive_suite()
        assert len(s) >= 21  # 7+7+7 + 2 combined


# ---------------------------------------------------------------------------
# IO tests
# ---------------------------------------------------------------------------

class TestScenarioIO:
    def test_load_scenarios_roundtrip(self):
        data = {
            'name': 'test_suite',
            'baseline': {'gravity_scale': 1.0},
            'scenarios': [
                {'name': 'g1.2', 'modifications': [
                    {'mode': 'scale', 'target': 'gravity_scale', 'value': 1.2}
                ], 'description': 'test'},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        suite = load_scenarios(path)
        assert suite.name == 'test_suite'
        assert len(suite.scenarios) == 1
        assert suite.scenarios[0].description == 'test'

    def test_save_results(self):
        summary = _make_summary()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)
        save_results(summary, path)
        data = json.loads(path.read_text())
        assert data['motion_id'] == 'test_motion'
        assert len(data['results']) == 3

    def test_load_cramer_results(self):
        data = {
            'label': 'crawl',
            'baseline_params': {'g': 1.0},
            'baseline_outputs': {'dx': 1.0},
            'runs': [
                {'scenario': 'g1.2', 'outputs': {'dx': 0.9}, 'success': True},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        summary = load_cramer_results(path)
        assert summary.motion_id == 'crawl'
        assert len(summary.results) == 1


# ---------------------------------------------------------------------------
# Robustness tests
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_perfect_robustness(self):
        baseline = {'dx': 1.0}
        results = [StressResult('s1', {}, {'dx': 1.0}, baseline, True)]
        summary = StressSummary('m', {}, baseline, results)
        r = Robustness(summary)
        assert r.score() == 1.0

    def test_zero_robustness_on_failure(self):
        baseline = {'dx': 1.0}
        results = [StressResult('s1', {}, {}, baseline, False)]
        summary = StressSummary('m', {}, baseline, results)
        r = Robustness(summary)
        assert r.score() == 0.0

    def test_per_metric_scores(self):
        summary = _make_summary()
        r = Robustness(summary, metrics=['dx', 'speed'])
        pms = r.per_metric_scores()
        assert 'dx' in pms
        assert 'speed' in pms

    def test_min_aggregation(self):
        summary = _make_summary()
        r = Robustness(summary, aggregation='min')
        assert 0.0 <= r.score() <= 1.0


class TestRegret:
    def test_zero_regret_is_best(self):
        baseline = {'dx': 1.0}
        results = [StressResult('s1', {}, {'dx': 1.0}, baseline, True)]
        summary = StressSummary('m', {}, baseline, results)
        reg = Regret(summary, [summary])  # compare to self
        r = reg.compute('dx')
        assert r['s1'] == 0.0


# ---------------------------------------------------------------------------
# Vulnerability tests
# ---------------------------------------------------------------------------

class TestVulnerability:
    def test_worst_scenarios(self):
        summary = _make_summary()
        vp = VulnerabilityProfile(summary)
        worst = vp.worst_scenarios(2)
        assert len(worst) == 2

    def test_best_scenarios(self):
        summary = _make_summary()
        vp = VulnerabilityProfile(summary)
        best = vp.best_scenarios(2)
        assert len(best) == 2

    def test_report_keys(self):
        summary = _make_summary()
        vp = VulnerabilityProfile(summary)
        report = vp.report()
        assert 'worst_scenarios' in report
        assert 'overall_robustness' in report
        assert 'per_metric_robustness' in report
