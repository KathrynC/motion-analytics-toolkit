"""Pre‑defined scenario suites for common environmental variations."""

from .base import Scenario, ScenarioSuite

def gravity_suite(name: str = "gravity_variations") -> ScenarioSuite:
    """Suite with gravity scaled from 0.5 to 2.0."""
    baseline = {'gravity_scale': 1.0, 'friction_scale': 1.0, 'max_force_scale': 1.0}
    scales = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]
    scenarios = [
        Scenario(
            name=f"gravity_{s:.1f}x",
            modifications=[{'mode': 'scale', 'target': 'gravity_scale', 'value': s}]
        )
        for s in scales
    ]
    return ScenarioSuite(name=name, scenarios=scenarios, baseline=baseline)

def friction_suite(name: str = "friction_variations") -> ScenarioSuite:
    """Suite with friction scaled from 0.3 to 3.0."""
    baseline = {'gravity_scale': 1.0, 'friction_scale': 1.0, 'max_force_scale': 1.0}
    scales = [0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0]
    scenarios = [
        Scenario(
            name=f"friction_{s:.1f}x",
            modifications=[{'mode': 'scale', 'target': 'friction_scale', 'value': s}]
        )
        for s in scales
    ]
    return ScenarioSuite(name=name, scenarios=scenarios, baseline=baseline)

def force_suite(name: str = "force_variations") -> ScenarioSuite:
    """Suite with max force scaled from 0.5 to 2.0."""
    baseline = {'gravity_scale': 1.0, 'friction_scale': 1.0, 'max_force_scale': 1.0}
    scales = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]
    scenarios = [
        Scenario(
            name=f"force_{s:.1f}x",
            modifications=[{'mode': 'scale', 'target': 'max_force_scale', 'value': s}]
        )
        for s in scales
    ]
    return ScenarioSuite(name=name, scenarios=scenarios, baseline=baseline)

def comprehensive_suite(name: str = "comprehensive") -> ScenarioSuite:
    """Combine all basic variations."""
    gravity = gravity_suite().scenarios
    friction = friction_suite().scenarios
    force = force_suite().scenarios
    baseline = {'gravity_scale': 1.0, 'friction_scale': 1.0, 'max_force_scale': 1.0}
    scenarios = gravity + friction + force
    # Add a few combined scenarios
    scenarios.append(Scenario(
        name="gravity_1.3+friction_0.7",
        modifications=[
            {'mode': 'scale', 'target': 'gravity_scale', 'value': 1.3},
            {'mode': 'scale', 'target': 'friction_scale', 'value': 0.7}
        ]
    ))
    scenarios.append(Scenario(
        name="gravity_1.5+force_0.7",
        modifications=[
            {'mode': 'scale', 'target': 'gravity_scale', 'value': 1.5},
            {'mode': 'scale', 'target': 'max_force_scale', 'value': 0.7}
        ]
    ))
    return ScenarioSuite(name=name, scenarios=scenarios, baseline=baseline)
