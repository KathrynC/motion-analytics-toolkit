"""Core classes for scenario definition and stress testing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np


@dataclass
class Scenario:
    """
    A named set of parameter modifications.
    
    Each modification is a dict with keys:
        - 'mode': one of 'scale', 'shift', 'set'
        - 'target': parameter name (e.g., 'gravity_scale', 'friction_scale')
        - 'value': float value to apply
    """
    name: str
    modifications: List[Dict[str, Union[str, float]]]
    description: str = ""
    
    def __add__(self, other: 'Scenario') -> 'Scenario':
        """Combine two scenarios (concatenate modifications)."""
        return Scenario(
            name=f"{self.name}+{other.name}",
            modifications=self.modifications + other.modifications,
            description=f"Combination of {self.name} and {other.name}"
        )


@dataclass
class ScenarioSuite:
    """A collection of scenarios to be run as a test battery."""
    name: str
    scenarios: List[Scenario]
    baseline: Dict[str, float]  # nominal parameter values
    
    def __len__(self):
        return len(self.scenarios)
    
    def get_param_sets(self) -> List[Dict[str, float]]:
        """Generate the parameter dictionaries for each scenario."""
        param_sets = []
        for scen in self.scenarios:
            params = self.baseline.copy()
            for mod in scen.modifications:
                target = mod['target']
                mode = mod['mode']
                val = mod['value']
                if mode == 'scale':
                    params[target] = params.get(target, 1.0) * val
                elif mode == 'shift':
                    params[target] = params.get(target, 0.0) + val
                elif mode == 'set':
                    params[target] = val
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            param_sets.append(params)
        return param_sets


class SimulatorInterface(ABC):
    """
    Abstract interface that any simulator must implement to work with StressTest.
    This matches the Simulator protocol from your System Architecture document.
    """
    
    @abstractmethod
    def run(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Run a single simulation with given parameters.
        Returns a dictionary of output metrics (e.g., {'dx': ..., 'dy': ..., 'speed': ...}).
        """
        pass
    
    @abstractmethod
    def get_baseline_params(self) -> Dict[str, float]:
        """Return the nominal (unperturbed) parameter set."""
        pass
    
    def get_output_keys(self) -> List[str]:
        """Return the names of output metrics (optional)."""
        return ['dx', 'dy', 'speed', 'efficiency']  # default, override if needed


@dataclass
class StressResult:
    """Result of running a single scenario on a motion."""
    scenario_name: str
    params: Dict[str, float]
    outputs: Dict[str, float]  # key: output metric, value: value under stress
    baseline_outputs: Dict[str, float]  # outputs at baseline
    success: bool  # whether simulation completed without errors


@dataclass
class StressSummary:
    """Collection of results for a full suite run."""
    motion_id: str
    baseline_params: Dict[str, float]
    baseline_outputs: Dict[str, float]
    results: List[StressResult]
    
    def get_output_matrix(self, metric: str) -> np.ndarray:
        """Return an array of the given metric across all scenarios."""
        return np.array([r.outputs.get(metric, np.nan) for r in self.results])


class StressTest:
    """
    Run a motion (via a simulator) through a suite of scenarios.
    """
    
    def __init__(self, simulator: SimulatorInterface):
        self.simulator = simulator
    
    def run_suite(self, suite: ScenarioSuite, motion_id: str = "unknown") -> StressSummary:
        """Run all scenarios in the suite and return a summary."""
        baseline_params = self.simulator.get_baseline_params()
        baseline_outputs = self.simulator.run(baseline_params)
        
        results = []
        param_sets = suite.get_param_sets()
        for scen, params in zip(suite.scenarios, param_sets):
            try:
                outputs = self.simulator.run(params)
                success = True
            except Exception as e:
                outputs = {}
                success = False
                # In a real implementation, you might log the error.
            results.append(StressResult(
                scenario_name=scen.name,
                params=params,
                outputs=outputs,
                baseline_outputs=baseline_outputs,
                success=success
            ))
        
        return StressSummary(
            motion_id=motion_id,
            baseline_params=baseline_params,
            baseline_outputs=baseline_outputs,
            results=results
        )
