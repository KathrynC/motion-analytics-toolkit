"""Input/output helpers for scenarios and results."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import Scenario, ScenarioSuite, StressSummary


def load_scenarios(path: Path) -> ScenarioSuite:
    """
    Load a scenario suite from a JSON file.
    
    Expected format:
    {
        "name": "suite_name",
        "baseline": {"gravity_scale": 1.0, "friction_scale": 1.0, ...},
        "scenarios": [
            {
                "name": "gravity+20%",
                "modifications": [{"mode": "scale", "target": "gravity_scale", "value": 1.2}],
                "description": "..."
            },
            ...
        ]
    }
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    scenarios = []
    for s in data['scenarios']:
        scenarios.append(Scenario(
            name=s['name'],
            modifications=s['modifications'],
            description=s.get('description', '')
        ))
    
    return ScenarioSuite(
        name=data['name'],
        scenarios=scenarios,
        baseline=data['baseline']
    )


def save_results(summary: StressSummary, path: Path):
    """Save stress test results to JSON."""
    data = {
        'motion_id': summary.motion_id,
        'baseline_params': summary.baseline_params,
        'baseline_outputs': summary.baseline_outputs,
        'results': [
            {
                'scenario_name': r.scenario_name,
                'params': r.params,
                'outputs': r.outputs,
                'success': r.success
            }
            for r in summary.results
        ]
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_cramer_results(path: Path, motion_id: Optional[str] = None) -> StressSummary:
    """
    Load pre‑computed robustness results from a Cramer‑toolkit JSON file.
    This allows using existing data without re‑running simulations.
    
    Expected format (from your cramer_bridge.py output):
    {
        "label": "crawl",
        "robustness_scores": {...},
        "vulnerability_profiles": {...},
        "runs": [
            {"scenario": "gravity+20%", "outputs": {"dx": ..., "speed": ...}, ...}
        ]
    }
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert to our internal format
    baseline_params = data.get('baseline_params', {})
    baseline_outputs = data.get('baseline_outputs', {})
    results = []
    for run in data.get('runs', []):
        results.append(StressResult(
            scenario_name=run['scenario'],
            params=run.get('params', {}),
            outputs=run['outputs'],
            baseline_outputs=baseline_outputs,
            success=run.get('success', True)
        ))
    
    return StressSummary(
        motion_id=motion_id or data.get('label', 'unknown'),
        baseline_params=baseline_params,
        baseline_outputs=baseline_outputs,
        results=results
    )
