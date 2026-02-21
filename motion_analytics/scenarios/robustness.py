"""Robustness scoring and regret analysis."""

import numpy as np
from typing import Dict, List, Optional, Callable
from .base import StressSummary


class Robustness:
    """
    Compute robustness scores (0 to 1) for a motion under a stress test.
    
    Robustness is defined as how well the motion maintains its key performance
    metrics across scenarios. Higher is better.
    """
    
    def __init__(self, summary: StressSummary, 
                 metrics: Optional[List[str]] = None,
                 aggregation: str = 'mean'):
        """
        Args:
            summary: Result of a stress test.
            metrics: List of metric names to consider (default: all output keys).
            aggregation: How to combine metrics: 'mean', 'min', or callable.
        """
        self.summary = summary
        self.metrics = metrics or list(summary.baseline_outputs.keys())
        self.aggregation = aggregation
    
    def score(self, tolerance: float = 0.1) -> float:
        """
        Compute overall robustness.
        
        For each scenario, compute degradation for each metric:
            degradation = |value - baseline| / (|baseline| + epsilon)
        Then compute a per-scenario score = 1 - min(1, degradation / tolerance).
        Finally aggregate across scenarios and metrics.
        """
        epsilon = 1e-10
        per_scenario_scores = []
        
        for res in self.summary.results:
            if not res.success:
                per_scenario_scores.append(0.0)
                continue
            metric_scores = []
            for metric in self.metrics:
                base = self.summary.baseline_outputs.get(metric, 0.0)
                val = res.outputs.get(metric, base)
                if abs(base) < epsilon:
                    # If baseline is zero, any deviation is infinite degradation.
                    deg = 1.0 if abs(val) > epsilon else 0.0
                else:
                    deg = abs(val - base) / (abs(base) + epsilon)
                # Map degradation to a score: 1 = no degradation, 0 = degraded beyond tolerance.
                score = max(0.0, 1.0 - min(1.0, deg / tolerance))
                metric_scores.append(score)
            # Aggregate across metrics for this scenario
            if self.aggregation == 'mean':
                scen_score = np.mean(metric_scores)
            elif self.aggregation == 'min':
                scen_score = np.min(metric_scores)
            else:
                scen_score = self.aggregation(metric_scores)
            per_scenario_scores.append(scen_score)
        
        # Aggregate across scenarios
        return float(np.mean(per_scenario_scores))
    
    def per_metric_scores(self, tolerance: float = 0.1) -> Dict[str, float]:
        """Compute robustness for each metric individually."""
        epsilon = 1e-10
        metric_scores = {m: [] for m in self.metrics}
        
        for res in self.summary.results:
            if not res.success:
                for m in self.metrics:
                    metric_scores[m].append(0.0)
                continue
            for metric in self.metrics:
                base = self.summary.baseline_outputs.get(metric, 0.0)
                val = res.outputs.get(metric, base)
                if abs(base) < epsilon:
                    deg = 1.0 if abs(val) > epsilon else 0.0
                else:
                    deg = abs(val - base) / (abs(base) + epsilon)
                score = max(0.0, 1.0 - min(1.0, deg / tolerance))
                metric_scores[metric].append(score)
        
        return {m: float(np.mean(scores)) for m, scores in metric_scores.items()}


class Regret:
    """
    Compute regret: the gap between a motion's performance and the best achievable
    under each scenario. This requires a reference set of other motions.
    """
    
    def __init__(self, summary: StressSummary, reference_results: List[StressSummary]):
        self.summary = summary
        self.reference = reference_results
    
    def compute(self, metric: str = 'dx') -> Dict[str, float]:
        """
        For each scenario, compute regret = (best_reference - this_motion) / (best_reference + epsilon).
        Returns dict mapping scenario name to regret (0 = optimal, 1 = worst).
        """
        epsilon = 1e-10
        regrets = {}
        
        # Build a map of scenario name to best reference value
        best_map = {}
        for ref in self.reference:
            for res in ref.results:
                if res.success:
                    key = res.scenario_name
                    val = res.outputs.get(metric, 0.0)
                    if key not in best_map or val > best_map[key]:
                        best_map[key] = val
        
        for res in self.summary.results:
            key = res.scenario_name
            if not res.success:
                regrets[key] = 1.0  # complete failure
                continue
            best = best_map.get(key, 0.0)
            val = res.outputs.get(metric, 0.0)
            if best <= epsilon:
                regret = 0.0 if val <= epsilon else 1.0
            else:
                regret = (best - val) / best
                regret = max(0.0, min(1.0, regret))
            regrets[key] = regret
        
        return regrets
