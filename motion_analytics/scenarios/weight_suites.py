"""Weight-space scenario suites for archetype robustness testing.

Five suite factories probe how archetypes respond to weight-space perturbations:
  1. weight_perturbation_suite  — cliffiness via per-synapse +-epsilon shifts
  2. synapse_ablation_suite     — load-bearing connections via zeroing
  3. archetype_boundary_suite   — ICM violation probing via directional pushes
  4. archetype_interpolation_suite — landscape smoothness between two archetypes
  5. cross_topology_suite       — motor-to-motor crosswiring patterns

Plus comprehensive_weight_suite combining suites 1-4 (not interpolation, which
requires two weight vectors).
"""

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base import Scenario, ScenarioSuite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNAPSE_NAMES_6 = ('w03', 'w04', 'w13', 'w14', 'w23', 'w24')
SYNAPSE_NAMES_10 = SYNAPSE_NAMES_6 + ('w33', 'w34', 'w43', 'w44')

_CROSSWIRED = ('w33', 'w34', 'w43', 'w44')

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _weight_vector_to_baseline(
    weights: Union[List[float], np.ndarray],
    include_crosswired: bool = False,
) -> Dict[str, float]:
    """Convert a 6-or-10 element weight vector to a string-keyed param dict.

    If *include_crosswired* is True, the result always has 10 entries
    (zero-padded if the input has only 6 elements).
    """
    weights = list(np.asarray(weights, dtype=float).flat)
    use_10 = include_crosswired or len(weights) > 6
    names = SYNAPSE_NAMES_10 if use_10 else SYNAPSE_NAMES_6

    if use_10 and len(weights) < 10:
        weights = weights[:6] + [0.0] * (10 - len(weights))

    return {n: float(weights[i]) for i, n in enumerate(names[:len(weights)])}


# ---------------------------------------------------------------------------
# Suite 1 — Weight Perturbation (cliffiness)
# ---------------------------------------------------------------------------

def weight_perturbation_suite(
    weights: Union[List[float], np.ndarray],
    epsilons: Sequence[float] = (0.05, 0.1, 0.2),
    synapse_subset: Optional[Sequence[str]] = None,
    name: str = "weight_perturbation",
) -> ScenarioSuite:
    """Perturb each synapse by +/-epsilon at multiple magnitudes.

    Parameters
    ----------
    weights : array-like
        6-or-10 element weight vector (baseline).
    epsilons : sequence of float
        Perturbation magnitudes.
    synapse_subset : sequence of str, optional
        Restrict to these synapse names (default: all 6 core).
    name : str
        Suite name.

    Returns a ScenarioSuite with 'shift' mode modifications.
    """
    baseline = _weight_vector_to_baseline(weights)
    targets = list(synapse_subset or SYNAPSE_NAMES_6)

    scenarios: List[Scenario] = []
    for syn in targets:
        for eps in epsilons:
            for sign, label in [(1, '+'), (-1, '-')]:
                scenarios.append(Scenario(
                    name=f"{syn}{label}{eps}",
                    modifications=[{
                        'mode': 'shift',
                        'target': syn,
                        'value': sign * eps,
                    }],
                ))
    return ScenarioSuite(name=name, scenarios=scenarios, baseline=baseline)


# ---------------------------------------------------------------------------
# Suite 2 — Synapse Ablation (load-bearing connections)
# ---------------------------------------------------------------------------

def synapse_ablation_suite(
    weights: Union[List[float], np.ndarray],
    include_pairs: bool = True,
    synapse_subset: Optional[Sequence[str]] = None,
    name: str = "synapse_ablation",
) -> ScenarioSuite:
    """Zero out individual synapses and optionally pairs.

    Parameters
    ----------
    weights : array-like
        Baseline weight vector.
    include_pairs : bool
        If True, also ablate all C(n,2) pairs.
    synapse_subset : sequence of str, optional
        Restrict to these synapse names (default: all 6 core).
    name : str
        Suite name.

    Returns a ScenarioSuite with 'set' mode modifications (value=0.0).
    """
    baseline = _weight_vector_to_baseline(weights)
    targets = list(synapse_subset or SYNAPSE_NAMES_6)

    scenarios: List[Scenario] = []

    # Singles
    for syn in targets:
        scenarios.append(Scenario(
            name=f"ablate_{syn}",
            modifications=[{'mode': 'set', 'target': syn, 'value': 0.0}],
        ))

    # Pairs
    if include_pairs:
        for a, b in combinations(targets, 2):
            scenarios.append(Scenario(
                name=f"ablate_{a}+{b}",
                modifications=[
                    {'mode': 'set', 'target': a, 'value': 0.0},
                    {'mode': 'set', 'target': b, 'value': 0.0},
                ],
            ))

    return ScenarioSuite(name=name, scenarios=scenarios, baseline=baseline)


# ---------------------------------------------------------------------------
# Suite 3 — Archetype Boundary (ICM violation probing)
# ---------------------------------------------------------------------------

def archetype_boundary_suite(
    weights: Union[List[float], np.ndarray],
    archetype_name: str = "",
    boundary_fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    name: Optional[str] = None,
) -> ScenarioSuite:
    """Push each weight in 3 directions at graded fractions.

    Directions per weight:
      - toward_zero:  interpolate current value toward 0.0
      - sign_flip:    interpolate toward -current_value
      - saturate:     interpolate toward +1.0 or -1.0 (sign-preserving)

    Degenerate probes (weight already at target) are skipped.

    Parameters
    ----------
    weights : array-like
        Baseline weight vector.
    archetype_name : str
        Used in scenario naming (e.g. ``"crab_w03_toward_zero_50%"``).
    boundary_fractions : sequence of float
        Interpolation fractions (0 = baseline, 1 = target).
    name : str, optional
        Suite name (default: ``"boundary_{archetype_name}"`` or ``"archetype_boundary"``).
    """
    baseline = _weight_vector_to_baseline(weights)
    suite_name = name or (f"boundary_{archetype_name}" if archetype_name else "archetype_boundary")
    prefix = f"{archetype_name}_" if archetype_name else ""

    scenarios: List[Scenario] = []
    for syn, val in baseline.items():
        directions: List[Tuple[str, float]] = []

        # toward_zero
        if abs(val) > 1e-10:
            directions.append(('toward_zero', 0.0))

        # sign_flip
        if abs(val) > 1e-10:
            directions.append(('sign_flip', -val))

        # saturate — toward +-1.0 preserving sign (or +1.0 if val==0)
        sat_target = 1.0 if val >= 0 else -1.0
        if abs(val - sat_target) > 1e-10:
            directions.append(('saturate', sat_target))

        for dir_label, target in directions:
            for frac in boundary_fractions:
                new_val = val + frac * (target - val)
                pct = int(frac * 100)
                scenarios.append(Scenario(
                    name=f"{prefix}{syn}_{dir_label}_{pct}%",
                    modifications=[{'mode': 'set', 'target': syn, 'value': new_val}],
                ))

    return ScenarioSuite(name=suite_name, scenarios=scenarios, baseline=baseline)


# ---------------------------------------------------------------------------
# Suite 4 — Archetype Interpolation (landscape smoothness)
# ---------------------------------------------------------------------------

def archetype_interpolation_suite(
    weights_a: Union[List[float], np.ndarray],
    weights_b: Union[List[float], np.ndarray],
    steps: int = 11,
    label_a: str = "A",
    label_b: str = "B",
    name: Optional[str] = None,
) -> ScenarioSuite:
    """Linear interpolation (1-t)*A + t*B at equal steps from 0 to 1.

    Mixed 6/10 element vectors are handled by zero-padding the shorter one.

    Parameters
    ----------
    weights_a, weights_b : array-like
        Endpoint weight vectors.
    steps : int
        Number of interpolation points (including both endpoints).
    label_a, label_b : str
        Names for the endpoints (used in scenario naming).
    name : str, optional
        Suite name.
    """
    a = list(np.asarray(weights_a, dtype=float).flat)
    b = list(np.asarray(weights_b, dtype=float).flat)

    # Zero-pad to equal length
    max_len = max(len(a), len(b))
    a = a + [0.0] * (max_len - len(a))
    b = b + [0.0] * (max_len - len(b))

    names_for_len = SYNAPSE_NAMES_10[:max_len]
    baseline = {n: float(a[i]) for i, n in enumerate(names_for_len)}
    suite_name = name or f"interp_{label_a}_to_{label_b}"

    scenarios: List[Scenario] = []
    for step_i in range(steps):
        t = step_i / (steps - 1) if steps > 1 else 0.0
        mods = []
        for i, syn in enumerate(names_for_len):
            val = (1 - t) * a[i] + t * b[i]
            mods.append({'mode': 'set', 'target': syn, 'value': val})
        scenarios.append(Scenario(
            name=f"interp_{label_a}_to_{label_b}_{t:.2f}",
            modifications=mods,
        ))

    return ScenarioSuite(name=suite_name, scenarios=scenarios, baseline=baseline)


# ---------------------------------------------------------------------------
# Suite 5 — Cross-Topology (motor-to-motor connections)
# ---------------------------------------------------------------------------

_CROSSWIRE_PATTERNS: Dict[str, List[str]] = {
    'self_back':  ['w33'],
    'self_front': ['w44'],
    'cross_sym':  ['w34', 'w43'],
    'cross_anti': ['w34', 'w43'],
    'full_cross': ['w33', 'w34', 'w43', 'w44'],
}


def cross_topology_suite(
    weights: Union[List[float], np.ndarray],
    crosswire_strengths: Sequence[float] = (0.1, 0.3, 0.5, 0.7),
    name: str = "cross_topology",
) -> ScenarioSuite:
    """Activate crosswired motor-to-motor weights in 5 patterns at +/-strength.

    Patterns:
      - self_back:   w33 only (back motor self-feedback)
      - self_front:  w44 only (front motor self-feedback)
      - cross_sym:   w34 = w43 (symmetric cross-coupling)
      - cross_anti:  w34 = -w43 (antisymmetric, CPG-like)
      - full_cross:  all 4 crosswired weights

    Parameters
    ----------
    weights : array-like
        Baseline weight vector (6 or 10 elements; crosswired weights zero-padded).
    crosswire_strengths : sequence of float
        Magnitudes to try for each pattern.
    name : str
        Suite name.
    """
    baseline = _weight_vector_to_baseline(weights, include_crosswired=True)

    scenarios: List[Scenario] = []
    for pattern_name, synapses in _CROSSWIRE_PATTERNS.items():
        for strength in crosswire_strengths:
            for sign in [1, -1]:
                signed = sign * strength
                mods = []
                for syn in synapses:
                    if pattern_name == 'cross_anti' and syn == 'w43':
                        val = -signed
                    else:
                        val = signed
                    mods.append({'mode': 'set', 'target': syn, 'value': val})
                scenarios.append(Scenario(
                    name=f"{pattern_name}_{'+' if sign > 0 else '-'}{strength}",
                    modifications=mods,
                ))

    return ScenarioSuite(name=name, scenarios=scenarios, baseline=baseline)


# ---------------------------------------------------------------------------
# Convenience Composite
# ---------------------------------------------------------------------------

def comprehensive_weight_suite(
    weights: Union[List[float], np.ndarray],
    archetype_name: str = "",
    name: Optional[str] = None,
) -> ScenarioSuite:
    """Combine perturbation, ablation, boundary, and cross-topology suites.

    Does NOT include interpolation (which requires two weight vectors).
    Uses a 10-element baseline for cross-topology compatibility.
    """
    baseline_10 = _weight_vector_to_baseline(weights, include_crosswired=True)
    suite_name = name or (f"comprehensive_{archetype_name}" if archetype_name else "comprehensive_weight")

    s1 = weight_perturbation_suite(weights)
    s2 = synapse_ablation_suite(weights)
    s3 = archetype_boundary_suite(weights, archetype_name=archetype_name)
    s4 = cross_topology_suite(weights)

    all_scenarios = s1.scenarios + s2.scenarios + s3.scenarios + s4.scenarios
    return ScenarioSuite(name=suite_name, scenarios=all_scenarios, baseline=baseline_10)


# ---------------------------------------------------------------------------
# Archetype-to-suite bridge helpers
# ---------------------------------------------------------------------------

def _get_archetype_weights(library, archetype_name: str) -> np.ndarray:
    """Extract weight vector from a named archetype, raising on failure."""
    arch = library.get(archetype_name)
    if arch is None:
        raise KeyError(f"Archetype '{archetype_name}' not found in library")
    wv = getattr(arch, 'weight_vector', None)
    if wv is None:
        raise ValueError(f"Archetype '{archetype_name}' has no weight_vector")
    return np.asarray(wv)


def suite_from_archetype(
    library,
    archetype_name: str,
    suite_type: str = "comprehensive",
    **kwargs,
) -> ScenarioSuite:
    """Build a weight suite for a named archetype from a library.

    Parameters
    ----------
    library : ArchetypeLibrary
        Library containing the archetype.
    archetype_name : str
        Name of the archetype to test.
    suite_type : str
        One of 'perturbation', 'ablation', 'boundary', 'cross_topology',
        'comprehensive'.
    **kwargs
        Forwarded to the underlying suite factory.
    """
    wv = _get_archetype_weights(library, archetype_name)

    factories = {
        'perturbation': weight_perturbation_suite,
        'ablation': synapse_ablation_suite,
        'boundary': archetype_boundary_suite,
        'cross_topology': cross_topology_suite,
        'comprehensive': comprehensive_weight_suite,
    }
    factory = factories.get(suite_type)
    if factory is None:
        raise ValueError(
            f"Unknown suite_type '{suite_type}'; choose from {list(factories)}"
        )

    if suite_type in ('boundary', 'comprehensive'):
        kwargs.setdefault('archetype_name', archetype_name)

    return factory(wv, **kwargs)


def interpolation_from_library(
    library,
    name_a: str,
    name_b: str,
    steps: int = 11,
    **kwargs,
) -> ScenarioSuite:
    """Build an interpolation suite between two named archetypes.

    Parameters
    ----------
    library : ArchetypeLibrary
        Library containing both archetypes.
    name_a, name_b : str
        Archetype endpoint names.
    steps : int
        Number of interpolation points (including both endpoints).
    **kwargs
        Forwarded to archetype_interpolation_suite.
    """
    wv_a = _get_archetype_weights(library, name_a)
    wv_b = _get_archetype_weights(library, name_b)
    kwargs.setdefault('label_a', name_a)
    kwargs.setdefault('label_b', name_b)
    return archetype_interpolation_suite(wv_a, wv_b, steps=steps, **kwargs)
