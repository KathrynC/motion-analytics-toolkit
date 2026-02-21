"""Built-in archetypes derived from your persona experiments."""

import numpy as np
from .base import ArchetypeLibrary
from .persona import PersonaArchetype

# Example: hand-crafted archetypes from the persona gaits.
# These could be loaded from a JSON file, but here we define them in code for convenience.

BUILTIN_ARCHETYPES = ArchetypeLibrary([
    PersonaArchetype(
        name="deleuze_fold",
        weight_vector=np.array([0.7, -0.7, 0.0, 0.0, 0.0, 0.0, 0.7, -0.7, 0.0, 0.0]),
        description="Self-feedback motif: w33=0.7, w44=-0.7"
    ),
    PersonaArchetype(
        name="deleuze_bwo",
        weight_vector=np.array([0.55]*6 + [0.0]*4),  # uniform magnitude, no hierarchy
        description="Body without Organs: uniform magnitude 0.55"
    ),
    PersonaArchetype(
        name="borges_mirror",
        weight_vector=np.array([0.3, -0.85, -0.26, -0.85, -0.26, -0.85, 0.0, 0.0, 0.0, 0.0]),
        description="Perfect antisymmetry between motors"
    ),
    PersonaArchetype(
        name="foucault_panopticon",
        weight_vector=np.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0]),
        description="Symmetric surveillance: all weights ±0.5"
    ),
    # ... add more as needed
])

# Optionally, you can also precompute feature vectors for these archetypes
# by running them through the feature extraction pipeline (requires telemetry).
# This would be done in a separate script.
