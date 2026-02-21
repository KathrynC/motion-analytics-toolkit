"""Built-in archetypes derived from persona experiments, with Lakoff grounding."""

import numpy as np
from .base import ArchetypeLibrary, GroundingCriterion, ICM
from .persona import PersonaArchetype


# ------------------------------------------------------------------
# Grounding criteria and ICMs for each archetype
# ------------------------------------------------------------------

_fold_grounding = [
    GroundingCriterion(
        feature='phase_lock', predicate='gt', value=0.7,
        rationale="Self-feedback requires high inter-leg coordination (fold = return to self)",
    ),
    GroundingCriterion(
        feature='curvature_complexity', predicate='gt', value=0.01,
        rationale="Folding path shows non-trivial curvature variation",
    ),
]

_fold_icm = ICM(
    name='fold_icm',
    background=[
        "Motion exhibits self-referential feedback loop",
        "Path folds back on itself rather than progressing linearly",
        "High coordination between limbs (fold = return)",
    ],
    violation_conditions=[
        GroundingCriterion(
            feature='phase_lock', predicate='lt', value=0.3,
            rationale="No self-feedback loop: limbs move independently",
        ),
        GroundingCriterion(
            feature='straightness', predicate='gt', value=0.95,
            rationale="Perfectly straight path contradicts folding",
        ),
    ],
)

_bwo_grounding = [
    GroundingCriterion(
        feature='duty_factor_asymmetry', predicate='lt', value=0.1,
        rationale="Body without Organs: no hierarchical differentiation between limbs",
    ),
    GroundingCriterion(
        feature='straightness', predicate='between', value=0.2, tolerance=0.8,
        rationale="BwO is neither purely directed nor purely chaotic",
    ),
]

_bwo_icm = ICM(
    name='bwo_icm',
    background=[
        "Motion lacks organ-like functional differentiation",
        "All limbs contribute equally (uniform magnitude)",
        "No dominant frequency or rhythm hierarchy",
    ],
    violation_conditions=[
        GroundingCriterion(
            feature='duty_factor_asymmetry', predicate='gt', value=0.4,
            rationale="Strong asymmetry implies organ-like specialization",
        ),
    ],
)

_mirror_grounding = [
    GroundingCriterion(
        feature='symmetry_index', predicate='near', value=0.0, tolerance=50.0,
        rationale="Mirror requires near-perfect antisymmetry between motor groups",
    ),
    GroundingCriterion(
        feature='phase_lock', predicate='gt', value=0.8,
        rationale="Mirror reflection demands tight phase coupling",
    ),
]

_mirror_icm = ICM(
    name='mirror_icm',
    background=[
        "Motion of one limb is the precise inverse of the other",
        "Temporal coupling is tight (reflection requires simultaneity)",
        "The two halves are distinguishable but complementary",
    ],
    violation_conditions=[
        GroundingCriterion(
            feature='phase_lock', predicate='lt', value=0.4,
            rationale="No reflection: limbs are decoupled",
        ),
    ],
)

_panopticon_grounding = [
    GroundingCriterion(
        feature='duty_factor_asymmetry', predicate='lt', value=0.05,
        rationale="Symmetric surveillance: all limbs equally active",
    ),
    GroundingCriterion(
        feature='workspace_volume', predicate='gt', value=0.0,
        rationale="Panopticon covers observable space (non-zero workspace)",
    ),
]

_panopticon_icm = ICM(
    name='panopticon_icm',
    background=[
        "Motion maintains symmetric monitoring of environment",
        "Equal distribution of contact and force across limbs",
        "Workspace is actively explored, not stationary",
    ],
    violation_conditions=[
        GroundingCriterion(
            feature='duty_factor_asymmetry', predicate='gt', value=0.3,
            rationale="Asymmetric motion contradicts panoptic symmetry",
        ),
    ],
)


# ------------------------------------------------------------------
# Builtin archetype library
# ------------------------------------------------------------------

BUILTIN_ARCHETYPES = ArchetypeLibrary([
    PersonaArchetype(
        name="deleuze_fold",
        weight_vector=np.array([0.7, -0.7, 0.0, 0.0, 0.0, 0.0, 0.7, -0.7, 0.0, 0.0]),
        description="Self-feedback motif: w33=0.7, w44=-0.7",
        grounding_criteria=_fold_grounding,
        icm=_fold_icm,
    ),
    PersonaArchetype(
        name="deleuze_bwo",
        weight_vector=np.array([0.55]*6 + [0.0]*4),
        description="Body without Organs: uniform magnitude 0.55",
        grounding_criteria=_bwo_grounding,
        icm=_bwo_icm,
    ),
    PersonaArchetype(
        name="borges_mirror",
        weight_vector=np.array([0.3, -0.85, -0.26, -0.85, -0.26, -0.85, 0.0, 0.0, 0.0, 0.0]),
        description="Perfect antisymmetry between motors",
        grounding_criteria=_mirror_grounding,
        icm=_mirror_icm,
    ),
    PersonaArchetype(
        name="foucault_panopticon",
        weight_vector=np.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0]),
        description="Symmetric surveillance: all weights +/-0.5",
        grounding_criteria=_panopticon_grounding,
        icm=_panopticon_icm,
    ),
])
