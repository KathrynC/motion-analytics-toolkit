"""Metaphor violation detection following Lakoff's ICM framework.

The MetaphorAuditor checks whether an archetype label's grounding criteria
and ICM assumptions hold for a given motion. If grounding fails or the ICM
is violated, the label is a metaphor violation — it highlights features
that aren't present and hides the absence.
"""

from typing import Dict, List

from .base import Archetype, ArchetypeLibrary, extract_behavioral_features, get_feature_layer
from ..core.schemas import Telemetry


class MetaphorAuditor:
    """Audit archetype labels against observed motion features."""

    def __init__(self, library: ArchetypeLibrary):
        self.library = library

    def audit(self, telemetry: Telemetry) -> Dict[str, Dict]:
        """Audit all archetypes in the library against the given telemetry.

        Returns:
            {archetype_name: {
                'similarity': float,
                'grounding_pass': bool,
                'failed_criteria': list[str],
                'icm_violated': bool,
                'icm_violations': list[str],
                'verdict': str,  # 'grounded', 'partial', 'violated'
            }}
        """
        features = extract_behavioral_features(telemetry)
        results = {}
        for arch in self.library.archetypes:
            results[arch.name] = self.audit_single(arch, features, telemetry)
        return results

    def audit_single(
        self,
        archetype: Archetype,
        features: Dict[str, float],
        telemetry: Telemetry = None,
    ) -> Dict:
        """Audit one archetype against extracted features.

        Args:
            archetype: The archetype to audit.
            features: Pre-extracted behavioral features dict.
            telemetry: Optional telemetry for similarity computation.
        """
        # Similarity score
        similarity = 0.0
        if telemetry is not None:
            similarity = archetype.similarity_to(telemetry)

        # Grounding check
        grounding_pass, failed_criteria = archetype.check_grounding(features)

        # ICM check
        icm_violated = False
        icm_violations: List[str] = []
        if archetype.icm is not None:
            icm_violations = archetype.icm.check_violations(features)
            icm_violated = len(icm_violations) > 0

        # Layer enforcement check
        layer_warnings: List[str] = []
        for gc in archetype.grounding_criteria:
            feature_layer = get_feature_layer(gc.feature)
            if feature_layer == 'linking':
                layer_warnings.append(
                    f"'{gc.feature}' is a linking feature used in grounding criterion: {gc.rationale}"
                )

        # Verdict
        if grounding_pass and not icm_violated:
            verdict = 'grounded'
        elif icm_violated:
            verdict = 'violated'
        else:
            verdict = 'partial'

        return {
            'similarity': float(similarity),
            'grounding_pass': grounding_pass,
            'failed_criteria': failed_criteria,
            'icm_violated': icm_violated,
            'icm_violations': icm_violations,
            'layer_warnings': layer_warnings,
            'verdict': verdict,
        }
