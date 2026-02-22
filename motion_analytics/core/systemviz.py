"""SystemViz Visual Vocabulary tagging for motion analytics entities.

Based on Peter Stoyko's Visual Vocabulary of Systems v1.1 (CC-BY).
Six categories: Driver, Signal, State, Boundary, Relation, Domain.

Each toolkit entity (lattice node, archetype, scenario parameter, image schema,
Wolfram class) can be tagged with SystemViz terms via a SystemVizProfile.
Default tagging functions provide sensible starting tags for each entity type.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any


class SystemVizCategory(Enum):
    """The six suits of the SystemViz Visual Vocabulary."""
    DRIVER = "driver"
    SIGNAL = "signal"
    STATE = "state"
    BOUNDARY = "boundary"
    RELATION = "relation"
    DOMAIN = "domain"


# Curated subset of ~65 elements most relevant to motion analytics.
SYSTEMVIZ_ELEMENTS: Dict[str, SystemVizCategory] = {
    # Drivers (active causes)
    'attractor': SystemVizCategory.DRIVER,
    'repeller': SystemVizCategory.DRIVER,
    'cycle': SystemVizCategory.DRIVER,
    'cascade': SystemVizCategory.DRIVER,
    'dampener': SystemVizCategory.DRIVER,
    'amplifier': SystemVizCategory.DRIVER,
    'disruptor': SystemVizCategory.DRIVER,
    'wave': SystemVizCategory.DRIVER,
    'lag': SystemVizCategory.DRIVER,
    'branching': SystemVizCategory.DRIVER,
    'inflection': SystemVizCategory.DRIVER,
    'equifinality': SystemVizCategory.DRIVER,
    'goal_seeking': SystemVizCategory.DRIVER,
    'enabler': SystemVizCategory.DRIVER,
    'opposer': SystemVizCategory.DRIVER,
    'conduit': SystemVizCategory.DRIVER,
    'diffusion': SystemVizCategory.DRIVER,
    'side_effect': SystemVizCategory.DRIVER,
    'outlier': SystemVizCategory.DRIVER,
    # Signals (communication)
    'feedback': SystemVizCategory.SIGNAL,
    'feed_forward': SystemVizCategory.SIGNAL,
    'indicator': SystemVizCategory.SIGNAL,
    'monitor': SystemVizCategory.SIGNAL,
    'noise': SystemVizCategory.SIGNAL,
    'cue': SystemVizCategory.SIGNAL,
    'decay': SystemVizCategory.SIGNAL,
    'trail': SystemVizCategory.SIGNAL,
    'echo': SystemVizCategory.SIGNAL,
    'framing': SystemVizCategory.SIGNAL,
    # States (conditions)
    'stock': SystemVizCategory.STATE,
    'transition': SystemVizCategory.STATE,
    'equilibrium': SystemVizCategory.STATE,
    'phase_space': SystemVizCategory.STATE,
    'tipping_point': SystemVizCategory.STATE,
    'criticality': SystemVizCategory.STATE,
    'emergent_property': SystemVizCategory.STATE,
    'mutation': SystemVizCategory.STATE,
    'self_similarity': SystemVizCategory.STATE,
    'accumulation': SystemVizCategory.STATE,
    'capacity': SystemVizCategory.STATE,
    'fault': SystemVizCategory.STATE,
    # Boundaries
    'container': SystemVizCategory.BOUNDARY,
    'semi_permeable': SystemVizCategory.BOUNDARY,
    'barrier': SystemVizCategory.BOUNDARY,
    'edge': SystemVizCategory.BOUNDARY,
    'fuzzy_boundary': SystemVizCategory.BOUNDARY,
    'bridge': SystemVizCategory.BOUNDARY,
    'buffer': SystemVizCategory.BOUNDARY,
    'bottleneck': SystemVizCategory.BOUNDARY,
    'liminality': SystemVizCategory.BOUNDARY,
    # Relations
    'cluster': SystemVizCategory.RELATION,
    'coupling': SystemVizCategory.RELATION,
    'synchrony': SystemVizCategory.RELATION,
    'coordination': SystemVizCategory.RELATION,
    'competition': SystemVizCategory.RELATION,
    'complementarity': SystemVizCategory.RELATION,
    'differentiation': SystemVizCategory.RELATION,
    'network': SystemVizCategory.RELATION,
    'holon': SystemVizCategory.RELATION,
    'adjacency': SystemVizCategory.RELATION,
    'symmathesy': SystemVizCategory.RELATION,
    # Domains
    'topology': SystemVizCategory.DOMAIN,
    'periphery': SystemVizCategory.DOMAIN,
    'centrality': SystemVizCategory.DOMAIN,
    'friction': SystemVizCategory.DOMAIN,
    'zone': SystemVizCategory.DOMAIN,
    'nesting': SystemVizCategory.DOMAIN,
    'levels_of_scale': SystemVizCategory.DOMAIN,
    'entropy': SystemVizCategory.DOMAIN,
    'uncertainty': SystemVizCategory.DOMAIN,
    'condition_shock': SystemVizCategory.DOMAIN,
    'density': SystemVizCategory.DOMAIN,
}


@dataclass
class SystemVizTag:
    """A SystemViz vocabulary tag applied to an entity."""
    element: str
    category: SystemVizCategory
    rationale: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'element': self.element,
            'category': self.category.value,
            'rationale': self.rationale,
            'confidence': self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemVizTag':
        return cls(
            element=data['element'],
            category=SystemVizCategory(data['category']),
            rationale=data.get('rationale', ''),
            confidence=data.get('confidence', 1.0),
        )


@dataclass
class SystemVizProfile:
    """Collection of SystemViz tags for a toolkit entity."""
    entity_type: str
    entity_id: str
    tags: List[SystemVizTag] = field(default_factory=list)

    def add(self, element: str, rationale: str = "", confidence: float = 1.0):
        """Add a tag, validating against known vocabulary."""
        if element not in SYSTEMVIZ_ELEMENTS:
            raise ValueError(f"Unknown SystemViz element: {element!r}")
        cat = SYSTEMVIZ_ELEMENTS[element]
        self.tags.append(SystemVizTag(element, cat, rationale, confidence))

    def categories_present(self) -> Set[SystemVizCategory]:
        return {t.category for t in self.tags}

    def by_category(self, category: SystemVizCategory) -> List[SystemVizTag]:
        return [t for t in self.tags if t.category == category]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'tags': [t.to_dict() for t in self.tags],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemVizProfile':
        profile = cls(entity_type=data['entity_type'], entity_id=data['entity_id'])
        for td in data.get('tags', []):
            profile.tags.append(SystemVizTag.from_dict(td))
        return profile


# --- Default tagging rules for toolkit entity types ---

def tag_image_schema(schema_name: str) -> SystemVizProfile:
    """Generate default SystemViz tags for an image schema."""
    profile = SystemVizProfile('image_schema', schema_name)
    SCHEMA_TAGS = {
        'PATH': [('conduit', 'PATH schema detects source-path-goal trajectory structure'),
                 ('trail', 'PATH metrics record displacement and curvature history')],
        'CYCLE': [('cycle', 'CYCLE schema detects periodic repetition'),
                  ('synchrony', 'CYCLE regularity measures temporal coordination')],
        'CONTACT': [('coupling', 'CONTACT schema measures ground interaction tightness'),
                    ('indicator', 'Contact state signals stance vs flight phase')],
        'BALANCE': [('equilibrium', 'BALANCE schema measures postural stability'),
                    ('feedback', 'COM height variance indicates balance feedback')],
        'FORCE': [('amplifier', 'FORCE schema tracks torque magnitude and asymmetry'),
                  ('enabler', 'Joint torque enables locomotion dynamics')],
    }
    for element, rationale in SCHEMA_TAGS.get(schema_name, []):
        profile.add(element, rationale)
    return profile


def tag_archetype(archetype_name: str, grounding_criteria=None, icm=None) -> SystemVizProfile:
    """Generate default SystemViz tags for an archetype."""
    profile = SystemVizProfile('archetype', archetype_name)
    profile.add('attractor', f'Archetype {archetype_name} is a behavioral attractor in weight space')
    profile.add('container', f'ICM defines the boundary conditions for {archetype_name}')
    if grounding_criteria:
        profile.add('indicator', 'Grounding criteria act as indicators validating archetype assignment')
    return profile


def tag_scenario(scenario_name: str, mode: str = '') -> SystemVizProfile:
    """Generate default SystemViz tags for a scenario."""
    profile = SystemVizProfile('scenario', scenario_name)
    profile.add('condition_shock', f'Scenario {scenario_name} is an environmental perturbation')
    if mode == 'shift':
        profile.add('wave', 'Shift-mode scenarios apply incremental perturbation')
    elif mode == 'set':
        profile.add('transition', 'Set-mode scenarios force a state transition')
    return profile


def tag_lattice_node(node_id: str, is_boundary: bool = False,
                     is_prototype: bool = False) -> SystemVizProfile:
    """Generate SystemViz tags for a lattice node based on its structural role."""
    profile = SystemVizProfile('lattice_node', node_id)
    profile.add('topology', 'Node exists in behavioral feature space lattice')
    if is_boundary:
        profile.add('edge', 'Node lies on boundary between clusters')
        profile.add('liminality', 'Boundary node occupies liminal space between behavioral families')
    if is_prototype:
        profile.add('centrality', 'Prototype node is central exemplar of its cluster')
        profile.add('attractor', 'Prototype acts as category attractor per Lakoff/Rosch')
    return profile


def tag_wolfram_class(wolfram_class: int) -> SystemVizProfile:
    """Generate SystemViz tags for a Wolfram CA classification result."""
    profile = SystemVizProfile('wolfram_class', f'class_{wolfram_class}')
    TAG_MAP = {
        1: [('equilibrium', 'Class 1: system converges to fixed point'),
            ('dampener', 'All dynamics are damped')],
        2: [('cycle', 'Class 2: system enters periodic orbit'),
            ('capacity', 'Bounded oscillation within finite state space')],
        3: [('entropy', 'Class 3: chaotic dynamics, high disorder'),
            ('noise', 'Label evolution appears random'),
            ('diffusion', 'Information spreads without structure')],
        4: [('emergent_property', 'Class 4: complex interacting structures emerge'),
            ('self_similarity', 'Localized structures show self-similar patterns'),
            ('criticality', 'System operates at edge of chaos')],
    }
    for element, rationale in TAG_MAP.get(wolfram_class, []):
        profile.add(element, rationale)
    return profile
