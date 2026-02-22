"""Tests for SystemViz Visual Vocabulary tagging."""

import pytest

from motion_analytics.core.systemviz import (
    SystemVizCategory, SystemVizTag, SystemVizProfile,
    SYSTEMVIZ_ELEMENTS,
    tag_image_schema, tag_archetype, tag_scenario,
    tag_lattice_node, tag_wolfram_class,
)


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestSystemVizConstants:
    def test_all_elements_have_valid_category(self):
        for element, category in SYSTEMVIZ_ELEMENTS.items():
            assert isinstance(category, SystemVizCategory), (
                f"{element} has invalid category type: {type(category)}"
            )

    def test_six_categories_present(self):
        categories = set(SYSTEMVIZ_ELEMENTS.values())
        assert len(categories) == 6
        assert categories == set(SystemVizCategory)

    def test_element_count_reasonable(self):
        assert len(SYSTEMVIZ_ELEMENTS) > 50


# ---------------------------------------------------------------------------
# SystemVizProfile tests
# ---------------------------------------------------------------------------

class TestSystemVizProfile:
    def test_add_valid_element(self):
        profile = SystemVizProfile('test', 'id_1')
        profile.add('attractor', 'test rationale', 0.9)
        assert len(profile.tags) == 1
        assert profile.tags[0].element == 'attractor'
        assert profile.tags[0].category == SystemVizCategory.DRIVER
        assert profile.tags[0].confidence == 0.9

    def test_add_unknown_element_raises(self):
        profile = SystemVizProfile('test', 'id_1')
        with pytest.raises(ValueError, match="Unknown SystemViz element"):
            profile.add('nonexistent_element')

    def test_categories_present(self):
        profile = SystemVizProfile('test', 'id_1')
        profile.add('attractor')  # DRIVER
        profile.add('feedback')   # SIGNAL
        profile.add('edge')       # BOUNDARY
        cats = profile.categories_present()
        assert cats == {SystemVizCategory.DRIVER, SystemVizCategory.SIGNAL, SystemVizCategory.BOUNDARY}

    def test_by_category(self):
        profile = SystemVizProfile('test', 'id_1')
        profile.add('attractor')  # DRIVER
        profile.add('cycle')      # DRIVER
        profile.add('feedback')   # SIGNAL
        drivers = profile.by_category(SystemVizCategory.DRIVER)
        assert len(drivers) == 2
        signals = profile.by_category(SystemVizCategory.SIGNAL)
        assert len(signals) == 1

    def test_serialization_roundtrip(self):
        profile = SystemVizProfile('archetype', 'fold')
        profile.add('attractor', 'behavioral attractor', 0.95)
        profile.add('container', 'ICM boundary')
        d = profile.to_dict()
        assert d['entity_type'] == 'archetype'
        assert d['entity_id'] == 'fold'
        assert len(d['tags']) == 2

        profile2 = SystemVizProfile.from_dict(d)
        assert profile2.entity_type == 'archetype'
        assert profile2.entity_id == 'fold'
        assert len(profile2.tags) == 2
        assert profile2.tags[0].element == 'attractor'
        assert profile2.tags[0].confidence == 0.95


# ---------------------------------------------------------------------------
# Default tagger tests
# ---------------------------------------------------------------------------

class TestDefaultTaggers:
    def test_tag_image_schema_path(self):
        profile = tag_image_schema('PATH')
        elements = [t.element for t in profile.tags]
        assert 'conduit' in elements
        assert 'trail' in elements
        assert profile.entity_type == 'image_schema'
        assert profile.entity_id == 'PATH'

    def test_tag_image_schema_unknown(self):
        profile = tag_image_schema('UNKNOWN')
        assert profile.entity_id == 'UNKNOWN'
        assert len(profile.tags) == 0

    def test_tag_archetype(self):
        profile = tag_archetype('deleuze_fold', grounding_criteria=['gc1'])
        elements = [t.element for t in profile.tags]
        assert 'attractor' in elements
        assert 'container' in elements
        assert 'indicator' in elements

    def test_tag_archetype_without_grounding(self):
        profile = tag_archetype('test_arch')
        elements = [t.element for t in profile.tags]
        assert 'attractor' in elements
        assert 'container' in elements
        assert 'indicator' not in elements

    def test_tag_scenario_shift(self):
        profile = tag_scenario('drift_test', mode='shift')
        elements = [t.element for t in profile.tags]
        assert 'condition_shock' in elements
        assert 'wave' in elements

    def test_tag_scenario_set(self):
        profile = tag_scenario('jump_test', mode='set')
        elements = [t.element for t in profile.tags]
        assert 'condition_shock' in elements
        assert 'transition' in elements

    def test_tag_lattice_node_boundary(self):
        profile = tag_lattice_node('node_42', is_boundary=True)
        elements = [t.element for t in profile.tags]
        assert 'topology' in elements
        assert 'edge' in elements
        assert 'liminality' in elements

    def test_tag_lattice_node_prototype(self):
        profile = tag_lattice_node('node_1', is_prototype=True)
        elements = [t.element for t in profile.tags]
        assert 'centrality' in elements
        assert 'attractor' in elements

    def test_tag_wolfram_class_4(self):
        profile = tag_wolfram_class(4)
        elements = [t.element for t in profile.tags]
        assert 'emergent_property' in elements
        assert 'criticality' in elements
        assert 'self_similarity' in elements

    def test_tag_wolfram_class_1(self):
        profile = tag_wolfram_class(1)
        elements = [t.element for t in profile.tags]
        assert 'equilibrium' in elements
        assert 'dampener' in elements
