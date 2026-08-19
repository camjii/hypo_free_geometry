import json
from pathlib import Path

import pytest

from prompt_templates import PromptTemplate
from templates import (
    N_TEMPLATES,
    TEMPLATE_ORDER,
    UNIVERSAL_TEMPLATES,
    _assert_distinct_prefixes,
    prefix,
    render,
)

REPO = Path(__file__).resolve().parent


def test_the_spec_requires_one_bare_template_and_four_framed():
    assert N_TEMPLATES == 5
    bare = [t for t in UNIVERSAL_TEMPLATES if prefix(t) == ""]
    assert [t.name for t in bare] == ["bare"]


def test_prefixes_are_distinct_so_no_two_templates_collide():
    # A causal model reads the concept from its prefix alone, so templates
    # sharing one return identical vectors and a coincident point.
    prefixes = [prefix(template) for template in UNIVERSAL_TEMPLATES]
    assert len(set(prefixes)) == len(prefixes)


def test_a_colliding_prefix_is_rejected():
    leading = PromptTemplate(
        name="subject", text="{value} is the item.", description="starts with value"
    )
    with pytest.raises(ValueError, match="share the prefix"):
        _assert_distinct_prefixes((UNIVERSAL_TEMPLATES[0], leading))


def test_every_value_span_locates_the_concept():
    for value in ("Monday", "12586269025", "C#", "New York"):
        rendered = render(value)
        assert len(rendered) == N_TEMPLATES
        for prompt, (start, end) in rendered:
            assert prompt[start:end] == value


def test_render_follows_the_declared_template_order():
    names = [template.name for template in UNIVERSAL_TEMPLATES]
    assert names == list(TEMPLATE_ORDER)


def test_every_concept_set_renders_under_every_template():
    document = json.loads((REPO / "concept_sets.json").read_text())
    for name, entry in document["sets"].items():
        for value in entry["values"]:
            assert len(render(value)) == N_TEMPLATES, name


def test_expectations_cover_every_concept_set():
    concepts = json.loads((REPO / "concept_sets.json").read_text())["sets"]
    expectations = json.loads((REPO / "expectations_v2.json").read_text())
    assert set(concepts) == set(expectations["expectations"])
    for name, record in expectations["expectations"].items():
        assert record["n_values"] == concepts[name]["n_values"], name
        for group in ("H0", "H1", "H2"):
            low, high = record["expected_counts"][group]
            assert 0 <= low <= high, name


def test_no_concept_set_contains_repeated_values():
    concepts = json.loads((REPO / "concept_sets.json").read_text())["sets"]
    for name, entry in concepts.items():
        assert len(entry["values"]) == len(set(entry["values"])), name
