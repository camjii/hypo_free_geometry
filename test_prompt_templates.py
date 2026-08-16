import pytest

from prompt_templates import (
    CHESS_SQUARES,
    DAYS,
    MONTHS,
    PROMPT_SETS,
    PROMPT_TEMPLATES,
    TEMPLATE_VARIANTS,
    PromptTemplate,
    get_prompt_set,
    get_prompt_template,
    get_template_variants,
    render_prompts,
)


def test_shared_prompt_sets_render_the_locked_experiment_wording():
    assert get_prompt_set("days").prompts[0] == "The day of the week is Monday"
    assert get_prompt_set("months").prompts[-1] == ("The month of the year is December")
    assert get_prompt_set("chess_squares").prompts[:2] == (
        "The chess board square is A1",
        "The chess board square is A2",
    )


def test_builtin_prompt_sets_have_fixed_unique_labels():
    assert len(DAYS) == len(set(DAYS)) == 7
    assert len(MONTHS) == len(set(MONTHS)) == 12
    assert len(CHESS_SQUARES) == len(set(CHESS_SQUARES)) == 64
    assert CHESS_SQUARES[0] == "A1"
    assert CHESS_SQUARES[-1] == "H8"
    assert set(PROMPT_SETS) == {"days", "months", "chess_squares"}


def test_every_shared_template_has_one_value_field_and_renders_cleanly():
    assert len(PROMPT_TEMPLATES) >= 20
    for name, template in PROMPT_TEMPLATES.items():
        rendered = template.render("example")
        assert rendered.count("example") == 1, name
        assert rendered == rendered.strip(), name


def test_template_rejects_ambiguous_or_empty_input():
    with pytest.raises(ValueError, match="exactly one"):
        PromptTemplate("bad", "No field", "invalid")
    with pytest.raises(ValueError, match="exactly one"):
        PromptTemplate("bad", "{value} and {other}", "invalid")
    with pytest.raises(ValueError, match="cannot be empty"):
        get_prompt_template("month_of_year").render("   ")


def test_render_prompts_preserves_order_and_requires_values():
    assert render_prompts("year", ("1990", "2000")) == (
        "In the year 1990",
        "In the year 2000",
    )
    with pytest.raises(ValueError, match="at least one"):
        render_prompts("year", ())


def test_every_canonical_family_has_ten_distinct_paraphrases():
    assert len(TEMPLATE_VARIANTS) == 23
    for family, variants in TEMPLATE_VARIANTS.items():
        assert len(variants) == 10, family
        assert get_prompt_template(family).text not in {v.text for v in variants}
        for number, variant in enumerate(variants, start=1):
            assert variant.name == f"{family}_v{number}"
            assert variant is get_prompt_template(variant.name)


def test_variants_do_not_disturb_the_locked_canonical_wording():
    assert get_prompt_template("month_of_year").text == "The month of the year is {value}"
    assert get_prompt_template("month_of_year_v2").text == (
        "The month of the year is {value}."
    )


def test_value_span_locates_the_concept_even_when_it_is_not_final():
    template = get_prompt_template("musical_key_v1")
    rendered = template.render("C")
    start, end = template.value_span("C")
    assert rendered == "The song is written in C major."
    assert rendered[start:end] == "C"
    trailing = get_prompt_template("month_of_year_v2")
    start, end = trailing.value_span("January")
    assert trailing.render("January")[start:end] == "January"


def test_unknown_variant_family_reports_available_choices():
    with pytest.raises(KeyError, match="no variants for template"):
        get_template_variants("missing")


def test_unknown_registry_names_report_available_choices():
    with pytest.raises(KeyError, match="unknown prompt template"):
        get_prompt_template("missing")
    with pytest.raises(KeyError, match="unknown prompt set"):
        get_prompt_set("missing")
