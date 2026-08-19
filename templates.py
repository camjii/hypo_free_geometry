"""The five universal prompt templates used for every concept set.

Earlier extraction runs gave each concept family its own wording, so any
geometry difference between families was confounded with prompt wording and
there was no shared template effect for preprocessing to remove.  These five
are applied unchanged to all sets, which makes the prompt a constant of the
experiment rather than a per-family parameter.

One template is bare and four are framed.  The bare template is the control:
if a shape appears only under the framed templates, the framing produced it.

The model is a causal decoder, so the residual stream above the concept
attends only to earlier positions.  Only the text *preceding* the value can
change the extracted vector; whatever follows it is inert.  Two templates
sharing a prefix therefore yield byte-identical activations and a duplicated
point in the cloud, so :func:`_assert_distinct_prefixes` rejects that at import
rather than letting it reach persistent homology as a zero-distance pair.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from prompt_templates import PromptTemplate

UNIVERSAL_TEMPLATES: tuple[PromptTemplate, ...] = (
    PromptTemplate(
        name="bare",
        text="{value}",
        description="The concept alone, with no framing (control)",
    ),
    PromptTemplate(
        name="answer",
        text="The answer is {value}.",
        description="Short declarative frame",
    ),
    PromptTemplate(
        name="consider",
        text="Consider {value} carefully.",
        description="Imperative frame",
    ),
    PromptTemplate(
        name="referring",
        text="We are referring to {value}.",
        description="First-person frame",
    ),
    PromptTemplate(
        name="example",
        text="Here is an example: {value}",
        description="Enumerative frame, longest prefix",
    ),
)

N_TEMPLATES = len(UNIVERSAL_TEMPLATES)


def prefix(template: PromptTemplate) -> str:
    """Return the template text preceding ``{value}``.

    In a causal model this prefix determines the extracted activation on its
    own, so it is the only part of a template that carries experimental
    meaning.
    """
    return template.text[: template.text.index("{value}")]


def _assert_distinct_prefixes(templates: tuple[PromptTemplate, ...]) -> None:
    seen: dict[str, str] = {}
    for template in templates:
        text = prefix(template)
        if text in seen:
            raise ValueError(
                f"templates {seen[text]!r} and {template.name!r} share the "
                f"prefix {text!r}; in a causal model they produce identical "
                "activations and a coincident point in every concept cloud"
            )
        seen[text] = template.name


_assert_distinct_prefixes(UNIVERSAL_TEMPLATES)

TEMPLATES_BY_NAME: Mapping[str, PromptTemplate] = MappingProxyType(
    {template.name: template for template in UNIVERSAL_TEMPLATES}
)

TEMPLATE_ORDER: tuple[str, ...] = tuple(
    template.name for template in UNIVERSAL_TEMPLATES
)


def render(value: str) -> tuple[tuple[str, tuple[int, int]], ...]:
    """Render one value under all five templates, in ``TEMPLATE_ORDER``.

    Each entry pairs the prompt with the character span the value occupies, so
    the caller can read the activation at the concept rather than at whatever
    token happens to end the prompt.
    """
    return tuple(
        (template.render(value), template.value_span(value))
        for template in UNIVERSAL_TEMPLATES
    )
