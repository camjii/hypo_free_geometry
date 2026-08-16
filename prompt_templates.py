"""Shared prompt templates and prompt sets for activation extraction.

The legacy experiment scripts under ``workingcodebase/`` are frozen for
reproducibility.  New extraction code should use this module so prompt wording
is defined in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from types import MappingProxyType
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PromptTemplate:
    """A named single-value prompt template."""

    name: str
    text: str
    description: str

    def __post_init__(self) -> None:
        fields = [
            field_name
            for _, field_name, _, _ in Formatter().parse(self.text)
            if field_name is not None
        ]
        if fields != ["value"]:
            raise ValueError(
                f"prompt template {self.name!r} must contain exactly one "
                "{value} field"
            )

    def render(self, value: str) -> str:
        """Render one non-empty label without changing the template wording."""
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"prompt value for {self.name!r} cannot be empty")
        return self.text.format(value=normalized)

    def value_span(self, value: str) -> tuple[int, int]:
        """Return the ``[start, end)`` character span the value occupies.

        Templates may place ``{value}`` mid-sentence and may end in punctuation,
        so the concept is not always the final token.  Callers use this span to
        read activations at the concept itself rather than at the last token.
        """
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"prompt value for {self.name!r} cannot be empty")
        start = self.text.index("{value}")
        return start, start + len(normalized)


@dataclass(frozen=True)
class PromptSet:
    """A fixed ordered label set paired with one shared template."""

    name: str
    template_name: str
    values: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"prompt set {self.name!r} cannot be empty")
        if len(self.values) != len(set(self.values)):
            raise ValueError(f"prompt set {self.name!r} contains duplicate values")

    @property
    def prompts(self) -> tuple[str, ...]:
        """Render prompts in the same order as ``values``."""
        return render_prompts(self.template_name, self.values)


def _template(name: str, text: str, description: str) -> PromptTemplate:
    return PromptTemplate(name=name, text=text, description=description)


_PROMPT_TEMPLATES = {
    template.name: template
    for template in (
        _template("year", "In the year {value}", "Calendar years"),
        _template("quantity", "The quantity is {value}", "Numeric quantities"),
        _template(
            "planet",
            "The planet in the solar system is {value}",
            "Solar-system planets",
        ),
        _template("chess_piece", "The chess piece is the {value}", "Chess-piece roles"),
        _template("color", "The color of the object is {value}", "Colors"),
        _template(
            "emotion",
            "The emotion they are feeling is {value}",
            "Emotion words",
        ),
        _template(
            "musical_note",
            "The musical note being played is {value}",
            "Chromatic musical notes",
        ),
        _template(
            "musical_key",
            "The song is written in the key of {value} major",
            "Major keys",
        ),
        _template("hour", "The time of day is {value}", "Clock hours"),
        _template(
            "compass_direction",
            "The compass direction is {value}",
            "Compass directions",
        ),
        _template("season", "The season of the year is {value}", "Calendar seasons"),
        _template(
            "taxonomy_concept",
            "The concept being discussed is the {value}",
            "Taxonomic concepts",
        ),
        _template(
            "kinship",
            "Their family relation is the {value}",
            "Family relations",
        ),
        _template("city", "The location of the city {value}", "City names"),
        _template(
            "amino_acid",
            "The amino acid in the protein is {value}",
            "Amino acids",
        ),
        _template(
            "political_person",
            "The political views of {value}",
            "Political figures",
        ),
        _template(
            "vowel_sound",
            'The vowel sound is pronounced "{value}"',
            "Vowel sounds",
        ),
        _template(
            "movement_direction",
            "The direction of movement is {value}",
            "Three-dimensional directions",
        ),
        _template(
            "chemical_element",
            "The chemical element is {value}",
            "Chemical elements",
        ),
        _template(
            "day_of_week",
            "The day of the week is {value}",
            "Days of the week",
        ),
        _template(
            "month_of_year",
            "The month of the year is {value}",
            "Months of the year",
        ),
        _template(
            "chess_square",
            "The chess board square is {value}",
            "Chess-board coordinates",
        ),
        _template(
            "calendar_date",
            "The calendar date is {value}",
            "Month-and-day calendar dates",
        ),
    )
}

_VARIANT_TEXTS: Mapping[str, tuple[str, ...]] = {
    "year": (
        "The event occurred in the year {value}.",
        "This took place during {value}.",
        "The year being referenced is {value}.",
        "The date belongs to the year {value}.",
        "The historical year is {value}.",
        "The specified year is {value}.",
        "We are referring to the year {value}.",
        "The relevant year is {value}.",
        "The year associated with this event is {value}.",
        "The calendar year is {value}.",
    ),
    "quantity": (
        "The quantity is {value}.",
        "There are {value} items.",
        "The total number is {value}.",
        "The amount present is {value}.",
        "The count is {value}.",
        "A total of {value} are present.",
        "The number of objects is {value}.",
        "The measured quantity is {value}.",
        "The specified amount is {value}.",
        "The collection contains {value} items.",
    ),
    "planet": (
        "The planet is {value}.",
        "The planet in the solar system is {value}.",
        "The celestial planet being discussed is {value}.",
        "The world being referenced is {value}.",
        "The solar-system planet is {value}.",
        "The named planet is {value}.",
        "The astronomical object is the planet {value}.",
        "The planet being described is {value}.",
        "The planetary body is {value}.",
        "The referenced planet is {value}.",
    ),
    "chess_piece": (
        "The chess piece is the {value}.",
        "The piece on the chessboard is the {value}.",
        "The player moves the {value}.",
        "The chessman being referenced is the {value}.",
        "The selected chess piece is the {value}.",
        "The piece in play is the {value}.",
        "The board piece is the {value}.",
        "The piece being discussed is the {value}.",
        "The chess move uses the {value}.",
        "The relevant piece is the {value}.",
    ),
    "color": (
        "The color is {value}.",
        "The object is colored {value}.",
        "Its visible color is {value}.",
        "The shade being described is {value}.",
        "The object's appearance is {value} in color.",
        "The specified color is {value}.",
        "The surface has a {value} color.",
        "The color being referenced is {value}.",
        "It appears {value}.",
        "The object's hue is {value}.",
    ),
    "emotion": (
        "The emotion being felt is {value}.",
        "They are feeling {value}.",
        "Their emotional state is {value}.",
        "The person experiences {value}.",
        "The feeling being expressed is {value}.",
        "Their current emotion is {value}.",
        "The emotional response is {value}.",
        "The person feels a sense of {value}.",
        "The expressed emotion is {value}.",
        "The feeling described is {value}.",
    ),
    "musical_note": (
        "The musical note is {value}.",
        "The note being played is {value}.",
        "The musician plays the note {value}.",
        "The sounded pitch is the note {value}.",
        "The written note is {value}.",
        "The note heard is {value}.",
        "The musical pitch being referenced is {value}.",
        "The performer sounds the note {value}.",
        "The specified note is {value}.",
        "The melody contains the note {value}.",
    ),
    "musical_key": (
        "The song is written in {value} major.",
        "The musical key is {value} major.",
        "The piece is in the key of {value} major.",
        "The composition uses {value} major.",
        "The tonal key is {value} major.",
        "The music is composed in {value} major.",
        "The key signature corresponds to {value} major.",
        "The track is set in {value} major.",
        "The composition's major key is {value}.",
        "The song's tonality is {value} major.",
    ),
    "hour": (
        "The hour is {value}.",
        "The time shown is {value}.",
        "The clock indicates {value}.",
        "The current hour is {value}.",
        "The specified time of day is {value}.",
        "The clock reads {value}.",
        "The time being referenced is {value}.",
        "It is currently {value}.",
        "The hour on the clock is {value}.",
        "The stated time is {value}.",
    ),
    "compass_direction": (
        "The compass direction is {value}.",
        "The direction indicated by the compass is {value}.",
        "The heading is {value}.",
        "The compass points toward {value}.",
        "The geographic direction is {value}.",
        "The indicated bearing is {value}.",
        "The direction of orientation is {value}.",
        "The referenced compass heading is {value}.",
        "The directional heading is {value}.",
        "The specified direction on the compass is {value}.",
    ),
    "season": (
        "The season is {value}.",
        "The season of the year is {value}.",
        "The current season is {value}.",
        "The time of year is {value}.",
        "The seasonal period is {value}.",
        "The referenced season is {value}.",
        "This occurs during {value}.",
        "The yearly season being described is {value}.",
        "The specified season is {value}.",
        "The part of the seasonal cycle is {value}.",
    ),
    "taxonomy_concept": (
        "The taxonomy concept is the {value}.",
        "The concept being discussed is the {value}.",
        "The taxonomic category is the {value}.",
        "The classification concept is the {value}.",
        "The biological taxonomy term is {value}.",
        "The referenced taxonomic concept is {value}.",
        "The classification level being discussed is {value}.",
        "The taxonomy term being referenced is {value}.",
        "The relevant biological classification is {value}.",
        "The concept within taxonomy is {value}.",
    ),
    "kinship": (
        "Their family relation is the {value}.",
        "The kinship relation is {value}.",
        "This person is their {value}.",
        "Their relative is the {value}.",
        "The familial relationship is {value}.",
        "The family member being referenced is the {value}.",
        "Their relation within the family is {value}.",
        "The specified kinship role is {value}.",
        "The person is related as their {value}.",
        "The family connection is {value}.",
    ),
    "city": (
        "The city is {value}.",
        "The location being referenced is the city of {value}.",
        "The urban location is {value}.",
        "The named city is {value}.",
        "The place being discussed is {value}.",
        "The city being described is {value}.",
        "The referenced urban center is {value}.",
        "The geographic location is the city {value}.",
        "The specified city is {value}.",
        "The location corresponds to {value}.",
    ),
    "amino_acid": (
        "The amino acid is {value}.",
        "The protein contains the amino acid {value}.",
        "The amino acid residue is {value}.",
        "The referenced amino acid is {value}.",
        "The protein sequence includes {value}.",
        "The specified residue is {value}.",
        "The amino acid being discussed is {value}.",
        "The molecular residue is {value}.",
        "The protein's amino acid at this position is {value}.",
        "The biological molecule being referenced is the amino acid {value}.",
    ),
    "political_person": (
        "The political views of {value} are being discussed.",
        "The politician being referenced is {value}.",
        "The political figure is {value}.",
        "The person whose political positions are considered is {value}.",
        "The public figure in this political context is {value}.",
        "The political positions belong to {value}.",
        "The individual associated with these political views is {value}.",
        "The political person being discussed is {value}.",
        "The referenced political figure is {value}.",
        "The person whose politics are being examined is {value}.",
    ),
    "vowel_sound": (
        'The vowel sound is pronounced "{value}".',
        'The vowel pronunciation is "{value}".',
        'The spoken vowel sound is "{value}".',
        'The vowel produces the sound "{value}".',
        'The phonetic vowel sound is "{value}".',
        'The vowel is articulated as "{value}".',
        'The pronunciation being referenced is "{value}".',
        'The vowel is heard as "{value}".',
        'The specified vowel sound is "{value}".',
        'The vocalic sound is pronounced "{value}".',
    ),
    "movement_direction": (
        "The direction of movement is {value}.",
        "The object moves {value}.",
        "The motion proceeds {value}.",
        "The movement is directed {value}.",
        "The object travels {value}.",
        "The indicated movement direction is {value}.",
        "The motion heads {value}.",
        "The direction of travel is {value}.",
        "The object is moving {value}.",
        "The movement proceeds in the {value} direction.",
    ),
    "chemical_element": (
        "The chemical element is {value}.",
        "The element being referenced is {value}.",
        "The substance contains the element {value}.",
        "The atomic element is {value}.",
        "The specified chemical element is {value}.",
        "The element in question is {value}.",
        "The referenced element on the periodic table is {value}.",
        "The chemical species being discussed is the element {value}.",
        "The elemental substance is {value}.",
        "The periodic-table element is {value}.",
    ),
    "day_of_week": (
        "The day of the week is {value}.",
        "The specified weekday is {value}.",
        "The day being referenced is {value}.",
        "It occurs on {value}.",
        "The weekday is {value}.",
        "The calendar day of the week is {value}.",
        "The relevant day is {value}.",
        "The event falls on {value}.",
        "The named day of the week is {value}.",
        "The scheduled day is {value}.",
    ),
    "month_of_year": (
        "The month is {value}.",
        "The month of the year is {value}.",
        "The specified calendar month is {value}.",
        "The event occurs in {value}.",
        "The referenced month is {value}.",
        "The calendar period is {value}.",
        "The month being discussed is {value}.",
        "The named month is {value}.",
        "The relevant month of the year is {value}.",
        "The date falls within {value}.",
    ),
    "chess_square": (
        "The chessboard square is {value}.",
        "The piece is located on square {value}.",
        "The referenced chess square is {value}.",
        "The board coordinate is {value}.",
        "The position on the chessboard is {value}.",
        "The selected square is {value}.",
        "The chess coordinate being discussed is {value}.",
        "The piece occupies {value}.",
        "The specified board square is {value}.",
        "The location on the chessboard is {value}.",
    ),
    "calendar_date": (
        "The calendar date is {value}.",
        "The specified date is {value}.",
        "The event occurs on {value}.",
        "The date being referenced is {value}.",
        "The calendar shows {value}.",
        "The relevant date is {value}.",
        "The event is dated {value}.",
        "The stated calendar day is {value}.",
        "The recorded date is {value}.",
        "The date on the calendar is {value}.",
    ),
}


def _build_variants() -> dict[str, tuple[PromptTemplate, ...]]:
    """Register ``<family>_v<n>`` paraphrases alongside each canonical template.

    Variants exist to test whether a recovered geometry is a property of the
    concept or an artifact of one particular sentence.  They are additive: the
    canonical templates above keep their locked wording.
    """
    built: dict[str, tuple[PromptTemplate, ...]] = {}
    for family, texts in _VARIANT_TEXTS.items():
        if family not in _PROMPT_TEMPLATES:
            raise ValueError(f"variant family {family!r} has no canonical template")
        if len(texts) != len(set(texts)):
            raise ValueError(f"variant family {family!r} contains duplicate wording")
        base = _PROMPT_TEMPLATES[family]
        variants = []
        for number, text in enumerate(texts, start=1):
            variant = PromptTemplate(
                name=f"{family}_v{number}",
                text=text,
                description=f"{base.description} (paraphrase {number})",
            )
            _PROMPT_TEMPLATES[variant.name] = variant
            variants.append(variant)
        built[family] = tuple(variants)
    return built


_TEMPLATE_VARIANTS = _build_variants()

TEMPLATE_VARIANTS: Mapping[str, tuple[PromptTemplate, ...]] = MappingProxyType(
    _TEMPLATE_VARIANTS
)

PROMPT_TEMPLATES: Mapping[str, PromptTemplate] = MappingProxyType(_PROMPT_TEMPLATES)


DAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
CHESS_SQUARES = tuple(
    f"{file_name}{rank}" for file_name in "ABCDEFGH" for rank in range(1, 9)
)

_PROMPT_SETS = {
    prompt_set.name: prompt_set
    for prompt_set in (
        PromptSet(
            name="days",
            template_name="day_of_week",
            values=DAYS,
            description="Seven weekdays in calendar order",
        ),
        PromptSet(
            name="months",
            template_name="month_of_year",
            values=MONTHS,
            description="Twelve months in calendar order",
        ),
        PromptSet(
            name="chess_squares",
            template_name="chess_square",
            values=CHESS_SQUARES,
            description="All 64 chess squares in file-major order",
        ),
    )
}

PROMPT_SETS: Mapping[str, PromptSet] = MappingProxyType(_PROMPT_SETS)


def get_prompt_template(name: str) -> PromptTemplate:
    """Return a shared template with an actionable error for unknown names."""
    try:
        return PROMPT_TEMPLATES[name]
    except KeyError as error:
        choices = ", ".join(PROMPT_TEMPLATES)
        raise KeyError(
            f"unknown prompt template {name!r}; choose from: {choices}"
        ) from error


def get_prompt_set(name: str) -> PromptSet:
    """Return a built-in prompt set with an actionable error for unknown names."""
    try:
        return PROMPT_SETS[name]
    except KeyError as error:
        choices = ", ".join(PROMPT_SETS)
        raise KeyError(
            f"unknown prompt set {name!r}; choose from: {choices}"
        ) from error


def get_template_variants(family: str) -> tuple[PromptTemplate, ...]:
    """Return the paraphrase variants registered for a canonical template."""
    try:
        return TEMPLATE_VARIANTS[family]
    except KeyError as error:
        choices = ", ".join(TEMPLATE_VARIANTS)
        raise KeyError(
            f"no variants for template {family!r}; choose from: {choices}"
        ) from error


def render_prompts(template_name: str, values: Iterable[str]) -> tuple[str, ...]:
    """Render an ordered sequence of values using a shared template."""
    template = get_prompt_template(template_name)
    prompts = tuple(template.render(value) for value in values)
    if not prompts:
        raise ValueError("at least one prompt value is required")
    return prompts
