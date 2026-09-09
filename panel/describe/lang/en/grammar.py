"""English grammar primitives: the age clause, yes/no, and what counts as a non-answer."""
from typing import Union

NON_SUBSTANTIVE = ("don't know", "prefer not to say", "don't want to say")
YEAR_WORDS = ("old", "years", "year")


def is_non_substantive(response) -> bool:
    """`don't know` / `prefer not to say`, in any casing: an answer that says nothing about the person."""
    text = str(response).lower()
    return any(marker in text for marker in NON_SUBSTANTIVE)


def decapitalize(text: str) -> str:
    return text[0].lower() + text[1:]


def _drop_year_words(text: str) -> str:
    """"44 years old" -> "44"; a band's own unit is not repeated in the clause."""
    words = text.split()
    while words and words[-1].lower().rstrip(".") in YEAR_WORDS:
        words = words[:-1]
    return " ".join(words).strip()


def age_clause(age) -> str:
    """
    The "aged ..." clause for an exact age or an age band.

    Args:
        age: an exact age (28, "28"), a band ("30-44", "25-34 years old"), an open band ("60+",
            "More than 60 years old") or any other phrase.

    Returns:
        "aged 28", "aged between 30 and 44", "aged 60 or older", "aged more than 60".
    """
    try:
        number = float(age)
    except (TypeError, ValueError):
        number = None
    if number is not None:
        return f"aged {int(number) if number.is_integer() else number}"

    value = " ".join(str(age).split())
    if value.endswith("+"):
        return f"aged {_drop_year_words(value[:-1])} or older"
    for dash in ("–", "-"):
        if dash in value:
            low, high = (_drop_year_words(part) for part in value.split(dash, 1))
            if low and high:
                return f"aged between {low} and {high}"
    return f"aged {decapitalize(_drop_year_words(value))}"


def parse_yes_no(response: str) -> Union[bool, None]:
    """True for "yes", False for "no", None for anything else."""
    normalized = response.lower().strip()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    return None
