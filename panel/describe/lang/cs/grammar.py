from enum import Enum


class Gender(Enum):
    """Represents grammatical gender for Czech nouns."""
    MALE = "muž"
    FEMALE = "žena"
    NEUTRAL = "nevím"  # Placeholder for unknown/neutral

def declension_gender_ya(word: str, gender: Gender) -> str:
    """
    Applies the '-ý' vs '-á' adjectival ending based on gender.
    Defaults to the masculine '-ý'. Example: hezk -> hezký/hezká
    """
    if gender is Gender.FEMALE:
        return word + 'á'
    return word + 'ý'


YEAR_WORDS = ("let", "roku", "roků", "roky", "rok")


def _drop_year_word(text: str) -> str:
    """"44 let" -> "44"; a band's own unit is not repeated in the clause."""
    words = text.split()
    if words and words[-1].lower().rstrip(".") in YEAR_WORDS:
        words = words[:-1]
    return " ".join(words).strip()


def age_clause(age) -> str:
    """
    The Czech "je mi ..." clause for an exact age or an age band.

    Bands are written several ways across the data - the grid uses "30-44" and "60+", the
    survey's own range columns use "30 - 44 let" and "Více než 60 let" - so a trailing year
    word is dropped and a leading phrase is decapitalised to fit mid-sentence.

    Args:
        age: an exact age (28, "28"), a band ("30-44", "30 - 44 let"), an open band ("60+",
            "Více než 60 let") or any other phrase ("do 24").

    Returns:
        "je mi 28 let", "je mi mezi 30 a 44 lety", "je mi 60 a více let",
        "je mi více než 60 let".
    """
    try:
        number = float(age)
    except (TypeError, ValueError):
        number = None
    if number is not None:
        return f"je mi {int(number) if number.is_integer() else number} let"

    value = " ".join(str(age).split())
    if value.endswith("+"):
        return f"je mi {_drop_year_word(value[:-1])} a více let"
    for dash in ("–", "-"):
        if dash in value:
            low, high = (_drop_year_word(part) for part in value.split(dash, 1))
            if low and high:
                return f"je mi mezi {low} a {high} lety"
    return f"je mi {decapitalize(_drop_year_word(value))} let"


def declension_negation_ne(word: str) -> str:
    """Applies the 'ne-' negation prefix (lowercases the word)."""
    return 'ne' + word.lower()


def decapitalize(string: str) -> str:
    return string[0].lower() + string[1:]


NON_SUBSTANTIVE = ("nevím", "nechci")


def is_non_substantive(response) -> bool:
    """`nevím` / `nechci uvést` in any casing: an answer that says nothing about the person."""
    text = str(response).lower()
    return any(marker in text for marker in NON_SUBSTANTIVE)


def parse_yes_no(response: str) -> bool | None:
    normalized_response = response.lower().strip()

    affirmative_answers = {'ano'}
    negative_answers = {'ne'}

    if normalized_response in affirmative_answers:
        return True
    elif normalized_response in negative_answers:
        return False
    else:
        return None
