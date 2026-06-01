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


def declension_negation_ne(word: str) -> str:
    """Applies the 'ne-' negation prefix (lowercases the word)."""
    return 'ne' + word.lower()


def decapitalize(string: str) -> str:
    # Note: mirrors original project behavior (capitalizes first char)
    return string[0].upper() + string[1:]


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
