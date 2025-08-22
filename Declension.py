from enum import Enum, auto

class Gender(Enum):
  """Represents grammatical gender for Czech nouns."""
  MALE = auto()
  FEMALE = auto()
  NEUTRAL = auto() #Neviem ako sa to preklada

# --- Step 2: Refactor the functions to use the Enum ---

# def declension_gender_a(word: str, gender: Gender) -> str:
#   """
#   Applies the '-a' declension for feminine nouns.
#   Defaults to the base word for masculine nouns.
#   Example: mladý -> mladá
#   """
#   if gender is Gender.FEMALE:
#     return word + 'a'
#   return word

def declension_gender_ya(word: str, gender: Gender) -> str:
  """
  Applies the '-ý' vs '-á' adjectival ending based on gender.
  Defaults to the masculine '-ý'.
  Example: hezk -> hezký/hezká
  """
  # Specifically check if the gender is FEMALE.
  if gender is Gender.FEMALE:
    return word + 'á'

  # Default to the masculine form.
  return word + 'ý'

def declension_negation_ne(word: str) -> str:
  """
  Applies the 'ne-' negation prefix.
  """
  # This function does not depend on gender, so it remains the same.
  return 'ne' + word.lower()