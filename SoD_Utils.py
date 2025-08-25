import Text_Utils

LANGUAGE_EN = 'EN'
LANGUAGE_CZ = 'CZ'

def get_dataset_path(language: str) -> str:
  return f'SoD-{language}/Society_of_Distrust - {language}.sav'

def process_employment(row, skip: str = 'Ne'):
    # Column names for employment statuses contain 'EMPLOYMENT_' and number from range from 1 to 10.
    employment_cols = [col for col in row.index if 'EMPLOYMENT_' in col]

    # Filter out 'Ne' (No) responses and collect the others.
    employment_statuses = [row[col] for col in employment_cols if row.get(col) != skip]

    return ', '.join(employment_statuses).lower()


def process_income(row):
  income_raw = row.get('INCOMEP', '')

  if not income_raw or is_non_substantive_responses(income_raw):
    return None

  return Text_Utils.decapitalize(income_raw)


def process_town_size(row):
  city_size_raw = row.get('VMB', '')

  if 'Méně než 1.000' in city_size_raw:
    city_size_raw += ' obyvatel'

  return Text_Utils.decapitalize(city_size_raw)

def gender_to_enum_gender(gender:str):
  if gender.lower() == 'žena':
    return Text_Utils.Gender.FEMALE
  else:
    return Text_Utils.Gender.MALE

def decline_region_to_Locative(region_name):
    """
    Applies Czech grammar rules to correctly decline a region name for use
    in a sentence like "Žiji v [region name]".

    For example: 'Plzeňský kraj' -> 'Plzeňském kraji'
    """
    # Handles special case for Prague 'Hlavní město Praha'
    if 'Praha' in region_name:
        return 'Praze'

    # General grammar rules for other regions
    declined_name = region_name.replace('raj', 'raji')  # Covers Kraj and kraj
    if 'ký' in declined_name:
        declined_name = declined_name.replace('ký', 'kém')

    return declined_name

def is_non_substantive_responses(response):
    lower_response = response.lower()
    return 'nevím' in lower_response or 'nechci' in lower_response
