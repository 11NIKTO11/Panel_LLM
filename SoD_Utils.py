from collections import Counter

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


def _apply_negation_if(verb: str, negate: bool) -> str:
    return Text_Utils.declension_negation_ne(verb) if negate else verb

def _format_opinion_statement(gender, opinion, topic_string):
    if opinion == is_non_substantive_responses(opinion):
        return "" # Return an empty string if there is no opinion

    return f"Jsem {Text_Utils.declension_gender_ya(opinion[:-3],gender).lower()}, že je Česká republika členským státem {topic_string}."
# --- Main Function to Create the Description ---

def create_respondent_description(respondent):
    """
    Generates a descriptive Czech paragraph about a survey respondent
    by combining their answers into grammatically correct sentences.

    Args:
        respondent (dict): A dictionary containing the processed data for one person,
                           with English keys (e.g., 'gender', 'region').

    Returns:
        str: A multi-sentence description of the respondent in Czech.
    """
    # --- 1. Build the description sentence by sentence ---
    # Using a list of sentences is cleaner than repeated string concatenation.
    gender = gender_to_enum_gender(respondent['gender'])
    description_parts = []

    # --- Basic Demographics ---
    description_parts.append(f"Jsem {respondent['gender']}, je mi {respondent['age']} let, mé vzdělání je {respondent['education_level']}.")

    # --- Location ---
    # This now uses the helper function for complex Czech grammar.
    description_parts.append(f"Žiji v {decline_region_to_Locative(respondent['region'])}, v okresu {respondent['district']} a obci o velikosti {respondent['town_size']}.")

    # --- Socioeconomic Status ---
    description_parts.append(f"Z hlediska zaměstnání jsem {respondent['employment_status']}")
    if respondent['income_range']:
        description_parts.append(f"a příjem naší domácnosti je {respondent['income_range']}")
    description_parts[-1]+="."

    # Living Standard
    living_standard = respondent['living_standard']
    if is_non_substantive_responses(living_standard):
        verb = _apply_negation_if('mám', "ani" in respondent['living_standard'])
        description_parts.append(f"{verb.capitalize()} {living_standard} životní úroveň.")

    # Interest in Politics
    interest = respondent['interest_in_politics']
    if is_non_substantive_responses(interest):
        description_parts.append(f"{interest.capitalize()} o politiku.")

    # EU and NATO opinions now use the dedicated helper function
    if respondent.__contains__('opinion_on_eu'):
        description_parts.append(_format_opinion_statement(gender, respondent['opinion_on_eu'], "EU"))
    if respondent.__contains__('opinion_on_nato'):
        description_parts.append(_format_opinion_statement(gender, respondent['opinion_on_nato'], "NATO"))

    # COVID Vaccination Status
    if respondent.__contains__('covid_vaccinated'):
        vacc_status = respondent['covid_vaccinated']
        if is_non_substantive_responses(vacc_status):
            verb = _apply_negation_if("jsem",Text_Utils.parse_yes_no(vacc_status))
            description_parts.append(f"{verb.capitalize()} {Text_Utils.declension_gender_ya('očkován',gender)} proti covidu.")

    # --- 2. Combine all parts into a final paragraph ---
    # Filter out any empty strings that may have been returned by helpers (e.g., for 'Nevím' answers)
    # and join the parts with a space.
    full_description = " ".join(part for part in description_parts if part)

    return full_description

def get_actual_results(respondents):
    votes = [vote for vote in respondents["voted_party"].to_list() if not is_non_substantive_responses(vote) and not "Nebyl" in vote]
    return {key: 100 * value / len(votes) for key, value in Counter(votes).items()}