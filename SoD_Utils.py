from collections import Counter

import Data_Utils
import Text_Utils
import pandas as pd

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

def process_SoD_response(row, eu=False, nato=False, covid=False):
    # --- 1. Basic Information ---
    # Renames Czech keys to English and performs initial data cleaning.
    processed_data = {
        'gender': row.get('GENDER').lower(),
        'age': int(row.get('AGE1', 0)),
        'education_level': row.get('EDU', '').lower(),
        'region': row.get('KRAJ'),
        'district': row.get('OKRES'),
        'town_size': process_town_size(row),
        'employment_status': process_employment(row),
        'income_range': process_income(row),
        'living_standard': row.get('Q19','').lower(),
        'interest_in_politics': row.get('Q20','').lower(),
        'voted_party': row.get('Q21'),
    }

    # --- 2. Additional Survey Questions ---
    # Merges the dictionary of additional questions into the main one.
    if eu:
        processed_data['opinion_on_eu'] = row.get('Q18', '').lower()
    if nato:
        processed_data['opinion_on_nato'] = row.get('Q17', '').lower()
    if covid:
        processed_data['opinion_on_covid'] = row.get('Q23', '').lower()

    return processed_data

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
    if 'Česko' in region_name:
        return 'Česku'

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
    if not is_non_substantive_responses(living_standard):
        verb = _apply_negation_if('mám', "ani" in respondent['living_standard'])
        description_parts.append(f"{verb.capitalize()} {living_standard} životní úroveň.")

    # Interest in Politics
    interest = respondent['interest_in_politics']
    if not is_non_substantive_responses(interest):
        description_parts.append(f"{interest.capitalize()} o politiku.")

    # EU and NATO opinions now use the dedicated helper function
    if respondent.__contains__('opinion_on_eu'):
        description_parts.append(_format_opinion_statement(gender, respondent['opinion_on_eu'], "EU"))
    if respondent.__contains__('opinion_on_nato'):
        description_parts.append(_format_opinion_statement(gender, respondent['opinion_on_nato'], "NATO"))

    # COVID Vaccination Status
    if respondent.__contains__('covid_vaccinated'):
        vacc_status = respondent['covid_vaccinated']
        if not is_non_substantive_responses(vacc_status):
            verb = _apply_negation_if("jsem",Text_Utils.parse_yes_no(vacc_status))
            description_parts.append(f"{verb.capitalize()} {Text_Utils.declension_gender_ya('očkován',gender)} proti covidu.")

    # --- 2. Combine all parts into a final paragraph ---
    # Filter out any empty strings that may have been returned by helpers (e.g., for 'Nevím' answers)
    # and join the parts with a space.
    full_description = " ".join(part for part in description_parts if part)

    return full_description

def get_actual_results(respondents:pd.DataFrame, count_non_substantive_as_not_voted=False) -> pd.Series:
    """
    Aggregate actual voting results from survey respondents into a Series.

    Args:
        respondents: DataFrame with 'voted_party' column containing survey responses
        count_non_substantive_as_not_voted: If True, count "Nevím"/"Nechci uvést" as "Not Voted"

    Returns:
        pd.Series: Aggregated results with attendance and party probabilities
            - VOTED: probability of voting
            - NOT_VOTED: probability of not voting
            - Party columns: vote share for each party (normalized to sum to 1)
    """
    votes = [vote for vote in respondents["voted_party"].to_list() if not is_non_substantive_responses(vote)]
    party_votes = [vote for vote in votes if not "Nebyl" in vote]
    voted = len(party_votes) / (len(respondents) if count_non_substantive_as_not_voted else len(votes))

    # Count party votes
    party_counts = Counter(party_votes)
    total_party_votes = len(party_votes)

    # Build result series
    result_dict = {
        Data_Utils.VOTED: voted,
        Data_Utils.NOT_VOTED: 1 - voted
    }

    # Add all parties with their probabilities (0 if not in data)
    for party in Data_Utils.PARTY_COLUMNS_2021:
        result_dict[party] = party_counts.get(party, 0) / total_party_votes if total_party_votes > 0 else 0.0

    return pd.Series(result_dict)