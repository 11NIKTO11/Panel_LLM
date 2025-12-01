from utils import czech_grammar as cz

def process_employment(row, skip: str = 'Ne'):
    # Column names for employment statuses contain 'EMPLOYMENT_' and number from range from 1 to 10.
    employment_cols = [col for col in row.index if 'EMPLOYMENT_' in col]

    # Filter out 'Ne' (No) responses and collect the others.
    employment_statuses = [row[col] for col in employment_cols if row.get(col) != skip]

    return ', '.join(employment_statuses).lower()

def is_non_substantive_responses(response):
    lower_response = str(response).lower()
    return 'nevím' in lower_response or 'nechci' in lower_response

def process_income(row):
    income_raw = row.get('INCOMEP', '')

    if not income_raw or is_non_substantive_responses(income_raw):
        return None

    return cz.decapitalize(income_raw)

def process_town_size(row):
    city_size_raw = row.get('VMB', '')

    if 'Méně než 1.000' in city_size_raw:
        city_size_raw += ' obyvatel'

    return cz.decapitalize(city_size_raw)

def process_respondent(row, eu=False, nato=False, covid=False):
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

    if eu:
        processed_data['opinion_on_eu'] = row.get('Q18', '').lower()
    if nato:
        processed_data['opinion_on_nato'] = row.get('Q17', '').lower()
    if covid:
        processed_data['opinion_on_covid'] = row.get('Q23', '').lower()

    return processed_data

def gender_to_enum_gender(gender: str):
    g = (gender or '').lower()
    if g == 'žena':
        return cz.Gender.FEMALE
    elif g == 'muž':
        return cz.Gender.MALE
    else:
        return cz.Gender.NEUTRAL

def decline_region_to_locative(region_name):
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

def _apply_negation_if(verb: str, negate: bool) -> str:
    return cz.declension_negation_ne(verb) if negate else verb

def _format_opinion_statement(gender, opinion, topic_string):
    if is_non_substantive_responses(opinion):
        return ""  # Return an empty string if there is no opinion

    return f"Jsem {cz.declension_gender_ya(opinion[:-3],gender).lower()}, že je Česká republika členským státem {topic_string}."

def create_respondent_description(respondent):
    """
    Generates a descriptive Czech paragraph about a survey respondent
    by combining their answers into grammatically correct sentences.
    """
    gender = gender_to_enum_gender(respondent['gender'])
    description_parts = []

    # Basic Demographics
    description_parts.append(f"Jsem {respondent['gender']}, je mi {respondent['age']} let, mé vzdělání je {respondent['education_level']}.")

    # Location
    description_parts.append(f"Žiji v {decline_region_to_locative(respondent['region'])}, v okresu {respondent['district']} a obci o velikosti {respondent['town_size']}.")

    # Socioeconomic Status
    description_parts.append(f"Z hlediska zaměstnání jsem {respondent['employment_status']}")
    if respondent['income_range']:
        description_parts.append(f"a příjem naší domácnosti je {respondent['income_range']}")
    description_parts[-1] += "."

    # Living Standard
    living_standard = respondent['living_standard']
    if not is_non_substantive_responses(living_standard):
        verb = _apply_negation_if('mám', "ani" in respondent['living_standard'])
        description_parts.append(f"{verb.capitalize()} {living_standard} životní úroveň.")

    # Interest in Politics
    interest = respondent['interest_in_politics']
    if not is_non_substantive_responses(interest):
        description_parts.append(f"{interest.capitalize()} o politiku.")

    # EU and NATO opinions
    if 'opinion_on_eu' in respondent:
        description_parts.append(_format_opinion_statement(gender, respondent['opinion_on_eu'], "EU"))
    if 'opinion_on_nato' in respondent:
        description_parts.append(_format_opinion_statement(gender, respondent['opinion_on_nato'], "NATO"))

    # COVID Vaccination Status
    if 'covid_vaccinated' in respondent:
        vacc_status = respondent['covid_vaccinated']
        if not is_non_substantive_responses(vacc_status):
            verb = _apply_negation_if("jsem", cz.parse_yes_no(vacc_status))
            description_parts.append(f"{verb.capitalize()} {cz.declension_gender_ya('očkován',gender)} proti covidu.")

    full_description = " ".join(part for part in description_parts if part)
    return full_description
