import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

from utils import constants, czech_grammar as cz

REGION_FORMS = Path(__file__).parent / "forms" / "regions.json"


@lru_cache(maxsize=1)
def _region_forms() -> dict:
    """The region case forms, read once on first use (never at import)."""
    return json.loads(REGION_FORMS.read_text(encoding="utf-8"))["forms"]


def region_in(region_name: str) -> str:
    """
    The region as a prepositional phrase: "ve Středočeském kraji", "v Praze".

    The preposition is part of the form because it vocalises with what follows, which is
    what the rules this replaced could not see.

    Raises:
        KeyError: if the region is not in datasets/sod/forms/regions.json.
    """
    return _region_forms()[region_name]["in"]


def region_of(region_name: str) -> str:
    """
    The region in the genitive: "Středočeského kraje", "Prahy".

    Raises:
        KeyError: if the region is not in datasets/sod/forms/regions.json.
    """
    return _region_forms()[region_name]["of"]

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
    """
    Map one raw `.sav` row to the clean_data schema.

    `id` is the survey's own respondent identifier, kept so descriptions and results join on
    identity rather than on row position. `age` is the exact age (AGE1) and `age_range` the
    survey's own band (AGE2, e.g. "25-34 let"); a describer prefers the exact one.
    """
    processed_data = {
        'id': int(row.get('id')),
        'gender': row.get('GENDER').lower(),
        'age': int(row.get('AGE1', 0)),
        'age_range': row.get('AGE2'),
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

def _apply_negation_if(verb: str, negate: bool) -> str:
    return cz.declension_negation_ne(verb) if negate else verb

def _format_opinion_statement(gender, opinion, topic_string):
    """
    NOTE: broken for the vocabulary we have seen. The `[:-3]` slice assumes the answer is an
    adjective stem plus a three-letter ending, so "spíše souhlasím" becomes "spíše souhlaá"
    (see tests/descriptor_test.ipynb section 6). No dataset in the repo carries
    `opinion_on_eu` / `opinion_on_nato`, so the real level vocabulary is unknown and fixing
    it would be guesswork. If those columns arrive, the natural Czech is the verb form
    itself, which needs no gender: "Spíše souhlasím s tím, že je Česká republika členským
    státem EU."
    """
    if is_non_substantive_responses(opinion):
        return ""  # Return an empty string if there is no opinion

    return f"Jsem {cz.declension_gender_ya(opinion[:-3],gender).lower()}, že je Česká republika členským státem {topic_string}."

def create_respondent_description(respondent):
    """
    Generates a descriptive Czech paragraph about a survey respondent
    by combining their answers into grammatically correct sentences.

    Survey-specific by design: it splices the survey's own answer strings straight into the
    sentences, which is the only reason they come out grammatical - "velmi se zajímám"
    becomes "Velmi se zajímám o politiku.". A synthetic grid profile carries a different set
    of columns and has its own describer in `datasets/grid/describe.py`; the
    two share their wording, deliberately, but not their code.
    """
    gender = gender_to_enum_gender(respondent['gender'])
    description_parts = []

    # Basic Demographics
    description_parts.append(f"Jsem {respondent['gender']}, je mi {respondent['age']} let, mé vzdělání je {respondent['education_level']}.")

    # Location
    description_parts.append(f"Žiji {region_in(respondent['region'])}, v okresu {respondent['district']} a obci o velikosti {respondent['town_size']}.")

    # Socioeconomic Status
    # Both answers are missing for some respondents - 9 have no employment status, 347 no
    # income. A bare `if value:` does not catch that: a float NaN is truthy, which is how
    # "a příjem naší domácnosti je nan." ended up in 347 of the legacy prompts.
    employment = respondent['employment_status'] if pd.notna(respondent['employment_status']) else None
    income = respondent['income_range'] if pd.notna(respondent['income_range']) else None
    if employment is not None and income is not None:
        description_parts.append(
            f"Z hlediska zaměstnání jsem {employment} a příjem naší domácnosti je {income}.")
    elif employment is not None:
        description_parts.append(f"Z hlediska zaměstnání jsem {employment}.")
    elif income is not None:
        description_parts.append(f"Příjem naší domácnosti je {income}.")

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
    if 'covid_vaccinated' in respondent and pd.notna(respondent['covid_vaccinated']):
        vaccinated = cz.parse_yes_no(respondent['covid_vaccinated'])
        if vaccinated is not None:
            verb = _apply_negation_if("jsem", not vaccinated)
            description_parts.append(f"{verb.capitalize()} {cz.declension_gender_ya('očkovan', gender)} proti covidu.")

    full_description = " ".join(part for part in description_parts if part)
    return full_description

def get_actual_results(respondents: pd.DataFrame, topic, count_non_substantive_as_not_voted: bool = False) -> pd.Series:
    """
    Aggregate what respondents said they voted - the *claimed* reference series.

    Args:
        respondents: DataFrame with a 'voted_party' column of survey responses
        topic: `utils.topic.Topic` naming the option columns and the attendance pair
        count_non_substantive_as_not_voted: If True, count "Nevím"/"Nechci uvést" as not voted

    Returns:
        pd.Series: attendance (if the topic has it), then each option's share of the
            claimed votes, normalized to sum to 1
    """
    # Filter out totally non-substantive answers unless counting them as not voted in the denominator.
    votes_all = respondents["voted_party"].astype(str).tolist()
    votes_subst = [vote for vote in votes_all if not is_non_substantive_responses(vote)]
    party_votes = [vote for vote in votes_subst if "Nebyl" not in vote]

    denom = len(respondents) if count_non_substantive_as_not_voted else len(votes_subst)
    voted = (len(party_votes) / denom) if denom > 0 else 0.0

    # Count party votes
    party_counts = Counter(party_votes)
    total_party_votes = len(party_votes)

    # Build result series
    result_dict = {}
    if topic.turnout:
        attended, abstained = topic.turnout
        result_dict[attended] = voted
        result_dict[abstained] = 1 - voted

    # Add all options with their share (0 if nobody claimed it)
    for party in topic.options:
        if total_party_votes > 0:
            result_dict[party] = party_counts.get(party, 0) / total_party_votes
        else:
            result_dict[party] = 0.0

    return pd.Series(result_dict)