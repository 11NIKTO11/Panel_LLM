"""
Map the survey's raw `.sav` rows to the `clean_data.csv` schema.

Survey-specific by design: it knows the raw column names (`AGE1`, `KRAJ`, `Q21`) and the
answer strings they carry, and changes when the survey's own columns do. The Czech phrases
built from those answers live in `describe.py`.
"""
from utils import czech_grammar as cz


def is_non_substantive_responses(response):
    lower_response = str(response).lower()
    return 'nevím' in lower_response or 'nechci' in lower_response

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

    return cz.decapitalize(income_raw)

def process_town_size(row):
    city_size_raw = row.get('VMB', '')

    if 'Méně než 1.000' in city_size_raw:
        city_size_raw += ' obyvatel'

    return cz.decapitalize(city_size_raw)

def process_respondent(row):
    """
    Map one raw `.sav` row to the clean_data schema.

    `id` is the survey's own respondent identifier, kept so descriptions and results join on
    identity rather than on row position. `age` is the exact age (AGE1) and `age_range` the
    survey's own band (AGE2, e.g. "25-34 let"); a describer prefers the exact one.

    Every answer the survey gives is emitted, including the three no v1 description reads.
    Which of them reach a prompt is the template's decision: a sentence whose column is
    absent is dropped, so the variant is the gate, not this function.
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
        # Q18 asks about EU membership, Q17 about NATO; Q23 is vaccination, not an opinion.
        'opinion_on_eu': row.get('Q18','').lower(),
        'opinion_on_nato': row.get('Q17','').lower(),
        'covid_vaccinated': row.get('Q23','').lower(),
    }

    return processed_data
