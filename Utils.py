LANGUAGE_EN = 'EN'
LANGUAGE_CZ = 'CZ'

def get_SoD_dataset_path(language: str) -> str:
  return f'SoD-{language}/Society_of_Distrust - {language}.sav'

def decapitalize(string: str) -> str:
  return string[0].upper() + string[1:]