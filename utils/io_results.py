import json
import os
from typing import Dict, Any, Protocol, Tuple, Union, Type

from utils.constants import PROMPT
from utils.llm import BaseLLMClient
from utils.voting import BaseModel
import pandas as pd


class SeriesConvertible(Protocol):
    def to_series(self) -> pd.Series: ...

ResultValue = Union[SeriesConvertible, Tuple[str, SeriesConvertible]]

def _serialize_result_value(value: ResultValue) -> Any:
    if isinstance(value, tuple) and len(value) == 2:
        prompt, result = value
        payload = result.model_dump() if hasattr(result, 'model_dump') else result
        return {"prompt": prompt, "result": payload}
    return value.model_dump() if hasattr(value, 'model_dump') else value

def create_parameter_description(client_type: Type[BaseLLMClient], model: str, results: Type[BaseModel], n: int, temp: Union[int, float]) -> str:
    return f"{client_type.__name__}_{model}_{results.__name__}_n={n}_t={round(float(temp),2)}"

def save_results_to_json(results_by_id: Dict[int, ResultValue], filename: str):
    """
    Serializes a dict mapping respondent IDs to VotingResult objects and saves it to a JSON file.
    """
    serializable = {str(k): _serialize_result_value(v) for k, v in results_by_id.items()}
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved {len(results_by_id)} results (with IDs) to {filename}")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")

def load_results_from_json(filename: str, class_type):
    """
    Loads voting results from a JSON file saved as a dict mapping IDs to objects,
    and returns a dict that preserves respondent IDs.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data_from_file = json.load(f)
            results_by_id: Dict[int, Any] = {}
            if isinstance(data_from_file, dict):
                for k, item in data_from_file.items():
                    results_by_id[int(k)] = class_type(**item)
            elif isinstance(data_from_file, list):
                for idx, item in enumerate(data_from_file):
                    results_by_id[str(idx)] = class_type(**item)
            else:
                print(f"Unsupported JSON structure in {filename}: {type(data_from_file)}")
                return {}
        print(f"Successfully loaded {len(results_by_id)} results (with IDs) from {filename}")
        return results_by_id
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {filename}.")
        return {}

def results_to_dataframe(results_by_id: Dict[int, tuple[ str, ResultValue]]) -> pd.DataFrame:
    if not results_by_id:
        return pd.DataFrame()

    rows = []
    index = []
    for rid, (prompt, result) in results_by_id.items():
        index.append(int(rid))
        s = result.to_series()
        s[PROMPT] = prompt
        rows.append(s)

    df = pd.DataFrame(rows)
    df.index = index
    df = df.sort_index()

    return df

def load_actual_results() -> pd.DataFrame:
    """
    Load actual election results from CSV.

    Returns:
        pd.DataFrame: DataFrame with columns [region, VOTED, NOT_VOTED, ...party columns...]
    """
    csv_path = os.path.join("data", "election_data.csv")
    return pd.read_csv(csv_path, encoding="utf-8-sig")
