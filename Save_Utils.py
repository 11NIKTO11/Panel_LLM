import json
from typing import Dict, Any, Protocol

import pandas as pd

class SeriesConvertible(Protocol):
    def to_series(self) -> pd.Series: ...

def save_results_to_json(results_by_id: Dict[int, Any], filename: str):
    """
    Serializes a dict mapping respondent IDs to VotingResult objects and saves it to a JSON file.

    Args:
        results_by_id: Dict where keys are respondent IDs (int/str) and values are Pydantic VotingResult objects.
        filename: Output JSON path.
    """
    # Convert values to serializable dicts; stringify keys to ensure JSON object keys are strings
    serializable = {str(k): (v.model_dump() if hasattr(v, 'model_dump') else v) for k, v in results_by_id.items()}
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

    Returns:
        Dict[int, VotingResult]
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data_from_file = json.load(f)
            results_by_id: Dict[int, Any] = {}
            if isinstance(data_from_file, dict):
                for k, item in data_from_file.items():
                    results_by_id[int(k)] = class_type(**item)
            elif isinstance(data_from_file, list):
                # If list provided, fall back to enumerated keys as strings
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



def results_to_dataframe(results_by_id: Dict[int, SeriesConvertible]) -> pd.DataFrame:
    if not results_by_id:
        return pd.DataFrame()

    rows = []
    index = []
    for rid, val in results_by_id.items():
        index.append(int(rid))
        s = val.to_series()
        rows.append(s)

    df = pd.DataFrame(rows)
    df.index = index
    df = df.sort_index()

    return df
