import json

def save_results_to_json(results: list, filename: str):
    """
    Serializes a list of VotingResult objects and saves them to a JSON file.

    Args:
        results: A list of VotingResult Pydantic objects.
        filename: The path to the file where the results will be saved.
    """
    # Convert each Pydantic model to a dictionary using .model_dump()
    results_as_dicts = [result.model_dump() for result in results]

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Use json.dump to write the list of dicts to the file
            # ensure_ascii=False correctly handles Czech characters
            # indent=4 makes the file human-readable
            json.dump(results_as_dicts, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved {len(results)} results to {filename}")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")

def load_results_from_json(filename: str, class_type):
    """
    Loads voting results from a JSON file and deserializes them into
    a list of VotingResult objects.

    Args:
        filename: The path to the JSON file to load.

    Returns:
        A list of VotingResult objects.
    """
    loaded_results = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Load the raw data from the JSON file
            data_from_file = json.load(f)

            # Re-create Pydantic models from the loaded dictionaries
            # This automatically validates the data against your schema
            for item in data_from_file:
                loaded_results.append(class_type(**item))

        print(f"Successfully loaded and validated {len(loaded_results)} results from {filename}")
        return loaded_results
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {filename}.")
        return []
