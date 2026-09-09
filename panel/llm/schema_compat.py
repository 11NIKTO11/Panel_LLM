"""Per-provider fixups of Pydantic-generated JSON schemas for strict structured outputs."""


def set_additional_properties_false(schema: dict) -> dict:
    """
    Recursively traverses a JSON schema and sets 'additionalProperties'
    to False for all objects.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
        for key, value in schema.items():
            schema[key] = set_additional_properties_false(value)
    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            schema[i] = set_additional_properties_false(item)
    return schema


def move_ref_descriptions(schema: dict) -> dict:
    """
    Finds descriptions next to $refs and moves them into the referenced definition.
    This is required for models like gpt-4.1-nano.
    """
    if isinstance(schema, dict):
        # The pattern to fix is a dictionary with both '$ref' and 'description'
        if '$ref' in schema and 'description' in schema:
            ref_path = schema['$ref']
            description = schema.pop('description')  # Remove description from here

            # The path is typically '#/$defs/ModelName'
            try:
                parts = ref_path.strip('#/').split('/')
                # Find the root of the schema to navigate from
                # This is a simplification; a more robust solution might need to pass the root down.
                # For a typical Pydantic schema, this will work if called on the top-level dict.
                if '$defs' in schema:
                    target_def = schema['$defs'][parts[1]]
                    if 'description' not in target_def:  # Don't overwrite existing description
                        target_def['description'] = description
                    else:
                        print(f"Warning: Description for ref {ref_path} already exists")
            except (KeyError, IndexError) as e:
                # Could not find the referenced definition, just leave it.
                print(f"Warning: Could not move description for ref {ref_path}: {e}")
                schema['description'] = description  # Put it back if failed

        # Recurse through the rest of the schema
        for key, value in schema.items():
            schema[key] = move_ref_descriptions(value)

    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            schema[i] = move_ref_descriptions(item)

    return schema


def remove_min_max(schema: dict) -> dict:
    """
    Recursively removes 'minimum' and 'maximum' keys from a JSON schema.
    Required for models like claude-sonnet-4-5.
    """
    if isinstance(schema, dict):
        if schema.get("type") in ["number", "integer"]:
            schema.pop("minimum", None)
            schema.pop("maximum", None)

        for key, value in schema.items():
            schema[key] = remove_min_max(value)

    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            schema[i] = remove_min_max(item)

    return schema
