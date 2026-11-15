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
    A description beside a `$ref` appended to the description of the definition it refers to,
    where strict mode accepts it (required for models like gpt-4.1-nano). Nothing is dropped:
    a definition that has a description keeps it, the moved one following after a blank line.

    Raises:
        ValueError: on a described `$ref` that does not point into the schema's own `$defs`.
    """
    definitions = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node and "description" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                if not node["$ref"].startswith("#/$defs/") or name not in definitions:
                    raise ValueError(f"cannot move the description beside {node['$ref']}: no such definition")
                target = definitions[name]
                target["description"] = "\n\n".join(part for part in (target.get("description"), node.pop("description")) if part)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return schema


def bounds_as_text(schema: dict) -> dict:
    """
    A number's `minimum` and `maximum`, which Claude's strict mode refuses as keywords, written
    at the end of its description instead, `{minimum: 0, maximum: 1}` - as Anthropic's own SDK does.
    """
    if isinstance(schema, dict):
        if schema.get("type") in ("number", "integer"):
            bounds = {key: schema.pop(key) for key in ("minimum", "maximum") if key in schema}
            if bounds:
                text = "{" + ", ".join(f"{key}: {value}" for key, value in bounds.items()) + "}"
                schema["description"] = "\n\n".join(part for part in (schema.get("description"), text) if part)
        for value in schema.values():
            bounds_as_text(value)
    elif isinstance(schema, list):
        for item in schema:
            bounds_as_text(item)
    return schema
