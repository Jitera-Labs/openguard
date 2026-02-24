import importlib
import json
import pkgutil
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Type

import yaml
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

import src.guard_types
from src.guard_meta import GuardMeta


def get_type_string(prop: Dict[str, Any]) -> str:
    if "type" in prop:
        t = prop["type"]
        if isinstance(t, list):
            return " | ".join(str(item) for item in t)
        return str(t)
    if "anyOf" in prop:
        types = []
        for a in prop["anyOf"]:
            if "type" in a:
                types.append(str(a["type"]))
        if types:
            return " | ".join(types)
    return "any"


def generate_config_table(config_schema: Type[BaseModel]) -> str:
    schema = config_schema.model_json_schema()
    properties = schema.get("properties", {})
    if not properties:
        return "No configuration required.\n"

    lines = [
        "| Field | Type | Default | Description |",
        "|---|---|---|---|",
    ]

    for name, prop in properties.items():
        t = get_type_string(prop)

        # Get default from Pydantic model fields
        field_info = config_schema.model_fields[name]
        if field_info.default is not PydanticUndefined:
            default_val = field_info.default
        elif field_info.default_factory is not None:
            default_val = field_info.default_factory()  # type: ignore[call-arg]
        else:
            default_val = "Required"

        if default_val == "Required":
            default = "Required"
        elif isinstance(default_val, (list, dict)):
            default = json.dumps(default_val)
        elif isinstance(default_val, (set, frozenset)):
            default = json.dumps(sorted(list(default_val)))
        elif default_val is None:
            default = "None"
        elif isinstance(default_val, str):
            default = f'"{default_val}"'
        else:
            default = str(default_val)

        description = prop.get("description", "").replace("\n", " ")
        lines.append(f"| `{name}` | `{t}` | `{default}` | {description} |")

    return "\n".join(lines) + "\n"


def generate_examples(examples: List[Dict[str, Any]], module_name: str) -> str:
    if not examples:
        return "No examples provided.\n"

    blocks = []
    for i, ex in enumerate(examples):
        if "config" in ex and isinstance(ex["config"], dict):
            config = ex["config"]
            description = ex.get("description") or ex.get("name") or f"Example {i + 1}"
        else:
            config = ex
            description = f"Example {i + 1}"

        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
        indented_yaml = textwrap.indent(yaml_str, "  ").rstrip()

        block = f"```yaml\n# {description}\ntype: {module_name}\nconfig:\n{indented_yaml}\n```"
        blocks.append(block)

    return "\n\n".join(blocks) + "\n"


def main() -> None:
    # Use relative path based on the root of the project
    # The root is one level up from scripts
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "public" / "src" / "content" / "docs" / "guards"
    output_dir.mkdir(parents=True, exist_ok=True)

    package = src.guard_types
    # Get the path safely
    package_path = getattr(package, "__path__", [])
    for _, module_name, is_pkg in pkgutil.iter_modules(package_path):
        if is_pkg:
            continue

        full_module_name = f"src.guard_types.{module_name}"
        module = importlib.import_module(full_module_name)

        if not hasattr(module, "META"):
            continue

        meta = getattr(module, "META")
        if not isinstance(meta, GuardMeta):
            continue

        config_table = generate_config_table(meta.config_schema)
        examples_str = generate_examples(meta.examples, module_name)

        docs = meta.docs.strip() if meta.docs else ""

        mdx_content = f"""---
title: {meta.name}
description: {meta.description}
---

{meta.description}

{docs}

## Configuration

{config_table}
## Examples

{examples_str}
"""
        output_file = output_dir / f"{module_name}.mdx"
        output_file.write_text(mdx_content)
        print(f"Generated {output_file}")


if __name__ == "__main__":
    main()
