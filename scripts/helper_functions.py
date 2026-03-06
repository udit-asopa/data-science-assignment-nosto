from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def parse_product_list(value: Any) -> list[int]:
	if isinstance(value, list):
		return [int(item) for item in value]
	if value is None:
		return []
	if isinstance(value, str):
		text = value.strip()
		if not text:
			return []
		try:
			parsed = ast.literal_eval(text)
		except (ValueError, SyntaxError):
			return []
		if isinstance(parsed, list):
			return [int(item) for item in parsed]
	return []


def ensure_parent_dir(file_path: str | Path) -> Path:
	path = Path(file_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def save_json(data: dict[str, Any], output_path: str | Path) -> Path:
	path = ensure_parent_dir(output_path)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2)
	return path
