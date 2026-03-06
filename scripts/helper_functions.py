from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def parse_product_list(value: Any) -> list[int]:
	"""Parse a raw product-list value (string repr or actual list) into list[int].

	Handles: already-parsed lists, None / empty, JSON-style strings, and
	Python-literal strings produced by the data source.
	"""
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
	"""Create all parent directories for file_path if they do not yet exist."""
	path = Path(file_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def save_json(data: dict[str, Any], output_path: str | Path) -> Path:
	"""Serialise *data* to a pretty-printed JSON file at *output_path*."""
	path = ensure_parent_dir(output_path)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2)
	return path


def load_json(input_path: str | Path) -> dict[str, Any]:
	"""Load and return a JSON file as a plain dict."""
	path = Path(input_path)
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def print_section(title: str, width: int = 64) -> None:
	"""Print a labelled divider line to stdout."""
	pad = max(0, width - len(title) - 4)
	print(f"\n── {title} {'─' * pad}")
