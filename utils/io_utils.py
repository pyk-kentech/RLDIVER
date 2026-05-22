"""Input and output helpers for models and experiment logs."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path`` object.

    If ``path`` looks like a file path because it has a suffix, the parent
    directory is created instead. This allows callers to pass either
    ``results`` or ``results/model.pkl``.
    """

    target = Path(path)
    directory = target.parent if target.suffix else target
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save a Python object with the standard ``pickle`` module."""

    path = Path(path)
    ensure_dir(path)
    with path.open("wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    """Load a Python object saved with ``save_pickle``."""

    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Save a sequence of dictionaries as a CSV file.

    The header is inferred from the first row. If ``rows`` is empty, an empty
    file is created so downstream scripts can still detect the output path.
    """

    path = Path(path)
    ensure_dir(path)
    rows = list(rows)

    with path.open("w", newline="", encoding="utf-8") as file:
        if not rows:
            return

        fieldnames = _ordered_fieldnames(rows)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ordered_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return stable CSV field names while preserving first-seen order."""

    fieldnames: list[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))

    return fieldnames


__all__ = ["ensure_dir", "save_pickle", "load_pickle", "save_csv"]
