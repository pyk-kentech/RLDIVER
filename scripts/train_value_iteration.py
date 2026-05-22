"""Convenience wrapper for running Value Iteration."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from davethediver_rl.main import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--mode", "value_iteration"])
    main()
