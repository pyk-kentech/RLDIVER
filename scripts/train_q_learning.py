"""Convenience wrapper for training Q-learning."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from main import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--mode", "train_q_learning"])
    main()
