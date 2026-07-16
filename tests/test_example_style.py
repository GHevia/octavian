from __future__ import annotations

from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_examples_are_flat_mission_scripts() -> None:
    """Examples should read like editable configuration, not app modules."""
    guarded_examples = []
    for path in sorted(EXAMPLES_DIR.rglob("*.py")):
        if "__main__" in path.read_text(encoding="utf-8"):
            guarded_examples.append(str(path.relative_to(EXAMPLES_DIR)))

    assert guarded_examples == []
