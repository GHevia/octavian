from __future__ import annotations

from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
COMPOSABLE_DIR = EXAMPLES_DIR / "composable"


def test_examples_are_flat_mission_scripts() -> None:
    """Examples should read like editable configuration, not app modules."""
    guarded_examples = []
    for path in sorted(EXAMPLES_DIR.rglob("*.py")):
        if "__main__" in path.read_text(encoding="utf-8"):
            guarded_examples.append(str(path.relative_to(EXAMPLES_DIR)))

    assert guarded_examples == []


def test_composable_examples_are_grouped_by_frame() -> None:
    """Composable scripts should live in a discoverable topical folder."""
    assert list(COMPOSABLE_DIR.glob("*.py")) == []
    categories = {
        path.parent.name
        for path in COMPOSABLE_DIR.rglob("*.py")
    }
    assert categories == {"cislunar", "earth_centered", "relative"}
