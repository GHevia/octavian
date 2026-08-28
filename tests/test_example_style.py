from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
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


def test_composable_example_numbers_are_unique_and_grouped() -> None:
    """Each regime owns one contiguous range in the global progression."""
    ordered_categories = ("earth_centered", "relative", "cislunar")
    grouped_numbers = []
    for category in ordered_categories:
        numbers = sorted(
            int(path.name.split("_", 1)[0])
            for path in (COMPOSABLE_DIR / category).glob("*.py")
        )
        assert numbers == list(range(numbers[0], numbers[-1] + 1))
        grouped_numbers.extend(numbers)

    assert grouped_numbers == list(range(1, len(grouped_numbers) + 1))


def test_python_examples_compile_and_have_module_docstrings() -> None:
    """Every executable guide should be syntactically valid and self-describing."""
    failures: list[str] = []
    for path in sorted(EXAMPLES_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            module = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
            continue
        if not ast.get_docstring(module):
            failures.append(f"{path.relative_to(ROOT)}: missing module docstring")

    assert failures == []


def test_capability_index_lists_every_executable_example() -> None:
    """The task-oriented index must not omit a runnable Python/config example."""
    capability_index = (ROOT / "docs" / "examples" / "index.md").read_text(encoding="utf-8")
    omitted = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(EXAMPLES_DIR.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".json", ".yaml", ".yml"}
        and path.relative_to(ROOT).as_posix() not in capability_index
    ]

    assert omitted == []


def test_documented_example_paths_exist() -> None:
    """Example paths in user documentation should remain valid after reorganizing."""
    documentation = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").rglob("*.md")),
        *sorted(EXAMPLES_DIR.rglob("README.md")),
    ]
    missing: list[str] = []
    pattern = re.compile(r"examples/[A-Za-z0-9_./-]+\.(?:py|json|yaml|yml)")
    for document in documentation:
        text = document.read_text(encoding="utf-8")
        for match in pattern.findall(text):
            if not (ROOT / match).is_file():
                missing.append(f"{document.relative_to(ROOT)}: {match}")

    assert missing == []
