from pathlib import Path

from octavian._runtime import enable_native_runtime
from octavian.diagnostics import RuntimeCheck, format_runtime_report, path_is_within


def test_path_is_within_accepts_descendant_path(tmp_path: Path) -> None:
    descendant = tmp_path / "environment" / "site-packages" / "asset.pyd"
    descendant.parent.mkdir(parents=True)
    descendant.touch()

    assert path_is_within(descendant, tmp_path / "environment")


def test_path_is_within_rejects_sibling_path(tmp_path: Path) -> None:
    environment = tmp_path / "environment"
    sibling = tmp_path / "other-environment" / "asset.pyd"
    environment.mkdir()
    sibling.parent.mkdir()
    sibling.touch()

    assert not path_is_within(sibling, environment)


def test_runtime_report_includes_every_check() -> None:
    report = format_runtime_report(
        (
            RuntimeCheck("user site disabled", True, "PYTHONNOUSERSITE is active."),
            RuntimeCheck("asset location", False, "C:/wrong/site-packages/asset.pyd"),
        )
    )

    assert "Octavian runtime diagnostics" in report
    assert "[PASS] user site disabled" in report
    assert "[FAIL] asset location" in report


def test_native_runtime_adds_environment_library_directory(tmp_path: Path) -> None:
    runtime_directory = tmp_path / "Library" / "bin"
    runtime_directory.mkdir(parents=True)
    added: list[str] = []

    handles = enable_native_runtime(prefix=tmp_path, add_dll_directory=added.append)

    assert added == [str(runtime_directory)]
    assert handles == (None,)


def test_native_runtime_is_noop_without_runtime_directory(tmp_path: Path) -> None:
    added: list[str] = []

    handles = enable_native_runtime(prefix=tmp_path, add_dll_directory=added.append)

    assert handles == ()
    assert added == []
