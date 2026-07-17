from pathlib import Path

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
