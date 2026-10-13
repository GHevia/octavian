"""Native build defaults, editable artifact lifetime, and explicit opt-out."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "octavian_build_hook", Path(__file__).parents[1] / "hatch_build.py"
)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def hook(tmp_path, target="wheel"):
    (tmp_path / "octavian").mkdir(exist_ok=True)
    return build.CustomBuildHook(str(tmp_path), {}, None, None, str(tmp_path), target)


@pytest.mark.parametrize("version", ["standard", "editable"])
def test_default_build_retains_binary_and_licenses_after_cleanup(tmp_path, monkeypatch, version):
    monkeypatch.delenv("OCTAVIAN_BUILD_NATIVE", raising=False)
    monkeypatch.setattr(build, "build_environment", lambda: {})
    monkeypatch.setattr(build, "asset_headers", lambda root: tmp_path / "headers")
    include = tmp_path / "include"
    include.mkdir()
    (include / "Python.h").touch()
    monkeypatch.setattr(build.sysconfig, "get_path", lambda key: str(include))
    destination = None

    def compile_extension(command, **kwargs):
        nonlocal destination
        for item in command:
            if item.startswith("-DCMAKE_INSTALL_PREFIX="):
                destination = Path(item.split("=", 1)[1])
        if "--install" in command:
            destination.mkdir()
            (destination / "octavian_harmonics_native.test.so").write_bytes(b"native")
            licenses = destination / "octavian_harmonics_native_licenses"
            licenses.mkdir()
            (licenses / "ASSET.txt").write_text("upstream license")

    monkeypatch.setattr(build.subprocess, "run", compile_extension)
    builder = hook(tmp_path)
    data = {"force_include": {}}
    builder.initialize(version, data)
    assert data["pure_python"] is False
    assert data["infer_tag"] is True
    if version == "standard":
        assert set(data["force_include"].values()) == {
            "octavian/octavian_harmonics_native.test.so",
            "octavian/_native_licenses",
        }
    staging = Path(builder._staging.name)
    builder.finalize(version, data, "unused")
    assert not staging.exists()
    if version == "editable":
        assert (tmp_path / "octavian/octavian_harmonics_native.test.so").read_bytes() == b"native"
        assert (tmp_path / "octavian/_native_licenses/ASSET.txt").read_text() == "upstream license"


@pytest.mark.parametrize("target,setting", [("sdist", "1"), ("wheel", "0")])
def test_source_archive_and_explicit_python_only_build_need_no_compiler(
    tmp_path, monkeypatch, target, setting
):
    monkeypatch.setenv("OCTAVIAN_BUILD_NATIVE", setting)
    monkeypatch.setattr(build, "build_environment", lambda: pytest.fail("compiler requested"))
    data = {"force_include": {}}
    hook(tmp_path, target).initialize("editable", data)
    assert data == {"force_include": {}}


def test_native_failure_does_not_silently_produce_python_only_install(tmp_path, monkeypatch):
    monkeypatch.delenv("OCTAVIAN_BUILD_NATIVE", raising=False)
    monkeypatch.setattr(build, "build_environment", lambda: {})
    monkeypatch.setattr(build, "asset_headers", lambda root: tmp_path)
    include = tmp_path / "include"
    include.mkdir()
    (include / "Python.h").touch()
    monkeypatch.setattr(build.sysconfig, "get_path", lambda key: str(include))

    def failed_compile(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "cmake")

    monkeypatch.setattr(build.subprocess, "run", failed_compile)
    builder = hook(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        builder.initialize("editable", {"force_include": {}})
    builder.finalize("editable", {}, "unused")
    assert not list((tmp_path / "octavian").glob("*.so"))


def test_missing_python_headers_explains_required_dependency(tmp_path, monkeypatch):
    monkeypatch.delenv("OCTAVIAN_BUILD_NATIVE", raising=False)
    monkeypatch.setattr(build.sysconfig, "get_path", lambda key: str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="development headers"):
        hook(tmp_path).initialize("editable", {"force_include": {}})
