"""Build Octavian's native gravity backend for wheels and editable installations."""

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from tempfile import TemporaryDirectory

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ASSET_REVISION = "6cb73ac174b140ecc6ac1b9563012424f2f0a748"
ASSET_URL = "https://github.com/AlabamaASRL/asset_asrl.git"


def asset_headers(root):
    """Use an explicit offline checkout or fetch pinned headers into the build cache."""
    if os.environ.get("ASSET_SOURCE_DIR"):
        return Path(os.environ["ASSET_SOURCE_DIR"]).expanduser().resolve()
    cache = Path(root) / ".native-build"
    cache.mkdir(exist_ok=True)
    headers = cache / f"asset-{ASSET_REVISION}"
    if not headers.exists():
        with TemporaryDirectory(dir=cache, prefix="fetch-") as temporary:
            checkout = Path(temporary) / "asset"
            subprocess.run(["git", "init", str(checkout)], check=True)
            git = ["git", "-C", str(checkout)]
            subprocess.run([*git, "fetch", "--depth", "1", ASSET_URL, ASSET_REVISION], check=True)
            subprocess.run([*git, "checkout", "--detach", "FETCH_HEAD"], check=True)
            subprocess.run([*git, "remote", "add", "origin", ASSET_URL], check=True)
            subprocess.run(
                [
                    *git,
                    "submodule",
                    "update",
                    "--init",
                    "--depth",
                    "1",
                    "dep/eigen",
                    "dep/pybind11",
                    "dep/fmt",
                ],
                check=True,
            )
            try:
                checkout.rename(headers)
            except OSError:
                if not headers.is_dir():
                    raise
    return headers


def build_environment():
    """Select the compiler ABI used by the supported pip ASSET wheels."""
    environment = os.environ.copy()
    compiler = environment.get("CXX")
    if not compiler:
        candidates = (
            ("clang-cl",) if sys.platform == "win32" else ("clang++", "clang++-18", "clang++-17")
        )
        compiler = next((path for name in candidates if (path := shutil.which(name))), None)
        if compiler is None:
            raise RuntimeError(
                "Building Octavian requires Clang (clang-cl and the Visual Studio C++ tools on Windows). "
                "Install the native build prerequisites or set CXX to a compatible compiler. "
                "See docs/tutorials/spherical-harmonics.md. "
                "OCTAVIAN_BUILD_NATIVE=0 explicitly selects a Python-only developer install."
            )
        environment["CXX"] = compiler
    if sys.platform == "win32":
        environment["CXXFLAGS"] = (
            environment.get("CXXFLAGS", "") + " -fms-compatibility-version=19.40"
        )
    return environment


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        setting = os.environ.get("OCTAVIAN_BUILD_NATIVE", "1")
        if setting not in ("0", "1"):
            raise ValueError("OCTAVIAN_BUILD_NATIVE must be 0 or 1")
        if setting == "0":
            return
        if not (Path(sysconfig.get_path("include")) / "Python.h").is_file():
            raise RuntimeError(
                "Building Octavian requires development headers for this Python interpreter. "
                "Install the matching python3.x-dev package, or create the venv from a Python "
                "distribution that includes its headers (such as Conda). "
                "See docs/tutorials/spherical-harmonics.md."
            )
        environment = build_environment()
        headers = asset_headers(self.root)
        self._staging = TemporaryDirectory(prefix="octavian-native-")
        staging = Path(self._staging.name)
        source = Path(self.root) / "native" / "spherical_harmonics"
        install = staging / "install"
        build = staging / "build"
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DASSET_SOURCE_DIR={headers}",
                f"-DCMAKE_INSTALL_PREFIX={install}",
            ],
            check=True,
            env=environment,
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--parallel", "2"], check=True, env=environment
        )
        subprocess.run(["cmake", "--install", str(build)], check=True, env=environment)
        binaries = list(install.glob("*.so")) + list(install.glob("*.pyd"))
        if len(binaries) != 1:
            raise RuntimeError(f"Expected one native extension, found {binaries}")
        binary = binaries[0]
        licenses = install / "octavian_harmonics_native_licenses"
        if version == "editable":
            # Editable imports resolve to the checkout, not site-packages/octavian.
            # Keep the ABI-tagged extension beside the Python source after staging is removed.
            package = Path(self.root) / "octavian"
            pending = package / (binary.name + ".tmp")
            shutil.copy2(binary, pending)
            os.replace(pending, package / binary.name)
            shutil.copytree(licenses, package / "_native_licenses", dirs_exist_ok=True)
        else:
            build_data["force_include"][str(binary)] = f"octavian/{binary.name}"
            build_data["force_include"][str(licenses)] = "octavian/_native_licenses"
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

    def finalize(self, version, build_data, artifact_path):
        if hasattr(self, "_staging"):
            self._staging.cleanup()
