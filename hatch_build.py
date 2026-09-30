"""Opt-in native wheel build; ordinary source/editable installs need no compiler."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if (
            self.target_name != "wheel"
            or version == "editable"
            or os.environ.get("OCTAVIAN_BUILD_NATIVE") != "1"
        ):
            return
        self._staging = TemporaryDirectory(prefix="octavian-native-")
        staging = Path(self._staging.name)
        source = Path(self.root) / "native" / "spherical_harmonics"
        build = staging / "build"
        install = staging / "install"
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
                f"-DCMAKE_INSTALL_PREFIX={install}",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True)
        subprocess.run(["cmake", "--install", str(build)], check=True)
        binaries = list(install.glob("*.so")) + list(install.glob("*.pyd"))
        if len(binaries) != 1:
            raise RuntimeError(f"Expected one native extension, found {binaries}")
        build_data["force_include"][str(binaries[0])] = f"octavian/{binaries[0].name}"
        build_data["force_include"][
            str(install / "octavian_harmonics_native_licenses")
        ] = "octavian/_native_licenses"
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

    def finalize(self, version, build_data, artifact_path):
        if hasattr(self, "_staging"):
            self._staging.cleanup()
