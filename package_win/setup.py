#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from cx_Freeze import setup, Executable
# https://cx-freeze.readthedocs.io/en/stable/setup_script.html

# cx_Freeze walks the import graph recursively; the deep transitive import trees of
# numpy/scipy/matplotlib/sympy/pandas exceed Python's default recursion limit of 1000
# and raise RecursionError during _scan_code. Raise the limit before the build runs.
sys.setrecursionlimit(5000)

# This script lives in package_win/, one level below the project root. Anchor every
# source path to the root and put the root on sys.path so the project imports and
# the file globbing work regardless of the directory the build is launched from
# (the build itself is run from package_win/ so build/ and dist/ land there).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.app_config import APP_DEFINITIONS

# include_files entries are (absolute source, path-inside-the-build). Sources are
# absolute so the build does not depend on the current directory; destinations stay
# root-relative so the frozen layout mirrors the source tree.
files_to_include = [
    (str(ROOT / "LICENSE"), "LICENSE"),
    (str(ROOT / "README.md"), "README.md"),
    (str(ROOT / APP_DEFINITIONS["icon_path"]), APP_DEFINITIONS["icon_path"]),
    *[(str(file), str(file.relative_to(ROOT))) for file in (ROOT / "data").rglob("*")],
    ]

# sounddevice/soundfile ship their PortAudio/libsndfile binaries in package data
# folders that cx_Freeze's import analysis does not pick up. Locate them from the
# installed packages and bundle each at the same relative path when it exists
# (e.g. _sounddevice_data ships on Windows, not on Linux).
for pkg_name, data_dir_name in (("sounddevice", "_sounddevice_data"),
                                ("soundfile", "_soundfile_data")):
    module = __import__(pkg_name)
    data_dir = Path(module.__file__).parent / data_dir_name
    if data_dir.is_dir():
        files_to_include.append((str(data_dir), data_dir_name))

print("Warning.. Adding following additional files to package:")
for pair in files_to_include:
    print("\t" + pair[0])
print()

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["numpy", "scipy", "matplotlib", "sympy", "pandas",
                 "odf",  # dynamically imported by pandas as the .ods engine; not visible to static analysis
                 "sounddevice", "soundfile",  # ship their bundled PortAudio/libsndfile binaries
                 # The report attaches the session state with pypdf, imported inside the
                 # function that needs it. cx_Freeze does follow that import, but only if
                 # the package is installed in the build environment -- when it is not, the
                 # build succeeds quietly and the frozen application raises ImportError as
                 # soon as a report is written. Naming it here turns that into a build-time
                 # failure instead.
                 "pypdf"],
    # QtWebEngineCore, which the report uses to lay out and print the PDF, pulls
    # QtPrintSupport in from its C++ side. The import never appears in Python source,
    # and the stub cx_Freeze substitutes for the binding names every other dependency
    # of it but that one, so the module is left out and the frozen build raises
    # ImportError the moment the report is opened. Name it explicitly to have the
    # binding and its Qt6PrintSupport library packaged.
    "includes": ["PySide6.QtPrintSupport"],
    "include_files": files_to_include,
    # 0 prints every module cx_Freeze copies, on top of the missing-module and
    # missing-dependency reports that levels below 2 and 3 already show. Verbose,
    # but it is the only way to confirm from the build log that something made it
    # into the package.
    "silent_level": 0,
}

bdist_msi_options = {
    "extensions": [{"extension": "sscf",
                    "verb": "load",
                    "argument": '"%1"',
                    "executable": "main.exe",
                    }]
    }

executables=[Executable(str(ROOT / "main.py"),
                        copyright=APP_DEFINITIONS["copyright"],
                        base="gui",
                        shortcut_name=APP_DEFINITIONS["app_name"] + " v" + APP_DEFINITIONS["version"],
                        shortcut_dir="DesktopFolder",
                        icon=str(ROOT / APP_DEFINITIONS["icon_path"]),
                        ),
            ]

setup(
    name=APP_DEFINITIONS["app_name"],
    version=APP_DEFINITIONS["version"],
    description=APP_DEFINITIONS["description"],
    options={"build_exe": build_exe_options, "bdist_msi": bdist_msi_options},
    executables=executables,
)
