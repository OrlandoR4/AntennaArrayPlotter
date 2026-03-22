"""
build.py — Builds the Antenna Array Plotter into a standalone .exe using PyInstaller.
Run this script from your project folder:
    python build.py
"""

import subprocess
import sys
import os
import shutil

# ── Configuration ────────────────────────────────────────────────────────────

MAIN_SCRIPT     = "main.py"
APP_NAME        = "AntennaArrayPlotter"
ICON_FILE       = "icon.ico"

EXTRA_FILES     = [
    ("icon.ico", "."),
]

HIDDEN_IMPORTS  = [
    "scipy.special",
    "scipy.signal",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_build_artifacts():
    """Remove previous build/dist folders and .spec file for a clean build."""
    for path in ["build", "dist", f"{APP_NAME}.spec"]:
        if os.path.exists(path):
            print(f"  Removing {path}...")
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

def build():
    print("=" * 60)
    print(f"  Building {APP_NAME}")
    print("=" * 60)

    # Clean previous artifacts
    print("\n[1/3] Cleaning previous build artifacts...")
    clean_build_artifacts()

    # Assemble PyInstaller command
    print("\n[2/3] Running PyInstaller...")
    separator = ";" if sys.platform == "win32" else ":"

    command = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        f"--name={APP_NAME}",
    ]

    # Add icon if it exists
    if os.path.exists(ICON_FILE):
        command.append(f"--icon={ICON_FILE}")
    else:
        print(f"  Warning: {ICON_FILE} not found, skipping icon.")

    # Add extra data files
    for src, dst in EXTRA_FILES:
        if os.path.exists(src):
            command.append(f"--add-data={src}{separator}{dst}")
        else:
            print(f"  Warning: {src} not found, skipping.")

    # Add hidden imports
    for imp in HIDDEN_IMPORTS:
        command.append(f"--hidden-import={imp}")

    command.append(MAIN_SCRIPT)

    result = subprocess.run(command)

    # Result
    print("\n[3/3] Done.")
    if result.returncode == 0:
        exe_path = os.path.join("dist", f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME)
        print(f"\n  Build successful!")
        print(f"  Executable: {exe_path}")
    else:
        print("\n  Build FAILED. Re-run without --windowed to see errors:")
        print(f"    python -m PyInstaller --onefile --name={APP_NAME} {MAIN_SCRIPT}")
        sys.exit(1)

if __name__ == "__main__":
    build()