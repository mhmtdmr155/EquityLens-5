"""EquityLens-5 core environment and repository structure validation tests."""

import sys
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path of the project root dynamically."""
    return Path(__file__).resolve().parent.parent


def test_python_version() -> None:
    """Verify that Python execution environment is 3.12 or higher."""
    assert sys.version_info >= (3, 12), f"Expected Python >= 3.12, got: {sys.version}"


def test_project_root_detection() -> None:
    """Verify that the project root directory is correctly detected."""
    root = get_project_root()
    assert root.exists(), f"Project root does not exist: {root}"
    assert root.is_dir(), f"Project root is not a directory: {root}"


def test_critical_project_files_exist() -> None:
    """Verify that essential project governance and configuration files exist."""
    root = get_project_root()
    required_files = [
        "README.md",
        "AGENTS.md",
        ".gitignore",
        "pyproject.toml",
        "requirements-dev.txt",
    ]
    for filename in required_files:
        filepath = root / filename
        assert filepath.is_file(), f"Required file missing: {filename}"


def test_core_directories_exist() -> None:
    """Verify that architectural project directories exist."""
    root = get_project_root()
    required_dirs = [
        "backend",
        "frontend",
        "ml",
        "scripts",
        "docs",
        "tests",
    ]
    for dirname in required_dirs:
        dirpath = root / dirname
        assert dirpath.is_dir(), f"Required directory missing: {dirname}"
