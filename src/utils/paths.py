"""Helpers for resolving important project paths."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def get_configs_dir() -> Path:
    """Return the path to the ``configs`` directory."""
    return get_project_root() / "configs"


def get_contracts_dir() -> Path:
    """Return the path to the ``contracts`` directory."""
    return get_project_root() / "contracts"

