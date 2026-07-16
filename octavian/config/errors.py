"""Errors raised while reading declarative mission configuration."""

from __future__ import annotations


class MissionConfigError(ValueError):
    """A mission configuration file is malformed or internally inconsistent."""
