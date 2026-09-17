"""Compatibility shim.

All packaging metadata lives in ``pyproject.toml``; this file exists only so
that ``pip install -e .`` keeps working with older pip/setuptools versions that
still expect a setup script.
"""

from setuptools import setup

setup()
