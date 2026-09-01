"""Smoke tests: the package is installed and importable."""

import importlib


def test_package_imports():
    assert importlib.import_module("f1_strategy_engine") is not None
