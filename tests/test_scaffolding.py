"""Scaffolding smoke test: verifies the package is importable.

Real models and logic arrive in later sub-tasks (1.2 onward).
"""

import vendor_forecasting_agent


def test_package_importable() -> None:
    assert vendor_forecasting_agent.__version__ == "0.1.0"
