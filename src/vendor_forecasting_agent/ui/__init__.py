"""Presentation-only Streamlit UI package for the Vendor Forecasting Agent.

This package contains an optional, presentation-only MVP UI layered on top of
the existing deterministic demo flow. Nothing in this package is imported by the
core package or the test suite's collection path, and no core module imports
anything from here.

Modules:

* :mod:`vendor_forecasting_agent.ui.view_model` — pure, streamlit-free data
  shaping helpers (importable and unit-testable without streamlit).
* :mod:`vendor_forecasting_agent.ui.app` — the Streamlit entry point
  (``streamlit run src/vendor_forecasting_agent/ui/app.py``). It imports
  ``streamlit`` at module top level and is only executed via ``streamlit run``.

The UI recomputes no business logic: every number originates from
``run_demo(...)`` (which runs ``run_pipeline`` + ``explain(provider=None)``) and
is read from the returned ``DemoOutput.snapshot`` / ``DemoOutput.explanation``.
"""
