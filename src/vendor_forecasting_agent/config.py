"""Configuration loading for the Vendor Forecasting Agent (config.py).

This module is the ``Config_Module`` described in the design. It parses a
configuration source into a validated :class:`~vendor_forecasting_agent.schema.Config`
model, applying documented defaults for absent settings and translating
validation/IO failures into a single, descriptive :class:`ConfigError`.

Design constraints honored here (Requirement 2):

* **2.1** — Loading returns quickly (well under 5 seconds). There are no sleeps,
  retries, or waits; the only IO is a single local file read for the path case.
* **2.2** — The ``Random_Seed`` setting is exposed via ``Config.random_seed`` and
  is constrained to ``[0, 4_294_967_295]`` by the schema.
* **2.3** — Documented defaults are applied for any absent setting and are
  present on the returned ``Config``.
* **2.4** — On a type/range failure, a ``ConfigError`` is raised naming the
  offending setting(s) and the reason; no ``Config`` is returned.
* **2.5** — When the source is absent or cannot be read, a ``ConfigError`` is
  raised indicating the source could not be loaded; no ``Config`` is returned.
* **2.6** — No network or external API calls are performed. The path case only
  reads a local file using the standard-library ``json`` module.

Documented defaults
-------------------
The schema supplies defaults for most settings:

* ``forecast_horizon`` -> ``12``
* ``record_count`` -> ``1000``
* ``stable_threshold_pct`` -> ``5.0``

``Config.random_seed`` is a *required* field with no schema default (a seed must
be an explicit, reproducible choice). To let callers request an all-defaults
configuration (``load_config(None)`` or a mapping/file that omits the seed), this
module defines :data:`DEFAULT_RANDOM_SEED`. It is applied only when the source
does not provide ``random_seed``. ``0`` is chosen as the documented default
because it is the smallest valid seed, is unambiguous and reproducible, and makes
"defaults" behavior deterministic and easy to reason about.
"""

import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from .schema import Config

# Documented default for the required ``random_seed`` setting. Applied only when
# a source omits ``random_seed``. ``0`` is the smallest valid seed and yields a
# reproducible, unambiguous defaults configuration. See module docstring.
DEFAULT_RANDOM_SEED: int = 0


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or fails validation.

    The message names the offending setting and the reason for validation
    failures (Req 2.4), or indicates that the source could not be loaded for an
    absent/unreadable source (Req 2.5).
    """


def _apply_defaults(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a plain dict of the provided settings with documented defaults for
    absent values applied.

    Only the required ``random_seed`` needs a default injected here; the
    remaining defaults (``forecast_horizon``, ``record_count``,
    ``stable_threshold_pct``) are supplied by the ``Config`` schema itself when
    the key is absent (Req 2.3).
    """
    merged: dict[str, Any] = dict(values)
    merged.setdefault("random_seed", DEFAULT_RANDOM_SEED)
    return merged


def _build_config(values: Mapping[str, Any]) -> Config:
    """Apply defaults and construct a validated ``Config``.

    Re-raises any ``pydantic.ValidationError`` as a ``ConfigError`` naming the
    invalid setting(s) and reason (Req 2.4). No ``Config`` is returned on
    failure.
    """
    merged = _apply_defaults(values)
    try:
        return Config(**merged)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc']) or '<config>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration: {details}") from exc


def _load_mapping_from_path(source: str | Path) -> Mapping[str, Any]:
    """Read a local JSON config file into a mapping.

    Raises ``ConfigError`` if the source is absent, unreadable, or does not
    contain a JSON object (Req 2.5). Only a local file read is performed — no
    network access (Req 2.6).
    """
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Configuration source could not be loaded: {path} ({exc.strerror or exc})"
        ) from exc

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Configuration source could not be loaded: {path} (invalid JSON: {exc})"
        ) from exc

    if not isinstance(parsed, Mapping):
        raise ConfigError(
            f"Configuration source could not be loaded: {path} "
            f"(expected a JSON object, got {type(parsed).__name__})"
        )
    return parsed


def load_config(source: str | Path | Mapping[str, Any] | None = None) -> Config:
    """Parse and validate configuration into a :class:`Config` model.

    Accepts three kinds of source:

    * ``None`` — build an all-defaults configuration. The documented
      :data:`DEFAULT_RANDOM_SEED` is applied for ``random_seed`` and the schema
      defaults are used for the remaining settings.
    * ``Mapping`` — use the provided keys, apply documented defaults for any
      absent keys, and validate.
    * ``str`` / ``Path`` — treat as a path to a local JSON config file. The file
      is read locally (no network) and parsed into a mapping, then handled like
      the mapping case.

    Behavior:

    * Applies documented defaults for absent settings and includes them in the
      returned ``Config`` (Req 2.3).
    * Raises :class:`ConfigError` naming the invalid setting + reason on a
      type/range failure, returning no ``Config`` (Req 2.4).
    * Raises :class:`ConfigError` indicating the source could not be loaded when
      the path is absent/unreadable, returning no ``Config`` (Req 2.5).
    * Performs no network or external API calls; only a local file read for the
      path case (Req 2.6).
    * Completes quickly, well within 5 seconds (Req 2.1).
    """
    if source is None:
        return _build_config({})
    if isinstance(source, Mapping):
        return _build_config(source)
    if isinstance(source, (str, Path)):
        return _build_config(_load_mapping_from_path(source))
    raise ConfigError(
        f"Configuration source could not be loaded: unsupported source type "
        f"{type(source).__name__}"
    )
