from __future__ import annotations

import os
from typing import Iterable

FALSEY_ENV_VALUES = {"0", "false", "no", "off"}
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def get_env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def get_env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError:
            return default

    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def get_env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw_value = os.getenv(name)
    if not raw_value:
        value = default
    else:
        try:
            value = float(raw_value)
        except ValueError:
            return default

    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def get_env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    normalized = raw_value.strip().lower()
    if normalized in TRUTHY_ENV_VALUES:
        return True
    if normalized in FALSEY_ENV_VALUES:
        return False
    return default


def get_env_csv(name: str, default: Iterable[str]) -> list[str]:
    raw_value = os.getenv(name)
    if not raw_value:
        return [item.strip().lower() for item in default if item and item.strip()]
    return [part.strip().lower() for part in raw_value.split(",") if part.strip()]
