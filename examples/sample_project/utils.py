"""
Shared utility functions for the sample e-commerce application.

Intentionally clean and well-documented — expected to be LOW risk.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier with an optional prefix."""
    uid = str(uuid.uuid4()).replace("-", "")[:12]
    return f"{prefix}{uid}" if prefix else uid


def validate_email(email: str) -> bool:
    """Return True if email matches a basic RFC-5322 pattern."""
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def hash_password(password: str, salt: str = "") -> str:
    """Return a hex SHA-256 hash of the password + salt."""
    payload = (password + salt).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value to [minimum, maximum]."""
    return max(minimum, min(value, maximum))


def paginate(items: list[Any], page: int, page_size: int) -> list[Any]:
    """Return a page-sized slice of items (1-indexed pages)."""
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be positive integers")
    start = (page - 1) * page_size
    return items[start : start + page_size]


def format_currency(amount: float, symbol: str = "$") -> str:
    """Format a float as a currency string."""
    return f"{symbol}{amount:,.2f}"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
