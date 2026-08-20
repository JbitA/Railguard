from __future__ import annotations

import hmac


def write_key_valid(provided: str | None, configured: str | None) -> bool:
    """A missing configured key means local-development mode; otherwise compare safely."""
    if not configured:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided, configured)
