import json
from datetime import date, datetime
from typing import Any


class _DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, cls=_DateEncoder, default=str)


def format_indian_number(n: float) -> str:
    """Format a number using Indian convention (lakhs/crores)."""
    n = abs(n)
    if n >= 1_00_00_000:
        return f"{n / 1_00_00_000:.2f} crore"
    if n >= 1_00_000:
        return f"{n / 1_00_000:.2f} lakh"
    return f"{n:,.0f}"


def truncate_text(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"
