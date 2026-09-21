"""JSON-safe request validation errors for the HTTP protocols."""

from __future__ import annotations

import math
from typing import Any

from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse


def _json_safe(value: Any) -> Any:
    """Coerce what json.dumps refuses (inf/nan, exceptions) into strings."""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def validation_error_response(exc: RequestValidationError) -> JSONResponse:
    # A body like {"presence_penalty": 1e999} parses to inf, and pydantic echoes
    # the offending value as the error's `input`; sending it back verbatim makes
    # the rejection itself unserializable (a 500 after a correct 422).
    return JSONResponse(status_code=422, content={"detail": _json_safe(list(exc.errors()))})
