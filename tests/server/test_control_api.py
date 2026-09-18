"""``/health`` is a lifecycle probe and ``/ready`` is a readiness probe, and the two must not be
confused: the client polls ``/health`` to render load progress, so it answers 200 for the whole
lifecycle, while a deploy gate needs a probe that fails closed. These tests pin both halves --
the status codes for every lifecycle state, and the fact that ``/health`` keeps answering 200
where ``/ready`` answers 503 (a gate that polls the wrong one waits ~30s instead of the real
load time).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Same shim as the sibling server tests: the venv may hold a non-editable install, and without
# this the file only tests the source tree when a test that does insert it collects first.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.server.control_api import register_control_routes  # noqa: E402

VERSION = "0.0.0-test"


def _state(**over):
    base = dict(
        instance_id="instance-under-test",
        fatal_error=None,
        maintenance_state="serving",
        config=SimpleNamespace(served_model_name="unit-model"),
        load_progress=None,
        ready_at=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _client(state):
    app = FastAPI(version=VERSION)
    register_control_routes(app, lambda: state)
    return TestClient(app)


@pytest.mark.parametrize(
    "state_kwargs",
    [
        {"maintenance_state": "loading"},
        {"maintenance_state": "failed"},
        {"maintenance_state": "stopping"},
        # A cache rebuild keeps status "ok" but the API gate rejects new work until it resolves,
        # so readiness has to consider maintenance -- a status-only probe would say "ready" here.
        {"maintenance_state": "rebuilding"},
        {"fatal_error": "worker died"},
    ],
    ids=["loading", "failed", "stopping", "rebuilding", "fatal"],
)
def test_ready_fails_closed_but_health_stays_200(state_kwargs):
    client = _client(_state(**state_kwargs))
    assert client.get("/ready").status_code == 503
    # The lifecycle probe keeps its 200 contract: the client renders progress off this body.
    assert client.get("/health").status_code == 200


def test_ready_is_200_only_while_serving():
    client = _client(_state(ready_at=0.0))
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["maintenance"] == "serving"
    assert body["model"] == "unit-model"


def test_loading_body_still_reports_progress():
    lp = SimpleNamespace(phase="weights", done_bytes=5, total_bytes=10)
    client = _client(_state(maintenance_state="loading", load_progress=lp))
    body = client.get("/ready").json()
    assert body["status"] == "loading"
    assert body["phase"] == "weights"
    assert body["progress"] == {"done_bytes": 5, "total_bytes": 10}
