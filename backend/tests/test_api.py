"""API surface tests.

These use FastAPI's TestClient against the real app but avoid live LLM calls
wherever possible - the FAQ fast path and the guardrail both answer without a
model, which makes most of the HTTP contract testable for free.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import _events_for, app
from app.tools.db import fetch


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    try:
        fetch("SELECT 1")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres unreachable ({exc})")


# ---------------------------------------------------------------------------
class TestOps:
    def test_health_reports_every_dependency(self, client):
        body = client.get("/health").json()
        assert body["status"] in ("ok", "degraded")
        assert "database" in body
        assert "vector_store" in body
        assert "llm" in body

    def test_root_points_at_the_docs(self, client):
        assert client.get("/").json()["docs"] == "/docs"

    def test_openapi_exposes_the_chat_endpoints(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert "/chat" in paths
        assert "/chat/sync" in paths


class TestValidation:
    def test_empty_message_is_rejected(self, client):
        assert client.post("/chat/sync", json={"message": ""}).status_code == 422

    def test_missing_message_is_rejected(self, client):
        assert client.post("/chat/sync", json={}).status_code == 422

    def test_overlong_message_is_rejected(self, client):
        r = client.post("/chat/sync", json={"message": "x" * 5000})
        assert r.status_code == 422


class TestGuardrailOverHttp:
    """Emergencies answer without any model call, so this is deterministic."""

    def test_emergency_is_intercepted(self, client):
        body = client.post(
            "/chat/sync", json={"message": "my father collapsed and is unconscious"}
        ).json()
        assert body["route"] == "EMERGENCY"
        assert "1990" in body["answer"]

    def test_diagnosis_request_is_refused(self, client):
        body = client.post(
            "/chat/sync", json={"message": "what disease do I have?"}
        ).json()
        assert body["route"] == "REFUSE"

    def test_response_carries_the_trace(self, client):
        body = client.post(
            "/chat/sync", json={"message": "my father collapsed and is unconscious"}
        ).json()
        assert body["trace"]
        assert any(e.get("node") == "guardrail" for e in body["trace"])


class TestThreads:
    def test_thread_id_is_generated_when_absent(self, client):
        body = client.post(
            "/chat/sync", json={"message": "my father collapsed"}
        ).json()
        assert body["thread_id"].startswith("web-")

    def test_thread_history_roundtrip(self, client):
        first = client.post(
            "/chat/sync", json={"message": "my father collapsed"}
        ).json()
        tid = first["thread_id"]
        try:
            thread = client.get(f"/threads/{tid}").json()
            assert thread["thread_id"] == tid
            assert thread["turn_count"] >= 1
            assert thread["history"]
        finally:
            client.delete(f"/threads/{tid}")

    def test_delete_clears_the_thread(self, client):
        body = client.post("/chat/sync", json={"message": "my father collapsed"}).json()
        tid = body["thread_id"]
        client.delete(f"/threads/{tid}")
        assert client.get(f"/threads/{tid}").json()["turn_count"] == 0

    def test_unknown_thread_is_empty_not_an_error(self, client):
        body = client.get("/threads/does-not-exist-xyz").json()
        assert body["history"] == []
        assert body["turn_count"] == 0


class TestSseStream:
    def test_stream_emits_typed_events_in_order(self, client):
        with client.stream(
            "POST", "/chat", json={"message": "my father collapsed and is unconscious"}
        ) as resp:
            assert resp.status_code == 200
            events = []
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())

        assert events[0] == "start"
        assert "token" in events
        assert events[-1] == "done"

    def test_done_payload_has_the_answer(self, client):
        with client.stream(
            "POST", "/chat", json={"message": "my father collapsed"}
        ) as resp:
            event, payload = None, None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event == "done":
                    payload = json.loads(line.split(":", 1)[1].strip())

        assert payload is not None
        assert payload["route"] == "EMERGENCY"
        assert "1990" in payload["answer"]


class TestEventMapping:
    """The node -> SSE translation, without needing a live turn."""

    def test_route_update_becomes_a_route_event(self):
        events = _events_for("classify", {
            "route": "SQL",
            "route_decision": {"confidence": 0.97, "rationale": "fee lookup"},
        })
        assert len(events) == 1
        assert events[0]["event"] == "route"
        assert json.loads(events[0]["data"])["route"] == "SQL"

    def test_sql_update_becomes_a_tool_call_event(self):
        events = _events_for("sql_agent", {
            "sql_calls": [{"tool": "find_doctors", "row_count": 3,
                           "sql_executed": "SELECT 1", "notes": []}]
        })
        assert events[0]["event"] == "tool_call"
        assert json.loads(events[0]["data"])["row_count"] == 3

    def test_retrieval_update_becomes_a_sources_event(self):
        events = _events_for("retrieve", {
            "retrieval": {"returned": 2},
            "documents": [{"citation": "A", "rerank_score": 0.9, "rank": 1}],
        })
        assert events[0]["event"] == "sources"
        assert len(json.loads(events[0]["data"])["documents"]) == 1

    def test_rewrite_is_only_emitted_when_it_happened(self):
        assert _events_for("contextualize", {"memory": {"rewritten": False}}) == []
        events = _events_for("contextualize", {
            "memory": {"rewritten": True, "original_query": "his fee?"},
            "contextual_query": "Dr X fee?",
        })
        assert events[0]["event"] == "rewrite"

    def test_clear_guardrail_emits_nothing(self):
        assert _events_for("check_safety", {"guardrail": {"verdict": "CLEAR"}}) == []
