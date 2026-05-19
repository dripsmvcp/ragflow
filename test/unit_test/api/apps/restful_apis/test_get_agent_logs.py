#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Regression tests for `get_agent_logs` (api/apps/restful_apis/agent_api.py).

Issue #14985: clicking the "Thinking" button on an embedded full-screen
chat returned `401 Unauthorized` because the embed sends an APIToken `beta`
field as `Authorization: Bearer <beta>`, but `_load_user` only checks the
`token` field. The route now accepts both flavours.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class _PassthroughManager:
    def route(self, *_args, **_kwargs):
        return lambda func: func


def _stub(monkeypatch, name, **attrs):
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


class _FakeAPITokenQuery:
    """Configurable double for APIToken.query(beta=...) lookups."""

    def __init__(self):
        self.beta_to_tenant = {}
        self.token_to_tenant = {}

    def __call__(self, **kwargs):
        beta = kwargs.get("beta")
        token = kwargs.get("token")
        tenant = None
        if beta is not None:
            tenant = self.beta_to_tenant.get(beta)
        elif token is not None:
            tenant = self.token_to_tenant.get(token)
        if tenant is None:
            return []
        return [SimpleNamespace(tenant_id=tenant)]


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)


def _load_agent_api(monkeypatch, *, current_user, api_token_query, redis, user_canvas_accessible=lambda *_a, **_k: True, authorization_header=""):
    """Load `api/apps/restful_apis/agent_api.py` with the minimum stubs needed
    to exercise `get_agent_logs`.

    The interesting knobs:
    * `current_user`        — what `@login_required`'s LocalProxy resolves to
    * `api_token_query`     — what `APIToken.query(beta=..., token=...)` returns
    * `redis`               — what `REDIS_CONN.get(key)` returns
    * `user_canvas_accessible` — whether the caller can read this agent
    * `authorization_header`  — value of `request.headers.get("Authorization")`
    """
    _stub(
        monkeypatch,
        "api.apps",
        current_user=current_user,
        login_required=lambda func: func,
    )
    _stub(monkeypatch, "api.apps.services.canvas_replica_service", CanvasReplicaService=SimpleNamespace())
    _stub(monkeypatch, "api.db", CanvasCategory=SimpleNamespace())
    _stub(monkeypatch, "api.db.db_models", Task=SimpleNamespace(), APIToken=SimpleNamespace(query=api_token_query))
    _stub(
        monkeypatch,
        "api.db.services.api_service",
        API4ConversationService=SimpleNamespace(
            get_by_id=lambda _sid: (False, None),
            save=lambda **_k: True,
            delete_by_id=lambda *_a, **_k: True,
            query=lambda **_k: [],
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.canvas_service",
        CanvasTemplateService=SimpleNamespace(),
        UserCanvasService=SimpleNamespace(
            accessible=user_canvas_accessible,
            query=lambda **_k: [],
        ),
        completion=lambda *_a, **_k: None,
        completion_openai=lambda *_a, **_k: None,
    )
    _stub(monkeypatch, "api.db.services.document_service", DocumentService=SimpleNamespace())
    _stub(monkeypatch, "api.db.services.file_service", FileService=SimpleNamespace())
    _stub(monkeypatch, "api.db.services.knowledgebase_service", KnowledgebaseService=SimpleNamespace())
    _stub(monkeypatch, "api.db.services.pipeline_operation_log_service", PipelineOperationLogService=SimpleNamespace())
    _stub(
        monkeypatch,
        "api.db.services.task_service",
        CANVAS_DEBUG_DOC_ID="",
        TaskService=SimpleNamespace(),
        queue_dataflow=lambda *_a, **_k: None,
    )
    _stub(
        monkeypatch,
        "api.db.services.user_service",
        TenantService=SimpleNamespace(),
        UserService=SimpleNamespace(get_by_id=lambda *_a, **_k: (False, None)),
    )
    _stub(monkeypatch, "api.db.services.user_canvas_version", UserCanvasVersionService=SimpleNamespace())
    _stub(
        monkeypatch,
        "api.utils.api_utils",
        add_tenant_id_to_kwargs=lambda func: func,
        get_data_error_result=lambda message="Sorry": {"code": 102, "message": message, "data": None},
        get_json_result=lambda code=0, message="", data=None: {"code": code, "message": message, "data": data},
        get_result=lambda **kwargs: kwargs,
        get_request_json=lambda: {},
        server_error_response=lambda exc: {"code": 500, "message": str(exc)},
        validate_request=lambda *_a, **_k: lambda func: func,
    )
    _stub(monkeypatch, "common.settings", retriever=SimpleNamespace(), kg_retriever=SimpleNamespace())
    _stub(monkeypatch, "common.ssrf_guard", assert_host_is_safe=lambda *_a, **_k: None)

    # Stub quart's `request` so the route can read its Authorization header.
    quart_stub = ModuleType("quart")
    quart_stub.Response = SimpleNamespace
    quart_stub.jsonify = lambda payload: payload
    quart_stub.request = SimpleNamespace(
        headers={"Authorization": authorization_header} if authorization_header else {},
    )
    # SimpleNamespace.headers does not have .get; promote it to a dict.
    quart_stub.request.headers = {"Authorization": authorization_header} if authorization_header else {}
    monkeypatch.setitem(sys.modules, "quart", quart_stub)

    # Stub rag.utils.redis_conn so the route's lazy import of REDIS_CONN finds it.
    _stub(monkeypatch, "rag", __path__=[])
    _stub(monkeypatch, "rag.utils", __path__=[])
    _stub(monkeypatch, "rag.utils.redis_conn", REDIS_CONN=redis)

    repo_root = Path(__file__).resolve().parents[5]
    module_path = repo_root / "api" / "apps" / "restful_apis" / "agent_api.py"
    spec = importlib.util.spec_from_file_location("test_get_agent_logs_agent_api", module_path)
    module = importlib.util.module_from_spec(spec)
    module.manager = _PassthroughManager()
    monkeypatch.setitem(sys.modules, "test_get_agent_logs_agent_api", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.p1
class TestGetAgentLogsBetaTokenAuth:
    """Regression for #14985: trace endpoint must accept the embed `beta` token."""

    @pytest.mark.p1
    def test_beta_token_returns_cached_log_payload(self, monkeypatch):
        """An embedded request (Bearer <beta>) must resolve to the agent owner
        and return the Redis-cached log payload."""
        query = _FakeAPITokenQuery()
        query.beta_to_tenant["ULG53523-beta"] = "tenant-owner-1"

        redis = _FakeRedis()
        redis.store["agent-1-msg-1-logs"] = json.dumps([{"event": "node_finished"}])

        module = _load_agent_api(
            monkeypatch,
            current_user=None,
            api_token_query=query,
            redis=redis,
            authorization_header="Bearer ULG53523-beta",
        )

        result = asyncio.run(module.get_agent_logs(agent_id="agent-1", message_id="msg-1"))

        assert result == {
            "code": 0,
            "message": "",
            "data": [{"event": "node_finished"}],
        }

    @pytest.mark.p1
    def test_missing_authorization_returns_unauthorized(self, monkeypatch):
        """No Authorization header and no logged-in user must yield an auth error,
        not crash or leak data."""
        module = _load_agent_api(
            monkeypatch,
            current_user=None,
            api_token_query=_FakeAPITokenQuery(),
            redis=_FakeRedis(),
            authorization_header="",
        )

        result = asyncio.run(module.get_agent_logs(agent_id="agent-1", message_id="msg-1"))

        assert result["code"] == 109  # RetCode.AUTHENTICATION_ERROR
        assert "Authentication" in result["message"]

    @pytest.mark.p1
    def test_beta_token_from_other_tenant_is_rejected(self, monkeypatch):
        """A valid beta token whose tenant does NOT own this agent must be
        denied by the access check — embed tokens scope to one agent's owner,
        not every agent in the world."""
        query = _FakeAPITokenQuery()
        query.beta_to_tenant["foreign-beta"] = "tenant-other"

        def _accessible_only_for_owner(agent_id, tenant_id):
            return tenant_id == "tenant-owner-1"

        module = _load_agent_api(
            monkeypatch,
            current_user=None,
            api_token_query=query,
            redis=_FakeRedis(),
            user_canvas_accessible=_accessible_only_for_owner,
            authorization_header="Bearer foreign-beta",
        )

        result = asyncio.run(module.get_agent_logs(agent_id="agent-1", message_id="msg-1"))

        # The route should refuse to read logs the caller has no access to.
        assert result["code"] != 0
        assert "permission" in result["message"].lower() or "access" in result["message"].lower()
