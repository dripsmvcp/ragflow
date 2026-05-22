#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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

"""Unit tests for the E2B sandbox provider.

The E2B SDK (e2b_code_interpreter) is an optional dependency that is not
installed in CI, so these tests inject a fake ``e2b_code_interpreter``
module into sys.modules before initialize() performs its lazy import. No
network calls are made; live verification against E2B Cloud is done
separately with a real API key.
"""

import sys
import types

import pytest

from agent.sandbox.providers.e2b import E2BProvider
from agent.sandbox.providers.base import SandboxInstance, ExecutionResult
from agent.sandbox.result_protocol import RESULT_MARKER_PREFIX

pytestmark = pytest.mark.p2


class _FakeLogs:
    def __init__(self, stdout=None, stderr=None):
        self.stdout = stdout or []
        self.stderr = stderr or []


class _FakeError:
    def __init__(self, name="", value="", traceback=""):
        self.name = name
        self.value = value
        self.traceback = traceback


class _FakeExecution:
    def __init__(self, stdout=None, stderr=None, error=None, execution_count=1):
        self.logs = _FakeLogs(stdout, stderr)
        self.error = error
        self.execution_count = execution_count


class _FakeSandbox:
    """Stand-in for e2b_code_interpreter.Sandbox.

    Records constructor kwargs and run_code calls; returns a scripted
    Execution. Class-level hooks let individual tests customize behavior.
    """

    instances = []
    next_execution = None
    run_error = None
    connect_calls = []

    def __init__(self, timeout=None, api_key=None, domain=None, **kwargs):
        self.timeout = timeout
        self.api_key = api_key
        self.domain = domain
        self.sandbox_id = f"sbx-{len(_FakeSandbox.instances)}"
        self.run_calls = []
        self.killed = False
        _FakeSandbox.instances.append(self)

    def run_code(self, code, language="python", timeout=None, **kwargs):
        self.run_calls.append({"code": code, "language": language, "timeout": timeout})
        if _FakeSandbox.run_error is not None:
            raise _FakeSandbox.run_error
        return _FakeSandbox.next_execution

    def kill(self):
        self.killed = True
        return True

    @classmethod
    def connect(cls, sandbox_id, api_key=None, domain=None, **kwargs):
        cls.connect_calls.append(sandbox_id)
        sbx = cls.__new__(cls)
        sbx.timeout = None
        sbx.api_key = api_key
        sbx.domain = domain
        sbx.sandbox_id = sandbox_id
        sbx.run_calls = []
        sbx.killed = False
        return sbx


@pytest.fixture
def fake_e2b(monkeypatch):
    """Inject a fake e2b_code_interpreter module exposing _FakeSandbox."""
    _FakeSandbox.instances = []
    _FakeSandbox.next_execution = None
    _FakeSandbox.run_error = None
    _FakeSandbox.connect_calls = []

    module = types.ModuleType("e2b_code_interpreter")
    module.Sandbox = _FakeSandbox
    monkeypatch.setitem(sys.modules, "e2b_code_interpreter", module)
    return _FakeSandbox


def _init(provider, **overrides):
    config = {"api_key": "e2b_test_key"}
    config.update(overrides)
    assert provider.initialize(config) is True
    return provider


# --------------------------------------------------------------------------
# initialize / health
# --------------------------------------------------------------------------

def test_initialize_requires_api_key(fake_e2b):
    assert E2BProvider().initialize({}) is False


def test_initialize_fails_without_sdk(monkeypatch):
    # Ensure the SDK import fails even if it happens to be installed.
    monkeypatch.setitem(sys.modules, "e2b_code_interpreter", None)
    assert E2BProvider().initialize({"api_key": "e2b_test_key"}) is False


def test_initialize_success_and_health(fake_e2b):
    provider = _init(E2BProvider())
    assert provider.health_check() is True


def test_health_check_false_before_initialize(fake_e2b):
    assert E2BProvider().health_check() is False


# --------------------------------------------------------------------------
# create / destroy
# --------------------------------------------------------------------------

def test_create_instance_passes_lifetime_and_credentials(fake_e2b):
    provider = _init(E2BProvider(), timeout=600, domain="example.dev")
    instance = provider.create_instance("python")

    assert isinstance(instance, SandboxInstance)
    assert instance.provider == "e2b"
    assert instance.status == "running"
    sbx = fake_e2b.instances[-1]
    assert instance.instance_id == sbx.sandbox_id
    assert sbx.timeout == 600
    assert sbx.api_key == "e2b_test_key"
    assert sbx.domain == "example.dev"


def test_create_instance_requires_initialization(fake_e2b):
    with pytest.raises(RuntimeError, match="not initialized"):
        E2BProvider().create_instance("python")


def test_destroy_instance_kills_sandbox(fake_e2b):
    provider = _init(E2BProvider())
    instance = provider.create_instance("python")
    sbx = fake_e2b.instances[-1]

    assert provider.destroy_instance(instance.instance_id) is True
    assert sbx.killed is True
    # Second destroy is a no-op (already gone) and still reports success.
    assert provider.destroy_instance(instance.instance_id) is True


def test_destroy_instance_returns_false_on_kill_error(fake_e2b):
    provider = _init(E2BProvider())
    instance = provider.create_instance("python")
    sbx = fake_e2b.instances[-1]

    def _boom():
        raise RuntimeError("kill failed")

    sbx.kill = _boom
    assert provider.destroy_instance(instance.instance_id) is False


# --------------------------------------------------------------------------
# execute_code
# --------------------------------------------------------------------------

def test_execute_code_happy_path_with_structured_result(fake_e2b):
    import base64
    import json

    payload = base64.b64encode(
        json.dumps({"present": True, "value": 42, "type": "json"}).encode("utf-8")
    ).decode("ascii")
    fake_e2b.next_execution = _FakeExecution(
        stdout=["hello\n", f"{RESULT_MARKER_PREFIX}{payload}\n"],
        stderr=[],
    )

    provider = _init(E2BProvider())
    instance = provider.create_instance("python")
    result = provider.execute_code(instance.instance_id, "def main():\n    return 42", "python", timeout=15)

    assert isinstance(result, ExecutionResult)
    assert result.exit_code == 0
    assert "hello" in result.stdout
    # The structured-result marker line is stripped from stdout.
    assert RESULT_MARKER_PREFIX not in result.stdout
    assert result.metadata["result_present"] is True
    assert result.metadata["result_value"] == 42
    assert result.metadata["status"] == "ok"

    # Python code is wrapped and the execution timeout is forwarded.
    sbx = fake_e2b.instances[-1]
    assert sbx.run_calls[0]["language"] == "python"
    assert sbx.run_calls[0]["timeout"] == 15
    assert "main" in sbx.run_calls[0]["code"]


def test_execute_code_maps_error_to_nonzero_exit(fake_e2b):
    fake_e2b.next_execution = _FakeExecution(
        stdout=[],
        stderr=["partial\n"],
        error=_FakeError(name="ValueError", value="boom", traceback="Traceback ..."),
    )

    provider = _init(E2BProvider())
    instance = provider.create_instance("python")
    result = provider.execute_code(instance.instance_id, "raise ValueError('boom')", "python")

    assert result.exit_code == 1
    assert result.metadata["status"] == "error"
    assert "ValueError" in result.stderr
    assert "boom" in result.stderr


def test_execute_code_normalizes_nodejs_to_javascript(fake_e2b):
    fake_e2b.next_execution = _FakeExecution(stdout=["ok\n"])
    provider = _init(E2BProvider())
    instance = provider.create_instance("python")

    provider.execute_code(instance.instance_id, "function main(){return 1}", "nodejs")
    sbx = fake_e2b.instances[-1]
    assert sbx.run_calls[0]["language"] == "javascript"


def test_execute_code_runs_other_language_raw(fake_e2b):
    fake_e2b.next_execution = _FakeExecution(stdout=["bash-out\n"])
    provider = _init(E2BProvider())
    instance = provider.create_instance("python")

    raw = "echo hi"
    provider.execute_code(instance.instance_id, raw, "bash")
    sbx = fake_e2b.instances[-1]
    # Non py/js languages are passed through unwrapped.
    assert sbx.run_calls[0]["language"] == "bash"
    assert sbx.run_calls[0]["code"] == raw


def test_execute_code_timeout_raises_timeout_error(fake_e2b):
    class _SDKTimeout(Exception):
        pass

    _SDKTimeout.__name__ = "TimeoutException"
    fake_e2b.run_error = _SDKTimeout("deadline exceeded")

    provider = _init(E2BProvider())
    instance = provider.create_instance("python")
    with pytest.raises(TimeoutError):
        provider.execute_code(instance.instance_id, "while True: pass", "python", timeout=5)


def test_execute_code_other_error_raises_runtime_error(fake_e2b):
    fake_e2b.run_error = ValueError("nope")
    provider = _init(E2BProvider())
    instance = provider.create_instance("python")
    with pytest.raises(RuntimeError, match="E2B execution failed"):
        provider.execute_code(instance.instance_id, "x", "python")


def test_execute_code_rejects_nonpositive_timeout(fake_e2b):
    fake_e2b.next_execution = _FakeExecution(stdout=[])
    provider = _init(E2BProvider())
    instance = provider.create_instance("python")
    with pytest.raises(RuntimeError, match="greater than 0"):
        provider.execute_code(instance.instance_id, "x", "python", timeout=-1)


def test_execute_code_reconnects_to_unknown_instance(fake_e2b):
    fake_e2b.next_execution = _FakeExecution(stdout=["ok\n"])
    provider = _init(E2BProvider())

    result = provider.execute_code("sbx-external", "def main():\n    return 1", "python")
    assert result.exit_code == 0
    assert fake_e2b.connect_calls == ["sbx-external"]


def test_execute_code_requires_initialization(fake_e2b):
    with pytest.raises(RuntimeError, match="not initialized"):
        E2BProvider().execute_code("x", "code", "python")


# --------------------------------------------------------------------------
# schema / languages / validation
# --------------------------------------------------------------------------

def test_get_config_schema_marks_api_key_secret_and_required():
    schema = E2BProvider.get_config_schema()
    assert schema["api_key"]["required"] is True
    assert schema["api_key"]["secret"] is True
    assert "timeout" in schema
    assert "request_timeout" in schema


def test_get_supported_languages():
    langs = E2BProvider().get_supported_languages()
    assert "python" in langs
    assert "javascript" in langs


def test_validate_config():
    provider = E2BProvider()
    assert provider.validate_config({"api_key": "k"}) == (True, None)
    ok, msg = provider.validate_config({})
    assert ok is False and "api_key" in msg
    ok, msg = provider.validate_config({"api_key": "k", "timeout": 1})
    assert ok is False and "lifetime" in msg.lower()
    ok, msg = provider.validate_config({"api_key": "k", "request_timeout": 9999})
    assert ok is False and "execution timeout" in msg.lower()
