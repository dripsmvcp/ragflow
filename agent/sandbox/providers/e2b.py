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

"""
E2B provider implementation.

This provider integrates with E2B Cloud (https://e2b.dev) for cloud-based
code execution in Firecracker microVMs, via the official
``e2b_code_interpreter`` SDK.

The SDK is an optional dependency: it is imported lazily inside
``initialize`` so this module always imports even when the package is not
installed. If a tenant selects the E2B provider without the SDK present,
``initialize`` logs a clear message and returns False (the same graceful
path used by the other SaaS providers).

Code is wrapped with the shared result-protocol helpers so that ``main()``
arguments and structured return values behave identically to the
self-managed and local providers.
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional

from agent.sandbox.result_protocol import (
    build_javascript_wrapper,
    build_python_wrapper,
    extract_structured_result,
)

from .base import SandboxProvider, SandboxInstance, ExecutionResult

logger = logging.getLogger(__name__)


class E2BProvider(SandboxProvider):
    """
    E2B provider implementation.

    Uses the E2B Cloud service for secure code execution in Firecracker
    microVMs. Each logical sandbox instance maps to one E2B ``Sandbox``;
    instances are tracked in-process by their E2B ``sandbox_id`` and can be
    reconnected to across calls.
    """

    def __init__(self):
        self.api_key: str = ""
        # E2B API domain; None lets the SDK use its default (e2b.dev).
        self.domain: Optional[str] = None
        # Sandbox lifetime in seconds (how long a created microVM stays alive).
        self.timeout: int = 300
        # Default per-execution request timeout in seconds.
        self.request_timeout: int = 30
        self._sandbox_cls = None
        self._instances: Dict[str, Any] = {}
        self._initialized: bool = False

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the provider with E2B credentials.

        Args:
            config: Configuration dictionary with keys:
                - api_key: E2B API key (required)
                - domain: E2B API domain (default: SDK default, e2b.dev)
                - timeout: Sandbox lifetime in seconds (default: 300)
                - request_timeout: Per-execution timeout in seconds (default: 30)

        Returns:
            True if initialization successful, False otherwise.
        """
        self.api_key = config.get("api_key", "")
        self.domain = config.get("domain") or None
        self.timeout = config.get("timeout", 300)
        self.request_timeout = config.get("request_timeout", 30)

        if not self.api_key:
            logger.error("E2B provider requires an api_key.")
            return False

        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            logger.error(
                "E2B SDK not installed. Run `pip install e2b-code-interpreter` "
                "to enable the E2B sandbox provider."
            )
            return False

        self._sandbox_cls = Sandbox
        self._initialized = True
        return True

    def _sandbox_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.domain:
            kwargs["domain"] = self.domain
        return kwargs

    def create_instance(self, template: str = "python") -> SandboxInstance:
        """
        Create a new E2B sandbox.

        Args:
            template: Programming language template (python, nodejs, ...).

        Returns:
            SandboxInstance whose instance_id is the E2B sandbox id.

        Raises:
            RuntimeError: If the provider is not initialized or creation fails.
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        language = self._normalize_language(template)

        try:
            sandbox = self._sandbox_cls(timeout=self.timeout, **self._sandbox_kwargs())
        except Exception as e:
            raise RuntimeError(f"Failed to create E2B sandbox: {e}")

        instance_id = getattr(sandbox, "sandbox_id", None) or str(uuid.uuid4())
        self._instances[instance_id] = sandbox

        return SandboxInstance(
            instance_id=instance_id,
            provider="e2b",
            status="running",
            metadata={
                "language": language,
                "domain": self.domain,
            },
        )

    def _get_sandbox(self, instance_id: str):
        """Return the live Sandbox for instance_id, reconnecting if needed."""
        sandbox = self._instances.get(instance_id)
        if sandbox is not None:
            return sandbox

        connect = getattr(self._sandbox_cls, "connect", None)
        if connect is None:
            raise RuntimeError(
                f"Unknown E2B instance {instance_id} and the installed SDK "
                "does not support reconnecting (Sandbox.connect)."
            )
        sandbox = connect(instance_id, **self._sandbox_kwargs())
        self._instances[instance_id] = sandbox
        return sandbox

    def execute_code(
        self,
        instance_id: str,
        code: str,
        language: str,
        timeout: int = 10,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute code in the E2B sandbox.

        Args:
            instance_id: E2B sandbox id returned by create_instance.
            code: Source code to execute.
            language: Programming language (python, nodejs, javascript, ...).
            timeout: Maximum execution time in seconds.
            arguments: Optional arguments dict passed to the code's main().

        Returns:
            ExecutionResult with stdout, stderr, exit_code, and metadata.

        Raises:
            RuntimeError: If the provider is not initialized or execution fails.
            TimeoutError: If execution exceeds the timeout.
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        normalized_lang = self._normalize_language(language)
        args_json = json.dumps(arguments or {}, ensure_ascii=False)

        # Wrap python/javascript so main(**arguments) runs and a structured
        # result is emitted; other languages run as-is.
        if normalized_lang == "python":
            payload, run_lang = build_python_wrapper(code, args_json), "python"
        elif normalized_lang == "javascript":
            payload, run_lang = build_javascript_wrapper(code, args_json), "javascript"
        else:
            payload, run_lang = code, normalized_lang

        exec_timeout = self.request_timeout if not timeout else int(timeout)
        if exec_timeout <= 0:
            raise RuntimeError(
                f"Execution timeout must be greater than 0 seconds, got {exec_timeout}."
            )

        sandbox = self._get_sandbox(instance_id)
        start_time = time.time()
        try:
            execution = sandbox.run_code(payload, language=run_lang, timeout=exec_timeout)
        except Exception as e:
            if "timeout" in type(e).__name__.lower() or "timeout" in str(e).lower():
                raise TimeoutError(f"Execution timed out after {exec_timeout} seconds")
            raise RuntimeError(f"E2B execution failed: {e}")

        execution_time = time.time() - start_time

        logs = getattr(execution, "logs", None)
        stdout = "".join(getattr(logs, "stdout", None) or []) if logs is not None else ""
        stderr = "".join(getattr(logs, "stderr", None) or []) if logs is not None else ""

        error = getattr(execution, "error", None)
        if error is not None:
            err_parts = [
                str(p)
                for p in (
                    getattr(error, "name", None),
                    getattr(error, "value", None),
                    getattr(error, "traceback", None),
                )
                if p
            ]
            err_text = "\n".join(err_parts)
            if err_text:
                stderr = f"{stderr}\n{err_text}" if stderr else err_text
        exit_code = 1 if error is not None else 0

        stdout, structured_result = extract_structured_result(stdout)

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time=execution_time,
            metadata={
                "instance_id": instance_id,
                "language": normalized_lang,
                "status": "error" if exit_code else "ok",
                "timeout": exec_timeout,
                "result_present": structured_result.get("present", False),
                "result_value": structured_result.get("value"),
                "result_type": structured_result.get("type"),
                "execution_count": getattr(execution, "execution_count", None),
            },
        )

    def destroy_instance(self, instance_id: str) -> bool:
        """
        Destroy an E2B sandbox.

        Args:
            instance_id: E2B sandbox id to destroy.

        Returns:
            True if the sandbox was killed (or already gone), False on error.
        """
        sandbox = self._instances.pop(instance_id, None)
        if sandbox is None:
            # Nothing tracked locally; treat as already destroyed.
            return True
        try:
            sandbox.kill()
        except Exception as e:
            logger.warning(f"Failed to kill E2B sandbox {instance_id}: {e}")
            return False
        return True

    def health_check(self) -> bool:
        """
        Check whether the provider is configured and the SDK is available.

        This is a local check (no API call) so it never consumes E2B quota;
        a bad key only surfaces on first use.

        Returns:
            True if initialized with an API key and the SDK loaded.
        """
        return bool(self._initialized and self.api_key and self._sandbox_cls is not None)

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported programming languages.

        Returns:
            List of language identifiers.
        """
        return ["python", "nodejs", "javascript", "r", "java", "bash"]

    @staticmethod
    def get_config_schema() -> Dict[str, Dict]:
        """
        Return configuration schema for the E2B provider.

        Returns:
            Dictionary mapping field names to their schema definitions.
        """
        return {
            "api_key": {
                "type": "string",
                "required": True,
                "label": "API Key",
                "placeholder": "e2b_...",
                "description": "E2B API key for authentication.",
                "secret": True,
                "scope": "runtime",
                "readonly": False,
            },
            "domain": {
                "type": "string",
                "required": False,
                "label": "API Domain",
                "placeholder": "e2b.dev",
                "description": "E2B API domain. Leave empty to use the SDK default (e2b.dev).",
                "scope": "runtime",
                "readonly": False,
            },
            "timeout": {
                "type": "integer",
                "required": False,
                "label": "Sandbox Lifetime (seconds)",
                "default": 300,
                "min": 5,
                "max": 3600,
                "description": "How long a created E2B sandbox stays alive. Unit: seconds.",
                "scope": "runtime",
                "readonly": False,
            },
            "request_timeout": {
                "type": "integer",
                "required": False,
                "label": "Execution Timeout (seconds)",
                "default": 30,
                "min": 5,
                "max": 300,
                "description": "Maximum time for a single code execution call. Unit: seconds.",
                "scope": "runtime",
                "readonly": False,
            },
        }

    def _normalize_language(self, language: str) -> str:
        """
        Normalize a language identifier to the E2B run_code format.

        Args:
            language: Language identifier (python, python3, nodejs, js, ...).

        Returns:
            Normalized language identifier.
        """
        if not language:
            return "python"

        lang_lower = language.lower()
        if lang_lower in ("python", "python3"):
            return "python"
        elif lang_lower in ("javascript", "nodejs", "js"):
            return "javascript"
        else:
            return lang_lower

    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate E2B provider configuration.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not config.get("api_key"):
            return False, "E2B api_key is required."

        timeout = config.get("timeout", 300)
        if isinstance(timeout, int) and (timeout < 5 or timeout > 3600):
            return False, "Sandbox lifetime must be between 5 and 3600 seconds."

        request_timeout = config.get("request_timeout", 30)
        if isinstance(request_timeout, int) and (request_timeout < 5 or request_timeout > 300):
            return False, "Execution timeout must be between 5 and 300 seconds."

        return True, None
