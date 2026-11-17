"""Versioned JSONL provider host. stdout is reserved for protocol messages."""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys

from .models import PROTOCOL_VERSION, PageInput
from .providers import ProviderError, create_provider, register_provider


@contextlib.contextmanager
def _quiet_backend():
    """Suppress framework output, including C-level prints, to keep PHI out of logs."""
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_copy, stderr_copy = os.dup(1), os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_copy, 1)
        os.dup2(stderr_copy, 2)
        os.close(stdout_copy)
        os.close(stderr_copy)


def main() -> int:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    os.environ.setdefault("GLOG_minloglevel", "3")
    provider = None
    for line in sys.stdin:
        request_id = None
        closing = False
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            if request.get("protocol_version") != PROTOCOL_VERSION or not isinstance(request_id, str):
                raise ProviderError("protocol_error", "Unsupported worker protocol or request identifier.")
            operation = request.get("op")
            with _quiet_backend():
                if operation == "init":
                    if provider is not None:
                        raise ProviderError("protocol_error", "Worker is already initialized.")
                    config = request["profile"]
                    if request.get("factory"):
                        module, qualname = request["factory"].split(":", 1)
                        factory = importlib.import_module(module)
                        for part in qualname.split("."):
                            factory = getattr(factory, part)
                        register_provider(config["adapter"], factory, replace=True)
                    provider = create_provider(config["adapter"], config)
                    result = provider.describe()
                elif operation == "describe" and provider is not None:
                    result = provider.describe()
                elif operation == "recognize" and provider is not None:
                    result = provider.recognize(PageInput.model_validate(request["page"])).model_dump(mode="json")
                elif operation == "close":
                    result = {}
                    closing = True
                else:
                    raise ProviderError("protocol_error", "Unknown operation or worker is not initialized.")
            response = dict(protocol_version=PROTOCOL_VERSION, request_id=request_id, ok=True, result=result)
        except ProviderError as exc:
            response = dict(protocol_version=PROTOCOL_VERSION, request_id=request_id, ok=False, error=dict(code=exc.code, message=str(exc)))
        except (ImportError, ModuleNotFoundError):
            response = dict(protocol_version=PROTOCOL_VERSION, request_id=request_id, ok=False, error=dict(code="dependency_unavailable", message="A dependency is unavailable in the selected worker environment."))
        except Exception:
            # Unknown exception strings can contain file contents, URL credentials or patient text.
            response = dict(protocol_version=PROTOCOL_VERSION, request_id=request_id, ok=False, error=dict(code="provider_failed", message="Provider failed; no values from this request were accepted."))
        sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        sys.stdout.flush()
        if closing:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
