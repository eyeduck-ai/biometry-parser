"""Bounded subprocess execution and process-tree RSS measurement."""
from __future__ import annotations

import json
import importlib.util
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .models import PROTOCOL_VERSION, PageInput, ProviderOutput
from .providers import ProviderError, registered_factory_spec
from .resource_usage import ResourceSampler


class WorkerClient:
    def __init__(self, profile, profile_id: str = "", *, env_overrides: dict[str, str] | None = None):
        self.profile = profile.model_dump(mode="json") if hasattr(profile, "model_dump") else dict(profile)
        self.profile_id = profile_id
        # Credentials supplied by a local application are scoped to this child.
        # Never mutate os.environ or include secret values in profiles/results.
        self._env_overrides = dict(env_overrides or {})
        self.timeout = float(self.profile.get("timeout_seconds", 120))
        self.process: subprocess.Popen | None = None
        self.startup_seconds: float | None = None
        self.peak_rss_bytes: int | None = None
        self.cpu_seconds: float | None = None
        self.last_resources: dict = {}
        self._sampler = None
        self._responses: queue.Queue = queue.Queue()
        self._stop = threading.Event()

    def start(self) -> dict:
        if self.process is not None:
            raise ProviderError("protocol_error", "Worker is already started.")
        started = time.perf_counter()
        factory = registered_factory_spec(self.profile["adapter"])
        executable = self.profile.get("python") or sys.executable
        env = os.environ.copy()
        env.update(self._env_overrides)
        env["PYTHONIOENCODING"] = "utf-8"
        # Make source-installed plugins discoverable in another interpreter without shell interpolation.
        # Never inject the parent's site-packages: an isolated worker owns its dependencies.
        core_root = Path(__file__).resolve().parent.parent
        paths = [] if {"site-packages", "dist-packages"}.intersection(core_root.parts) else [str(core_root)]
        if factory:
            module = factory.split(":", 1)[0]
            spec = importlib.util.find_spec(module)
            if spec and spec.origin:
                origin = Path(spec.origin).resolve()
                if not {"site-packages", "dist-packages"}.intersection(origin.parts):
                    root = origin.parent
                    ascents = len(module.split(".")) if origin.name == "__init__.py" else len(module.split(".")) - 1
                    for _ in range(ascents):
                        root = root.parent
                    paths.append(str(root))
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
        self._responses = queue.Queue()
        self._stop.clear()
        try:
            self.process = subprocess.Popen([executable, "-m", "biometry_parser.worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            raise ProviderError("dependency_unavailable", "The selected Python worker could not be started.") from None
        threading.Thread(target=self._read, daemon=True).start()
        self._sampler = ResourceSampler(self.process.pid)
        self._sampler.start()
        try:
            result = self._request("init", profile=self.profile, factory=factory)
        except Exception:
            self.close()
            raise
        self.startup_seconds = time.perf_counter() - started
        return result

    def _read(self):
        try:
            assert self.process is not None and self.process.stdout is not None
            stdout = self.process.stdout
            while True:
                line = stdout.readline(32 * 1024 * 1024 + 1)
                if not line:
                    break
                if len(line) > 32 * 1024 * 1024:
                    self._responses.put(None)
                    return
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _request(self, operation: str, **payload) -> dict:
        if self.process is None or self.process.stdin is None:
            raise ProviderError("protocol_error", "Worker is not started.")
        identifier = uuid.uuid4().hex
        message = dict(protocol_version=PROTOCOL_VERSION, request_id=identifier, op=operation, **payload)
        try:
            self.process.stdin.write(json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n")
            self.process.stdin.flush()
            line = self._responses.get(timeout=self.timeout)
        except queue.Empty:
            self._kill()
            raise ProviderError("provider_timeout", "Provider exceeded the configured request time limit.") from None
        except (BrokenPipeError, OSError):
            raise ProviderError("worker_failed", "Provider worker exited unexpectedly.") from None
        if line is None:
            raise ProviderError("worker_failed", "Provider worker exited without a complete response.")
        try:
            response = json.loads(line)
            if response.get("protocol_version") != PROTOCOL_VERSION or response.get("request_id") != identifier or type(response.get("ok")) is not bool:
                raise ValueError()
            if response["ok"]:
                return response["result"]
            error = response["error"]
            raise ProviderError(error["code"], error["message"])
        except (ValueError, TypeError, KeyError):
            raise ProviderError("protocol_error", "Provider worker returned an invalid protocol response.") from None

    def recognize(self, page: PageInput) -> ProviderOutput:
        if self._sampler:
            self._sampler.begin_scope()
        try:
            result = self._request("recognize", page=page.model_dump(mode="json"))
        finally:
            self.last_resources = self._sampler.end_scope() if self._sampler else {}
            if self._sampler:
                self.peak_rss_bytes = self._sampler.peak_rss_bytes
                self.cpu_seconds = self._sampler.cpu_seconds
        try:
            return ProviderOutput.model_validate(result)
        except ValueError:
            raise ProviderError("invalid_provider_response", "Provider output failed local contract validation.") from None

    def _kill(self):
        if self.process is None:
            return
        try:
            import psutil
            process = psutil.Process(self.process.pid)
            for child in process.children(recursive=True):
                try:
                    child.kill()
                except psutil.Error:
                    pass
        except (ImportError, OSError):
            pass
        except Exception:
            pass
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=3)

    def close(self):
        self._stop.set()
        if self._sampler is not None:
            self._sampler.close()
            self.peak_rss_bytes = self._sampler.peak_rss_bytes
            self.cpu_seconds = self._sampler.cpu_seconds
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.close()
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self._kill()
                self.process.wait(timeout=3)
        if self.process.stdout:
            self.process.stdout.close()
        self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()
