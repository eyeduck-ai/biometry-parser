"""Resource windows use synthetic processes/providers, never OCR engines."""
from types import SimpleNamespace
import sys

from PIL import Image
import pytest

from biometry_parser.config import ExtractionConfig, Profile, Workflow
from biometry_parser.models import PageInput, ProviderOutput
from biometry_parser.pipeline import extract
from biometry_parser.providers import ProviderError
from biometry_parser.resource_usage import ResourceSampler
from biometry_parser.worker_client import WorkerClient


class MissingProcess(Exception):
    pass


class SyntheticProcess:
    def __init__(self, pid, rss=100, cpu=0, born=1):
        self.pid, self.rss, self.cpu, self.born = pid, rss, cpu, born
        self.descendants = []

    def children(self, recursive):
        assert recursive is True
        return list(self.descendants)

    def memory_info(self):
        return SimpleNamespace(rss=self.rss)

    def cpu_times(self):
        return SimpleNamespace(user=self.cpu, system=0)

    def create_time(self):
        return self.born


def install_process(monkeypatch, root):
    def process(pid):
        assert pid == root.pid
        return root
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=process, Error=MissingProcess))


def test_second_page_peak_is_measured_again_without_inheriting_first_peak(monkeypatch):
    process = SyntheticProcess(100, rss=100, cpu=10)
    install_process(monkeypatch, process)
    sampler = ResourceSampler(process.pid)
    sampler.begin_scope()
    process.rss, process.cpu = 900, 12
    sampler.sample()
    first = sampler.end_scope()
    process.rss, process.cpu = 120, 13
    sampler.begin_scope()
    process.rss, process.cpu = 180, 13.5
    second = sampler.end_scope()
    assert first["peak_rss_bytes"] == 900
    assert first["cpu_seconds"] == 2
    assert second["peak_rss_bytes"] == 180
    assert second["cpu_seconds"] == .5
    assert sampler.peak_rss_bytes == 900
    assert second["resource_samples"] >= 2


def test_process_tree_cpu_keeps_observed_exited_children_and_separates_pid_reuse(monkeypatch):
    root = SyntheticProcess(100, rss=100, cpu=3)
    child = SyntheticProcess(101, rss=500, cpu=10, born=1)
    root.descendants = [child]
    install_process(monkeypatch, root)
    sampler = ResourceSampler(root.pid)
    sampler.begin_scope()
    root.cpu, child.cpu = 3.5, 11
    sampler.sample()
    root.descendants, root.cpu = [], 4
    first = sampler.end_scope()
    assert first["peak_rss_bytes"] == 600
    assert first["cpu_seconds"] == 2
    root.descendants = [SyntheticProcess(101, rss=20, cpu=.5, born=2)]
    sampler.sample()
    assert sampler.cpu_seconds == 15.5


def test_root_only_sampling_excludes_other_children(monkeypatch):
    root = SyntheticProcess(100, rss=100, cpu=1)
    root.descendants = [SyntheticProcess(101, rss=500, cpu=10)]
    install_process(monkeypatch, root)
    sampler = ResourceSampler(root.pid, children=False)
    sampler.sample()
    assert sampler.peak_rss_bytes == 100
    assert sampler.cpu_seconds == 1


def test_missing_psutil_preserves_unknown_resource_values(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    sampler = ResourceSampler(1)
    sampler.begin_scope()
    result = sampler.end_scope()
    assert result["peak_rss_bytes"] is None and result["cpu_seconds"] is None
    assert result["resource_samples"] == 0


def test_worker_collects_page_scope_even_when_recognition_fails(monkeypatch):
    client = WorkerClient({"adapter": "native_pdf"})
    calls = []
    class Sampler:
        peak_rss_bytes = 900
        cpu_seconds = 2
        def begin_scope(self): calls.append("begin")
        def end_scope(self):
            calls.append("end")
            return {"peak_rss_bytes": 180, "cpu_seconds": .5}
    client._sampler = Sampler()
    def fail(*args, **kwargs):
        raise ProviderError("provider_timeout", "fixture timeout")
    monkeypatch.setattr(client, "_request", fail)
    with pytest.raises(ProviderError, match="fixture timeout"):
        client.recognize(PageInput(page_number=1, image_path="unused", width=10, height=10))
    assert calls == ["begin", "end"]
    assert client.last_resources["peak_rss_bytes"] == 180
    assert client.peak_rss_bytes == 900


@pytest.mark.parametrize("fail_second", [False, True])
def test_pipeline_records_page_resources_and_failed_calls_without_reusing_peak(tmp_path, monkeypatch, fail_second):
    from biometry_parser import worker_client
    source = tmp_path / "synthetic.tiff"
    image = Image.new("RGB", (20, 30), "white")
    image.save(source, save_all=True, append_images=[image.copy()])
    class FixtureClient:
        def __init__(self, *args, **kwargs):
            self.last_resources = {}
            self.startup_seconds, self.peak_rss_bytes, self.cpu_seconds = .1, 900, 3
        def start(self): return {"family": "synthetic", "remote": True}
        def recognize(self, page):
            self.last_resources = {"peak_rss_bytes": 900 if page.page_number == 1 else 120,
                                   "cpu_seconds": 1 if page.page_number == 1 else .5,
                                   "resource_samples": 2, "sampling_interval_seconds": .05}
            if fail_second and page.page_number == 2:
                raise ProviderError("provider_timeout", "fixture timeout")
            return ProviderOutput(kind="claims", supported=False, metadata={"usage": {"promptTokenCount": 10},
                "transport_seconds": .2, "request_bytes": 100, "response_bytes": 50})
        def close(self): pass
    monkeypatch.setattr(worker_client, "WorkerClient", FixtureClient)
    config = ExtractionConfig(profiles={"fixture": Profile(adapter="gemini", model="synthetic")},
        workflows={"single": Workflow(profiles=["fixture"], mode="single")}, workflow="single")
    result = extract(source, config)
    run = result.runs[0]
    first, second = run.metadata["page_timings"]
    assert first["peak_rss_bytes"] == 900 and second["peak_rss_bytes"] == 120
    assert second["measurement_scope"] == "worker_process_tree"
    assert run.peak_rss_bytes == result.metadata["peak_worker_rss_bytes"] == 900
    assert run.metadata["cpu_seconds"] == 3
    if fail_second:
        assert second["outcome"] == "error" and run.outcome == "error"
        assert run.metadata["usage_complete"] is False
    else:
        assert second["transport_seconds"] == .2
        assert run.metadata["usage"]["promptTokenCount"] == 20
        assert run.metadata["transport_seconds"] == pytest.approx(.4)
