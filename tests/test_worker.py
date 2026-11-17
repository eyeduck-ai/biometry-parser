import time

import pytest

from biometry_parser.models import Box, PageInput, ProviderOutput, Span
from biometry_parser.providers import ProviderError, register_provider
from biometry_parser.worker_client import WorkerClient


class SlowProvider:
    def describe(self):return {'id':'slow-test','family':'test','kind':'spans','remote':False}
    def recognize(self,page):
        time.sleep(10)
        return ProviderOutput(kind='spans')


def slow_factory(config):return SlowProvider()


class BrokenProvider:
    def describe(self):return {'id':'broken-test','family':'test','kind':'spans','remote':False}
    def recognize(self,page):raise RuntimeError('patient-secret API-credential')


def broken_factory(config):return BrokenProvider()


def test_worker_roundtrip_reuses_same_process():
    client=WorkerClient({'adapter':'native_pdf','timeout_seconds':10},profile_id='native')
    try:
        assert client.start()['family']=='native_pdf'
        pid=client.process.pid
        page=PageInput(page_number=1,image_path='unused',width=10,height=10,native_spans=[Span(text='00123',bbox=Box(x0=0,y0=0,x1=1,y1=1))])
        assert client.recognize(page).spans[0].text=='00123'
        assert client.recognize(page).spans[0].text=='00123'
        assert client.process.pid==pid
        assert client.startup_seconds is not None
    finally:client.close()
    assert client.process is None


def test_custom_factory_and_timeout_kill():
    register_provider('slow-test',slow_factory,replace=True)
    client=WorkerClient({'adapter':'slow-test','timeout_seconds':10})
    try:
        client.start()
        client.timeout=0.2
        with pytest.raises(ProviderError) as error:
            client.recognize(PageInput(page_number=1,image_path='unused',width=10,height=10))
        assert error.value.code=='provider_timeout'
        assert client.process.poll() is not None
    finally:client.close()


def test_unknown_exceptions_are_redacted():
    register_provider('broken-test',broken_factory,replace=True)
    client=WorkerClient({'adapter':'broken-test','timeout_seconds':10})
    try:
        client.start()
        with pytest.raises(ProviderError) as error:
            client.recognize(PageInput(page_number=1,image_path='unused',width=10,height=10))
        assert error.value.code=='provider_failed'
        assert 'patient-secret' not in str(error.value)
    finally:client.close()


def test_installed_wheel_does_not_inject_parent_site_packages(monkeypatch,tmp_path):
    from biometry_parser import worker_client
    parent_packages=tmp_path/"parent"/"site-packages"
    monkeypatch.setattr(worker_client,"__file__",str(parent_packages/"biometry_parser"/"worker_client.py"))
    monkeypatch.delenv("PYTHONPATH",raising=False)
    captured={}
    def capture(*args,**kwargs):
        captured.update(kwargs["env"])
        raise OSError("fixture stops before starting a process")
    monkeypatch.setattr(worker_client.subprocess,"Popen",capture)
    with pytest.raises(ProviderError):
        WorkerClient({"adapter":"native_pdf","python":"isolated-python"}).start()
    assert str(parent_packages) not in captured.get("PYTHONPATH", "")
