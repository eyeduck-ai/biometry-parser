import json
import sys
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from PIL import Image

from biometry_parser.models import Box, PageInput, Span
from biometry_parser.providers import ProviderError, create_provider, list_providers, register_provider, registered_factory_spec
from biometry_parser.providers import vision
from biometry_parser.providers import local


@pytest.fixture
def page(tmp_path):
    path = tmp_path / 'page.png'
    Image.new('RGB', (100, 100), 'white').save(path)
    return PageInput(page_number=1, image_path=str(path), width=100, height=100, native_spans=[Span(text='PRIMARY ANSWER SECRET', bbox=Box(x0=0,y0=0,x1=1,y1=1))])


def claims_response(**overrides):
    claim = dict(field='al', eye='OD', raw_value='23.62', unit='mm', raw_label='AL', raw_text='AL: 23.62 mm', missing=None, flags=[])
    claim.update(overrides)
    return json.dumps(dict(supported=True, claims=[claim]))


def test_inventory_does_not_import_frameworks():
    names = {item['id'] for item in list_providers()}
    assert {'native_pdf','rapidocr','gemini','paddleocr_api'} <= names
    assert 'paddleocr' not in sys.modules and 'easyocr' not in sys.modules


def test_native_preserves_spans(page):
    result = create_provider('native_pdf').recognize(page)
    assert result.spans == page.native_spans


def test_optional_backends_never_download_without_explicit_models():
    for adapter in ['rapidocr']:
        with pytest.raises(ProviderError) as error:
            create_provider(adapter, {'options': {'cache_dir': '/nonexistent-biometry-test-models'}})
        assert error.value.code == 'model_unavailable'


@pytest.mark.parametrize('batch', [0,-1,33,True,'1',1.5])
def test_rapidocr_batch_size_is_positive_bounded_integer(batch):
    with pytest.raises(ProviderError) as error:
        create_provider('rapidocr', {'options':{'rec_batch_num':batch}})
    assert error.value.code=='invalid_configuration'


def test_rapidocr_memory_priority_defaults_are_explicit(monkeypatch):
    captured={}
    monkeypatch.setattr(local,'_model_paths',lambda config:{'det':'det.onnx','rec':'rec.onnx','hash':'fixture','version':'PP-OCRv6','size':'small'})
    monkeypatch.setitem(sys.modules,'rapidocr',SimpleNamespace(RapidOCR=lambda params:captured.update(params)))
    monkeypatch.setitem(sys.modules,'rapidocr.utils.typings',SimpleNamespace(EngineType=SimpleNamespace(ONNXRUNTIME='onnxruntime'),LangRec=lambda v:v,ModelType=lambda v:v,OCRVersion=lambda v:v))
    adapter=local.RapidOCRProvider({})
    assert captured['Rec.rec_batch_num']==1
    assert captured['EngineConfig.onnxruntime.enable_cpu_mem_arena'] is False
    assert adapter.describe()['rec_batch_num']==1




def test_custom_local_factory_cannot_cross_worker_boundary():
    register_provider('nonimportable-test', lambda config: None, replace=True)
    with pytest.raises(ProviderError, match='importable'):
        registered_factory_spec('nonimportable-test')


def test_gemini_requires_key_without_leaking(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    with pytest.raises(ProviderError) as error:
        create_provider('gemini', {'model':'configured-flash'})
    assert error.value.code == 'credentials_unavailable'


def test_gemini_visual_request_is_independent_and_strict(page, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','test-key-never-print')
    captured = {}
    def post(url, payload, headers, **kwargs):
        captured.update(url=url,payload=payload,headers=headers)
        return {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':claims_response()}]}}], 'usageMetadata':{'totalTokenCount':12}}
    monkeypatch.setattr(vision, '_post_json', post)
    output = create_provider('gemini', {'model':'configured-flash'}).recognize(page)
    assert output.claims[0].raw_value == '23.62'
    assert 'PRIMARY ANSWER SECRET' not in json.dumps(captured['payload'])
    assert captured['headers']['x-goog-api-key'] == 'test-key-never-print'
    assert 'test-key-never-print' not in output.model_dump_json()
    assert output.metadata['evidence_precision'] == 'page'


@pytest.mark.parametrize('overrides', [{'field':'invented'}, {'eye':None}, {'raw_value':23.62}, {'missing':'not_measured','raw_value':'23.62'}, {'missing':'conflict','raw_value':None}, {'bbox':{'x0':5}}])
def test_invalid_claims_are_rejected(overrides):
    with pytest.raises(ProviderError) as error:
        vision._parse_claims(claims_response(**overrides))
    assert error.value.code == 'invalid_provider_response'


@pytest.mark.parametrize('overrides', [{}, {'missing':'not_found','raw_value':None},
    {'missing':'unreadable','raw_value':None},
    {'missing':'not_measured','raw_value':None,'raw_text':'AL: ---'}])
def test_core_claim_outcomes_do_not_require_model_evidence_classification(overrides):
    supported, claims=vision._parse_claims(claims_response(**overrides))
    assert supported and len(claims)==1
    assert claims[0].missing==overrides.get('missing')
    assert claims[0].state_evidence=='legacy_unspecified'
    assert 'state_evidence' not in vision.claims_schema()['properties']['claims']['items']['properties']


def test_leading_zero_identity_is_preserved():
    supported, claims = vision._parse_claims(claims_response(field='patient_id',eye=None,raw_value='00123',unit=None))
    assert supported and claims[0].raw_value == '00123'






def test_gemini_request_spacing_is_local_and_respects_deadline(page,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','fixture-not-a-real-key')
    clock={'now':100.}
    sleeps=[];timeouts=[]
    monkeypatch.setattr(vision.time,'monotonic',lambda:clock['now'])
    def sleep(seconds):
        sleeps.append(seconds);clock['now']+=seconds
    monkeypatch.setattr(vision.time,'sleep',sleep)
    def post(*args,**kwargs):
        timeouts.append(kwargs['timeout'])
        return {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':claims_response()}]}}],
                'usageMetadata':{'promptTokenCount':1,'candidatesTokenCount':2}}
    monkeypatch.setattr(vision,'_post_json',post)
    provider=create_provider('gemini',{'model':'gemini-3.5-flash-lite','timeout_seconds':40,
        'options':{'min_request_interval_seconds':20,'retries':0}})
    provider.recognize(page)
    output=provider.recognize(page)
    assert sleeps==[20.] and output.metadata['throttle_seconds']==20
    assert timeouts==[39.,19.]




def test_http_error_redacts_response_body(monkeypatch):
    def fail(*args,**kwargs):
        raise HTTPError('https://service/?key=secret',401,'secret patient data',None,None)
    monkeypatch.setattr(vision,'urlopen',fail)
    with pytest.raises(ProviderError) as error:
        vision._post_json('https://service',{}, {},timeout=1,retries=0)
    assert str(error.value) == 'Vision endpoint returned HTTP 401.'


def test_no_retry_for_invalid_schema_response(monkeypatch):
    class Response:
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def read(self,n):return b'invalid'
    calls=[]
    monkeypatch.setattr(vision,'urlopen',lambda *args,**kwargs: calls.append(1) or Response())
    with pytest.raises(ProviderError):
        vision._post_json('https://service',{}, {},timeout=1,retries=3)
    assert len(calls)==1








