import hashlib
import io
import re

import pytest

from biometry_parser import model_store


def manifest(data=b'model'):
    artifact={'filename':'model.onnx','url':'https://example.test/model','sha256':hashlib.sha256(data).hexdigest()}
    return {'models':{'test':{'artifacts':{'det':artifact},'version':'test','size':'test'}}}


def test_no_implicit_download(tmp_path,monkeypatch):
    monkeypatch.setattr(model_store,'urlopen',lambda *a,**k:pytest.fail('unexpected network'))
    with pytest.raises(model_store.ModelStoreError) as error:
        model_store.resolve_model('test',tmp_path,manifest=manifest())
    assert error.value.code=='model_unavailable'


def test_packaged_manifest_contains_only_pinned_small_cpu_profile():
    models=model_store.load_manifest()['models']
    assert set(models)=={'ppocr-v6-small'}
    for name,model in models.items():
        assert model['registry_version']=='rapidocr-3.10.0'
        assert model['family']=='paddle_ppocr'
        assert set(model['artifacts'])=={'det','rec'}
        assert model['recognition_language']=='multilingual'
        for role,artifact in model['artifacts'].items():
            assert '/resolve/v3.10.0/onnx/' in artifact['url']
            assert 'latest' not in artifact['url'] and '/main/' not in artifact['url']
            assert re.fullmatch('[0-9a-f]{64}',artifact['sha256'])
            assert artifact['filename'].endswith('.onnx')




def test_hash_mismatch_is_rejected(tmp_path):
    directory=tmp_path/'test';directory.mkdir()
    (directory/'model.onnx').write_bytes(b'corrupt')
    with pytest.raises(model_store.ModelStoreError) as error:
        model_store.resolve_model('test',tmp_path,manifest=manifest())
    assert error.value.code=='model_hash_mismatch'


def test_download_atomic_and_reusable(tmp_path,monkeypatch):
    monkeypatch.setattr(model_store,'urlopen',lambda *a,**k:io.BytesIO(b'model'))
    result=model_store.download_model('test',tmp_path,manifest=manifest())
    assert result['det'].endswith('model.onnx')
    monkeypatch.setattr(model_store,'urlopen',lambda *a,**k:pytest.fail('already downloaded'))
    assert model_store.download_model('test',tmp_path,manifest=manifest())==result


def test_bad_download_never_replaces_verified_file(tmp_path,monkeypatch):
    monkeypatch.setattr(model_store,'urlopen',lambda *a,**k:io.BytesIO(b'wrong'))
    with pytest.raises(model_store.ModelStoreError):
        model_store.download_model('test',tmp_path,manifest=manifest())
    assert not (tmp_path/'test'/'model.onnx').exists()
    assert not list(tmp_path.rglob('*.part'))
