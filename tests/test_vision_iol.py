import json
import pytest
from biometry_parser.providers.vision import _parse_response, _parse_claims, claims_schema, PROMPT
from biometry_parser.providers import ProviderError


def response(**changes):
    payload = {"supported":True,"claims":[],"iol_calculations":[{"eye":"OD","lens_model":"Example MX60", "formula":"Barrett Universal II", "rows":[{"iol_power":"+15.00","predicted_refraction":"-0.01","highlight_status":"highlighted","highlight_evidence":"gray row"}]}]}
    payload.update(changes)
    return json.dumps(payload)


def test_iol_vision_schema_separate_from_summary_and_no_implicit_selection():
    supported, claims, tables = _parse_response(response())
    assert supported and not claims and len(tables)==1
    assert tables[0].rows[0].iol_power == "+15.00"
    assert tables[0].rows[0].selection == "unknown"
    assert tables[0].rows[0].highlight_status == "highlighted"
    assert "iol_calculations" in claims_schema()["required"]
    assert "surgical selection" in PROMPT
    assert "predicted_refraction, not target_refraction" in PROMPT
    # Backwards compatible callers still receive two values.
    assert _parse_claims(response()) == (True, [])


@pytest.mark.parametrize("tables", [None, {}, [{"rows":[{"iol_power":15}]}], [{"rows":[{"selection":"recommended"}]}], [{"verification_status":"cross_checked"}], [{"rows":[{"bbox":{"x0":.8,"x1":.2,"y0":0,"y1":1}}]}]])
def test_iol_vision_contract_rejects_invalid_or_self_verified_data(tables):
    with pytest.raises(ProviderError):
        _parse_response(response(iol_calculations=tables))


def test_unsupported_page_cannot_contain_iol_tables():
    with pytest.raises(ProviderError):
        _parse_response(response(supported=False))


def test_legacy_summary_response_still_accepted():
    assert _parse_response(json.dumps({"supported":True,"claims":[]})) == (True,[],[])


def test_literal_unit_suffix_is_separated_without_changing_numeric_source():
    claim={"field":"k1_axis","eye":"OD","raw_value":"165°","unit":"degree","raw_text":"K1 @165°"}
    supported, claims, _ = _parse_response(json.dumps({"supported":True,"claims":[claim]}))
    assert claims[0].raw_value == "165" and claims[0].unit == "degree"
    assert claims[0].raw_text == "K1 @165°"
    assert "printed_unit_suffix_separated" in claims[0].flags
    claim["raw_value"]="165° 2"
    assert _parse_response(json.dumps({"supported":True,"claims":[claim]}))[1][0].raw_value == "165° 2"


@pytest.mark.parametrize("field,raw_value,unit",[("k1_axis","165D","degree"),("al","23.62 um","mm"),
                                               ("al","23.62mm","um")])
def test_conflicting_literal_and_declared_units_remain_unreadable(field,raw_value,unit):
    from biometry_parser.models import PageInput, ProviderOutput
    from biometry_parser.parsing import parse_page
    _, claims, _ = _parse_response(json.dumps({"supported":True,"claims":[
        {"field":field,"eye":"OD","raw_value":raw_value,"unit":unit}]}))
    assert claims[0].raw_value == raw_value and "invalid_unit" in claims[0].flags
    page=PageInput(page_number=1,image_path="unused",width=100,height=140)
    parsed=parse_page(page,ProviderOutput(kind="claims",supported=True,claims=claims),profile_id="test")
    assert parsed.observations[0].status == "unreadable" and parsed.observations[0].value is None


@pytest.mark.parametrize("raw_value,unit",[("165°","degree"),("165deg","degrees"),("554µm","um"),("43.5D","d")])
def test_equivalent_explicit_unit_spellings_can_be_separated(raw_value,unit):
    _,claims,_=_parse_response(json.dumps({"supported":True,"claims":[
        {"field":"k1_axis" if raw_value.startswith("165") else "cct" if raw_value.startswith("554") else "k1_power",
         "eye":"OD","raw_value":raw_value,"unit":unit}]}))
    assert "printed_unit_suffix_separated" in claims[0].flags and "invalid_unit" not in claims[0].flags


def test_non_lite_can_keep_server_sampling_defaults(tmp_path, monkeypatch):
    from PIL import Image
    from biometry_parser.models import PageInput
    image=tmp_path/"page.png";Image.new("RGB",(100,100),"white").save(image)
    page=PageInput(page_number=1,image_path=str(image),width=100,height=100)
    from biometry_parser.providers.vision import GeminiProvider
    from biometry_parser.providers import vision
    monkeypatch.setenv("GEMINI_API_KEY","fixture-not-a-real-key")
    captured={}
    def post(url,payload,headers,**kwargs):
        captured.update(payload)
        return {"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":response()}]}}]}
    monkeypatch.setattr(vision,"_post_json",post)
    provider=GeminiProvider({"model":"gemini-3.6-flash","options":{"omit_sampling_parameters":True}})
    provider.recognize(page)
    assert "temperature" not in captured["generationConfig"]
    assert "topP" not in captured["generationConfig"]
