"""Source facts must not be inferred from an empty model response."""
from decimal import Decimal

import pytest
from pydantic import ValidationError

from biometry_parser.models import Box, DataStatus, ExtractionResult, FieldClaim, FieldResult, Observation, PageInput
from biometry_parser.parsing import parse_page
from biometry_parser.models import ProviderOutput
from biometry_parser.providers import paddle_api, vision


def page():
    return PageInput(page_number=1, image_path="unused", width=500, height=500)


def normalized(claim):
    return parse_page(page(),ProviderOutput(kind="claims",supported=True,claims=[claim]),profile_id="test-vlm")


def test_v11_defaults_and_valid_v10_files_are_loadable():
    assert ExtractionResult(source_sha256="test",workflow="test").schema_version=="1.2"
    old=ExtractionResult.model_validate({"schema_version":"1.0","source_sha256":"test","workflow":"test",
        "patient_id":{"status":"verified","value":"000123"}})
    assert old.schema_version=="1.0"
    assert old.patient_id.value==old.patient_id.candidate_value=="000123"
    assert old.patient_id.data_status==DataStatus.PRESENT
    assert old.patient_id.verification_status=="cross_checked"


def test_legacy_ambiguous_missing_is_not_confirmed_absence_or_blank():
    result=FieldResult.model_validate({"status":"not_reported","value":None})
    assert result.data_status==DataStatus.NOT_FOUND
    original=Observation(field="gender",status="not_reported",page_number=1)
    assert original.status=="not_reported" and original.state_evidence=="legacy_unspecified"
    parsed=normalized(FieldClaim(field="gender",missing="not_reported",raw_label="Gender"))
    assert parsed.observations[0].status=="not_found"
    assert "legacy_missing_semantics" in parsed.observations[0].flags


@pytest.mark.parametrize("state",["not_found","absent","blank"])
def test_new_nonpresent_source_states_cannot_contain_a_number(state):
    with pytest.raises(ValidationError):
        Observation(field="al",eye="OD",status=state,value=0,page_number=1)
    with pytest.raises(ValidationError):
        FieldClaim(field="al",eye="OD",missing=state,raw_value="0")


@pytest.mark.parametrize("data",[
    {"status":"not_measured","value":"23.62"},
    {"status":"blank","data_status":"blank","candidate_value":"23.62"},
    {"status":"unverified","data_status":"present","candidate_value":"23.62","value":"23.62"},
    {"status":"verified","value":None},
    {"status":"verified","value":"23.62","candidate_value":"23.63"},
    {"status":"verified","value":"23.62","verification_status":"unverified"},
    {"data_status":"not_found","verification_status":"cross_checked"},
])
def test_field_result_rejects_contradictory_fact_value_and_verification(data):
    with pytest.raises(ValidationError):
        FieldResult.model_validate(data)


def test_candidate_value_is_available_without_claiming_verification():
    result=FieldResult(status="unverified",data_status="present",candidate_value=Decimal("0"))
    assert result.value is None and result.candidate_value==0
    assert result.verification_status=="unverified"
    assert result.model_dump(mode="json")["candidate_value"]==0


def test_label_only_and_no_label_have_different_unknown_states():
    labeled=normalized(FieldClaim(field="gender",raw_label="Gender",raw_text="Gender:"))
    missing=normalized(FieldClaim(field="gender"))
    assert labeled.observations[0].status=="unreadable"
    assert labeled.observations[0].state_evidence=="label_only"
    assert missing.observations[0].status=="not_found"
    assert missing.observations[0].state_evidence=="none"


def test_explicit_blank_with_label_and_region_is_only_a_provider_assertion():
    result=normalized(FieldClaim(field="gender",missing="blank",raw_label="Gender",bbox=Box(x0=.1,y0=.1,x1=.3,y1=.2)))
    observation=result.observations[0]
    assert observation.status=="blank" and observation.value is None
    assert observation.state_evidence=="provider_assertion"
    assert "blank_requires_review" in observation.flags


@pytest.mark.parametrize("proof",["human_review","verified_template","confirmed_empty_region"])
def test_provider_blank_claim_cannot_self_authorize_confirmation(proof):
    result=normalized(FieldClaim(field="gender",missing="blank",raw_label="Gender",
        bbox=Box(x0=.1,y0=.1,x1=.3,y1=.2),state_evidence=proof))
    observation=result.observations[0]
    assert observation.status=="blank"
    assert observation.state_evidence=="provider_assertion"
    assert "blank_requires_review" in observation.flags


@pytest.mark.parametrize("overrides",[{"raw_label":""},{"bbox":None},{"bbox":Box(x0=.1,y0=.1,x1=.1,y1=.1)}])
def test_blank_without_source_region_or_visible_label_is_unreadable(overrides):
    data={"field":"gender","missing":"blank","raw_label":"Gender","bbox":Box(x0=.1,y0=.1,x1=.3,y1=.2)}
    result=normalized(FieldClaim(**{**data,**overrides}))
    assert result.observations[0].status=="unreadable"
    assert any(issue.code=="blank_unconfirmed" for issue in result.issues)


def test_provider_cannot_self_authorize_confirmed_absence():
    result=normalized(FieldClaim(field="al",eye="OD",missing="absent",raw_label="AL",
        state_evidence="human_review",flags=["manually_edited"]))
    observation=result.observations[0]
    assert observation.status=="not_found"
    assert observation.state_evidence=="provider_assertion"
    assert "absence_unconfirmed" in observation.flags
    assert any(issue.code=="absence_unconfirmed" for issue in result.issues)


def test_dash_is_explicit_marker_but_zero_is_a_real_value():
    marker=normalized(FieldClaim(field="tk1_power",eye="OD",raw_value="---")).observations[0]
    value=normalized(FieldClaim(field="tk1_power",eye="OD",raw_value="0",unit="D")).observations[0]
    assert marker.status=="not_measured" and marker.state_evidence=="explicit_marker"
    assert value.status=="present" and value.value==0


def test_schema_and_prompt_distinguish_unfound_blank_and_execution_failure():
    properties=vision.claims_schema()["properties"]["claims"]["items"]["properties"]
    assert {"not_found","blank","not_measured","unreadable"} <= set(properties["missing"]["enum"])
    assert "absent" not in properties["missing"]["enum"]
    assert "conflict" not in properties["missing"]["enum"]
    assert "bbox" in properties and "state_evidence" not in properties
    assert "Not finding a field never confirms absence" in vision.PROMPT


def test_ocr_evidence_preserves_markers_and_positions_without_claiming_final_states():
    spans=paddle_api._ocr_spans({"rec_texts":["TK1", "---"],
        "rec_boxes":[[10,20,40,30],[45,20,70,30]]},page())
    assert [span.text for span in spans]==["TK1","---"]
    assert spans[1].bbox==Box(x0=.09,y0=.04,x1=.14,y1=.06)
    assert all("status" not in span.model_dump() for span in spans)


def test_explicit_service_table_empty_cell_is_not_an_empty_ocr_line():
    box=Box(x0=.1,y0=.1,x1=.9,y1=.6)
    cell=paddle_api._value_claims("gender","Gender","",None,box,empty_cell=True)[0]
    line=paddle_api._value_claims("gender","Gender","",None,box)[0]
    assert cell.missing=="blank"
    assert normalized(cell).observations[0].state_evidence=="provider_assertion"
    assert line.missing=="unreadable"


def test_blank_power_alone_does_not_fabricate_an_axis_blank():
    box=Box(x0=.1,y0=.1,x1=.9,y1=.6)
    power=paddle_api._value_claims("tk1_power","TK1 (D)","","OD",box,empty_cell=True)
    pair=paddle_api._value_claims("tk1_power","TK1 (D @ Axis)","","OD",box,empty_cell=True)
    assert [claim.field for claim in power]==["tk1_power"]
    assert {claim.field for claim in pair}=={"tk1_power","tk1_axis"}
    assert all(claim.missing=="blank" for claim in pair)
    assert all("whole_keratometry_row_blank" in claim.flags for claim in pair)
