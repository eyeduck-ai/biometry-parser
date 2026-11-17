from decimal import Decimal
from pathlib import Path

import pytest

from biometry_parser.documents import open_document
from biometry_parser.models import Box, FieldClaim, PageInput, ProviderOutput, Span
from biometry_parser.parsing import parse_page


def page_with(*cells):
    spans = [Span(text=text, bbox=Box(x0=x, y0=y, x1=min(.99,x+.19), y1=y+.012)) for text,x,y in cells]
    return PageInput(page_number=1,image_path="unused",width=1000,height=1400,native_spans=spans)


def parsed(page):
    return parse_page(page, ProviderOutput(kind="spans", spans=page.native_spans), profile_id="test", family="test")


def lookup(result, field, eye=None, *, role="summary"):
    return [o for o in result.observations if o.field == field and o.eye == eye and o.role == role]


def tk_claim_output(*, claims=None, contract=True):
    return ProviderOutput(kind="claims",supported=True,
        metadata={"report_contract":"zeiss_iolmaster_700_image_claims_v2"} if contract else {},
        claims=claims if claims is not None else [FieldClaim(field="tk1_power",eye="OD",
            raw_label="TK1",raw_text="TK1: ---",missing="not_measured",
            bbox=Box(x0=.2,y0=.4,x1=.3,y1=.42))])


def test_explicit_whole_tk_item_marker_preserves_omitted_axis_state():
    page=PageInput(page_number=3,image_path="no-image",width=100,height=140)
    output=tk_claim_output()
    result=parse_page(page,output,profile_id="vision",family="vision-family")
    power=lookup(result,"tk1_power","OD")[0]
    axis=lookup(result,"tk1_axis","OD")[0]
    assert power.status==axis.status=="not_measured"
    assert axis.value is None and axis.unit=="degree"
    assert axis.state_evidence=="explicit_marker"
    assert axis.bbox==power.bbox and axis.raw_text==power.raw_text
    assert {"paired_tk_item_not_measured_v1","derived_not_model_axis","shared_item_marker_bbox"} <= set(axis.flags)
    assert axis.family==power.family=="vision-family"
    assert len(output.claims)==1  # Stored model output remains unchanged.
    from biometry_parser.pipeline import reconcile
    from biometry_parser.models import ExtractionResult,RunRecord
    final=reconcile(ExtractionResult(source_sha256="test",workflow="one",runs=[
        RunRecord(profile_id="vision",provider_id="vision",family="vision-family",observations=result.observations)]),["vision"],"single")
    assert final.scalar("tk1_axis","OD").data_status=="not_measured"
    assert final.scalar("tk1_axis","OD").verification_status=="unverified"
    assert final.scalar("tk1_axis","OD").value is None


@pytest.mark.parametrize("changes",[
    {"raw_label":"TK1 power"},{"raw_label":"TK1 (D)"},{"raw_label":"TK2"},
    {"raw_text":""},{"raw_text":"TK1: unreadable"},{"raw_text":"TK1: --- @ 90°"},
    {"raw_text":"TK1: —"},{"raw_value":"45.0","missing":None},
    {"missing":"unreadable"},{"role":"individual"},{"eye":None},
])
def test_tk_axis_missing_state_requires_whole_item_evidence(changes):
    output=tk_claim_output()
    output.claims[0]=output.claims[0].model_copy(update=changes)
    result=parse_page(PageInput(page_number=1,image_path="none",width=1,height=1),output,profile_id="vision")
    assert not any("derived_not_model_axis" in o.flags for o in result.observations)


def test_unknown_report_contract_cannot_derive_tk_axis_state():
    output=tk_claim_output(contract=False)
    empty=PageInput(page_number=1,image_path="none",width=1,height=1)
    assert not lookup(parse_page(empty,output,profile_id="vision"),"tk1_axis","OD")
    known=page_with(("IOLMaster 700",.1,.1))
    assert lookup(parse_page(known,output,profile_id="vision"),"tk1_axis","OD")[0].status=="not_measured"
    other=page_with(("IOLMaster 500",.1,.1))
    assert not lookup(parse_page(other,tk_claim_output(),profile_id="vision"),"tk1_axis","OD")


@pytest.mark.parametrize("axis",[
    FieldClaim(field="tk1_axis",eye="OD",raw_value="90",unit="degree"),
    FieldClaim(field="tk1_axis",eye="OD",missing="unreadable",raw_label="TK1 axis"),
    FieldClaim(field="tk1_axis",eye="OD",missing="not_found",raw_text="Axis"),
    FieldClaim(field="tk1_axis",eye="OD",missing="not_found",bbox=Box(x0=.2,y0=.4,x1=.3,y1=.42)),
    FieldClaim(field="tk1_axis",eye="OD",missing="not_measured",raw_text="---"),
])
def test_tk_pair_derivation_does_not_replace_substantive_axis_evidence(axis):
    output=tk_claim_output();output.claims.append(axis)
    result=parse_page(PageInput(page_number=1,image_path="none",width=1,height=1),output,profile_id="vision")
    assert len(lookup(result,"tk1_axis","OD"))==1
    assert not any("derived_not_model_axis" in o.flags for o in result.observations)


def test_tk_pair_can_supersede_evidence_free_not_found_without_overwriting_it():
    output=tk_claim_output();output.claims.append(FieldClaim(field="tk1_axis",eye="OD",missing="not_found"))
    result=parse_page(PageInput(page_number=1,image_path="none",width=1,height=1),output,profile_id="vision")
    axes=lookup(result,"tk1_axis","OD")
    assert [o.status for o in axes]==["not_found","not_measured"]
    assert len(output.claims)==2


def test_tk_pair_does_not_derive_from_contradictory_power_or_invent_a_bbox():
    page=PageInput(page_number=1,image_path="none",width=1,height=1)
    output=tk_claim_output();output.claims.append(FieldClaim(field="tk1_power",eye="OD",raw_value="45",unit="D"))
    assert not lookup(parse_page(page,output,profile_id="vision"),"tk1_axis","OD")
    output=tk_claim_output();output.claims[0].bbox=Box(x0=1,y0=1,x1=1,y1=1)
    axis=lookup(parse_page(page,output,profile_id="vision"),"tk1_axis","OD")[0]
    assert axis.bbox is None and "source_marker_bbox_unavailable" in axis.flags


def test_bilateral_aliases_and_explicit_missing_tk():
    p=page_with(("ZEISS IOLMaster 700",.1,.95), ("OD right",.1,.2), ("left OS",.8,.2),
                ("BIOMETRIC VALUES",.4,.3),
                ("Avg. K: 42.40 D",.1,.4), ("SE: 43.50 D",.6,.4),
                ("Ast. K: -2.05 D @95°",.1,.43), ("ΔK: -1.0 D @12°",.6,.43),
                ("Pupil: 7.6 mm",.1,.46), ("P: 6.9 mm",.6,.46),
                ("TK1: - - -",.1,.49), ("TK2: ---",.6,.49))
    result=parsed(p)
    assert lookup(result,"mean_k","OD")[0].value==Decimal("42.40")
    assert lookup(result,"mean_k","OS")[0].value==Decimal("43.50")
    assert lookup(result,"corneal_astigmatism","OD")[0].value==Decimal("-2.05")
    assert lookup(result,"corneal_astigmatism_axis","OD")[0].value==Decimal(95)
    assert lookup(result,"pupil","OS")[0].value==Decimal("6.9")
    for name,eye in [("tk1_power","OD"),("tk1_axis","OD"),("tk2_power","OS"),("tk2_axis","OS")]:
        assert lookup(result,name,eye)[0].status=="not_measured"


def test_monocular_page_uses_heading_for_both_columns():
    p=page_with(("IOLMaster 700",.1,.95), ("OS",.8,.2), ("Analyze",.4,.2),
                ("AL: 23.41 mm",.1,.4), ("WTW: 11.8 mm",.7,.4),
                ("K1: 45.21 D @71°",.1,.45), ("TK1: 44.30 D @70°",.7,.45))
    result=parsed(p)
    assert all(o.eye=="OS" for o in result.observations)
    assert lookup(result,"tk1_power","OS")[0].value==Decimal("44.30")
    assert lookup(result,"tk1_axis","OS")[0].value==Decimal(70)


def test_individual_repeated_se_values_are_not_summary():
    p=page_with(("IOLMaster 700",.1,.95), ("OD",.1,.2), ("Biometric values",.4,.2),
                ("SE: 45.51 D",.1,.4), ("K1: 44.91 D @165°",.1,.43),
                ("K2: 46.13 D @75°",.1,.46), ("ΔK: -1.22 D @165°",.1,.49),
                ("SE: 45.50 D ΔK: -1.23 D @165°",.1,.53),
                ("Reference image",.4,.8))
    result=parsed(p)
    assert result.report_type=="biometric"
    assert len(lookup(result,"mean_k","OD"))==1
    assert lookup(result,"mean_k","OD",role="individual")[0].value==Decimal("45.50")
    assert lookup(result,"corneal_astigmatism","OD",role="individual")[0].value==Decimal("-1.23")


def test_other_machine_cover_is_excluded():
    p=page_with(("TOPCON",.1,.1), ("AL: 26.50 mm",.1,.3), ("OD",.1,.2), ("Patient ID: 000123",.1,.15))
    assert parsed(p).supported is False
    assert parsed(p).observations==[]


def test_label_only_gender_does_not_prove_blank_and_preserves_metadata():
    p=page_with(("IOLMaster 700",.1,.95), ("OD",.1,.2),
                ("Gender",.6,.07), ("Patient ID: 000123",.1,.08),
                ("Date of calibration test: 1/1/2026",.1,.11),
                ("Date of measurement: 9/11/2026",.1,.14),
                ("Report dated 12/31/2026",.1,.97), ("AL: 26.07 mm",.1,.4))
    result=parsed(p)
    assert lookup(result,"gender")[0].status=="unreadable"
    assert lookup(result,"gender")[0].state_evidence=="label_only"
    assert lookup(result,"patient_id")[0].value=="000123"
    assert lookup(result,"measurement_date")[0].value=="2026-09-11"


def test_vlm_claims_use_same_conservative_normalization():
    p=page_with()
    output=ProviderOutput(kind="claims",supported=True,claims=[
        FieldClaim(field="cct",eye="OD",raw_value="0.554",unit="mm"),
        FieldClaim(field="al",eye="OD",raw_value="23.62"),
        FieldClaim(field="k1_axis",eye="OD",raw_value="270",unit="degree"),
        FieldClaim(field="tk1_power",eye="OD",raw_value="44.01",unit="D",flags=["manually_edited"]),
        FieldClaim(field="measurement_date",raw_value="9/11/2026"),
    ])
    result=parse_page(p,output,profile_id="vlm",date_order="auto")
    assert lookup(result,"cct","OD")[0].value==Decimal(554)
    assert lookup(result,"al","OD")[0].status=="unreadable"
    assert lookup(result,"k1_axis","OD")[0].value is None
    assert lookup(result,"measurement_date")[0].status=="unreadable"
    assert lookup(result,"tk1_power","OD")[0].flags==["manually_edited"]
    assert parse_page(p,output.model_copy(update={"supported":None}),profile_id="vlm").supported is False


def test_missing_eye_does_not_infer_od_from_left_column():
    p=page_with(("IOLMaster 700",.1,.95), ("AL: 23.62 mm",.1,.4))
    result=parsed(p)
    assert lookup(result,"al")[0].value is None
    assert lookup(result,"al")[0].status=="unreadable"
    assert any(i.code=="eye_unresolved" for i in result.issues)


def test_one_lost_header_on_bilateral_page_is_not_a_monocular_page():
    p=page_with(("IOLMaster 700",.1,.95), ("OD",.1,.2),
                ("AL: 23.62 mm",.1,.4), ("AL: 23.41 mm",.6,.4))
    result=parsed(p)
    assert len(lookup(result,"al"))==2
    assert all(o.value is None and o.status=="unreadable" for o in result.observations)


def test_reject_ambiguous_date_config_alias():
    p=page_with()
    output=ProviderOutput(kind="claims",supported=True,claims=[
        FieldClaim(field="measurement_date",raw_value="9/11/2026")])
    result=parse_page(p,output,profile_id="vlm",date_order="reject_ambiguous")
    assert lookup(result,"measurement_date")[0].status=="unreadable"
    iso=output.model_copy(update={"claims":[FieldClaim(field="measurement_date",raw_value="2026-09-11")]})
    assert lookup(parse_page(p,iso,profile_id="vlm",date_order="reject_ambiguous"),"measurement_date")[0].value=="2026-09-11"


def test_ocr_that_drops_tk_dash_glyphs_is_unreadable_not_unmeasured():
    p=page_with(("IOLMaster 700",.1,.95), ("OD",.1,.2), ("TK1",.1,.4))
    result=parsed(p)
    assert lookup(result,"tk1_power","OD")[0].status=="unreadable"
    assert lookup(result,"tk1_axis","OD")[0].status=="unreadable"


def test_comma_number_is_not_guessed_as_decimal_or_thousands():
    p=page_with()
    output=ProviderOutput(kind="claims",supported=True,claims=[
        FieldClaim(field="cct",eye="OD",raw_value="1,234",unit="um")])
    result=parse_page(p,output,profile_id="vlm")
    assert lookup(result,"cct","OD")[0].value is None
    assert lookup(result,"cct","OD")[0].status=="unreadable"


