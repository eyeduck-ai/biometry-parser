from decimal import Decimal

import pytest

from biometry_parser.aggregation import POLICY_VERSION, merge_observations
from biometry_parser.models import Box, Observation


def source(value=None, *, status="present", page=1, field="al", eye="OD", role="summary", **kwargs):
    return Observation(field=field, eye=eye, value=value, status=status, page_number=page,
                       role=role, **kwargs)


def test_present_survives_unreadable_unlisted_and_absent_pages_with_complete_provenance():
    observations=[source(Decimal("23.62"),bbox=Box(x0=.1,y0=.2,x1=.2,y1=.3),raw_text="AL: 23.62 mm",confidence=.9),
                  source(status="unreadable",page=2,raw_text="AL: ?",flags=["blurred"]),
                  source(status="not_reported",page=3)]
    before=[o.model_dump() for o in observations]
    merged=merge_observations(observations,"al","OD",pages=[1,2,3,4])
    assert (merged.status,merged.value,merged.reason_code)==("present",Decimal("23.62"),"value_retained_over_missing")
    assert merged.contributing_pages==[1]
    assert merged.ignored_pages==[2,3] and merged.missing_pages==[4]
    public=merged.to_dict()
    assert public["policy_version"]==POLICY_VERSION
    assert public["value"]==23.62
    assert public["pages"][0]["observations"][0]["bbox"]==observations[0].bbox.model_dump()
    assert public["pages"][1]["observations"][0]["flags"]==["blurred"]
    assert public["pages"][1]["observations"][0]["candidate_index"]==1
    assert public["pages"][3]["decision"]=="no_evidence" and public["pages"][3]["observations"]==[]
    assert [o.model_dump() for o in observations]==before


@pytest.mark.parametrize("other",["23.63","-23.62","2362","23.6201"])
def test_actual_value_disagreement_is_not_resolved_by_majority_or_missingness(other):
    observations=[source("23.62",page=1),source("23.62",page=2),source("23.62",page=3),
                  source(other,page=4),source(status="unreadable",page=5)]
    merged=merge_observations(observations,"al","OD")
    assert merged.status=="conflict" and merged.value is None
    assert merged.reason_code=="different_values"
    assert merged.conflicting_pages==[1,2,3,4] and merged.ignored_pages==[5]


@pytest.mark.parametrize("value",["44.5","0"])
def test_positive_or_zero_value_contradicts_explicit_not_measured(value):
    merged=merge_observations([source(value,field="tk1_power"),
        source(status="not_measured",field="tk1_power",page=2),
        source(status="not_reported",field="tk1_power",page=3)],"tk1_power","OD")
    assert merged.status=="conflict" and merged.reason_code=="value_vs_not_measured"


def test_explicit_not_measured_survives_missing_pages():
    merged=merge_observations([source(status="not_measured",field="tk1_axis",raw_text="TK1: ---"),
        source(status="unreadable",field="tk1_axis",page=2)],"tk1_axis","OD",pages=[1,2,3])
    assert merged.status=="not_measured" and merged.value is None
    assert merged.reason_code=="not_measured_retained_over_missing"
    assert merged.contributing_pages==[1] and merged.missing_pages==[3]


@pytest.mark.parametrize("states,expected",[
    ([],"not_found"),(["not_reported"],"not_found"),
    (["not_reported","unreadable"],"unreadable"),(["unreadable","unreadable"],"unreadable")])
def test_no_decisive_evidence_does_not_invent_a_measurement(states,expected):
    merged=merge_observations([source(status=state,page=i+1) for i,state in enumerate(states)],"al","OD",pages=[1,2])
    assert merged.status==expected and merged.value is None and merged.contributing_pages==[]


def test_numeric_representation_and_axis_equivalence_preserve_original():
    merged=merge_observations([source("23.60"),source(Decimal("23.6"),page=2)],"al","OD")
    assert merged.status=="present" and merged.value=="23.60"
    axis=merge_observations([source("180",field="k1_axis",unit="degree"),
        source(Decimal(0),field="k1_axis",unit="degree",page=2)],"k1_axis","OD")
    assert axis.status=="present" and axis.value=="180" and axis.contributing_pages==[1,2]


def test_identity_is_a_string_and_preserves_leading_zeroes():
    merged=merge_observations([source("001",field="patient_id",eye=None),
        source("1",field="patient_id",eye=None,page=2)],"patient_id",None)
    assert merged.status=="conflict"
    retained=merge_observations([source("001",field="patient_id",eye=None),
        source(status="not_reported",field="patient_id",eye=None,page=2)],"patient_id",None)
    assert retained.status=="present" and retained.value=="001"


def test_individual_rows_and_other_eye_do_not_override_summary_and_keep_candidate_indices():
    items=[source("99",eye="OS"),source("23.62"),source("25.1",page=2,role="individual"),
           source(status="unreadable",page=3)]
    merged=merge_observations(items,"al","OD")
    assert merged.value=="23.62"
    assert merged.pages[1]["reason_code"]=="ignored_individual"
    assert merged.pages[1]["observations"][0]["candidate_index"]==1
    assert merged.pages[2]["observations"][0]["candidate_index"]==2


@pytest.mark.parametrize("invalid",[None,"NaN","inf","not a number"])
def test_invalid_present_values_are_unreadable_evidence_not_actual_conflicts(invalid):
    merged=merge_observations([source("23.62"),source(invalid,page=2)],"al","OD")
    assert merged.status=="present" and merged.value=="23.62"
    assert merged.pages[1]["observations"][0]["effective_status"]=="unreadable"


def test_no_match_is_never_confirmed_absence_or_blank():
    merged=merge_observations([],"al","OD",pages=[1])
    assert merged.status=="not_found"
    assert merged.pages[0]["data_status"]=="not_found" and merged.pages[0]["decision"]=="no_evidence"
    for state in ("absent","blank"):
        claim=source(status=state)
        result=merge_observations([claim],"al","OD",pages=[1])
        assert result.status==("not_found" if state=="absent" else "unreadable")


def test_confirmed_empty_states_require_all_selected_pages_to_be_known():
    absent=source(status="absent",state_evidence="human_review")
    blank=source(status="blank",page=2,state_evidence="confirmed_empty_region")
    assert merge_observations([absent],"al","OD",pages=[1]).status=="absent"
    mixed=merge_observations([absent,blank],"al","OD",pages=[1,2])
    assert mixed.status=="blank" and mixed.contributing_pages==[1,2]
    missing=merge_observations([absent],"al","OD",pages=[1,2])
    assert missing.status=="not_found" and missing.reason_code=="empty_evidence_incomplete_scope"
    assert missing.pages[0]["status"]=="absent" and missing.missing_pages==[2]
    unclear=merge_observations([blank,source(status="unreadable")],"al","OD",pages=[1,2])
    assert unclear.status=="unreadable"


@pytest.mark.parametrize("definite",["present","not_measured"])
def test_known_values_and_markers_survive_blank_and_absent_pages(definite):
    observations=[source("0" if definite=="present" else None,status=definite),
        source(status="blank",page=2,state_evidence="confirmed_empty_region"),
        source(status="absent",page=3,state_evidence="verified_template")]
    merged=merge_observations(observations,"al","OD",pages=[1,2,3])
    assert merged.status==definite and merged.contributing_pages==[1]
    assert merged.ignored_pages==[2,3]
