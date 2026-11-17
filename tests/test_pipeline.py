from decimal import Decimal
import pytest
from biometry_parser.config import ExtractionConfig, Profile, Workflow
from biometry_parser.models import DataStatus, ExtractionResult, FieldResult, Observation, RunRecord, Status
from biometry_parser.pipeline import reconcile
from biometry_parser.aggregation import POLICY_VERSION

def obs(value="23.62", *, profile="a", family="pdf", field="al", eye="OD", **kwargs):
    return Observation(field=field, eye=eye, value=Decimal(value) if value is not None else None,
                       unit="mm", page_number=1, profile_id=profile, family=family, **kwargs)

def result(*observations):
    profiles = dict.fromkeys(o.profile_id for o in observations)
    return ExtractionResult(source_sha256="test", workflow="test", runs=[RunRecord(profile_id=p, provider_id=p, observations=[o for o in observations if o.profile_id==p]) for p in profiles])

def test_exact_agreement_only():
    r = reconcile(result(obs(), obs(profile="b", family="gemini")), ["a", "b"])
    assert r.eyes["OD"].al.value == Decimal("23.62")
    assert r.eyes["OD"].al.status == Status.VERIFIED
    assert r.eyes["OD"].al.data_status==DataStatus.PRESENT
    assert r.eyes["OD"].al.candidate_value==Decimal("23.62")
    assert r.eyes["OD"].al.verification_status=="cross_checked"


def test_source_value_and_processing_are_separate_from_verification():
    r=reconcile(result(obs()),["a"],mode="single")
    field=r.eyes["OD"].al
    assert field.value is None and field.status==Status.UNVERIFIED
    assert field.data_status==DataStatus.PRESENT and field.candidate_value==Decimal("23.62")
    assert field.verification_status=="unverified" and r.schema_version=="1.2"
    assert r.metadata["processing_status"]=="succeeded"
    failed=result(obs())
    failed.runs[0].outcome="error"
    failed=reconcile(failed,["a"],mode="single")
    assert failed.metadata["processing_status"]=="failed"
    assert failed.eyes["OD"].al.candidate_value==Decimal("23.62")


@pytest.mark.parametrize("state,proof",[("blank","confirmed_empty_region"),("absent","verified_template")])
def test_independent_empty_facts_can_be_verified_but_not_mistaken_for_values(state,proof):
    a=obs(None,status=state,state_evidence=proof)
    b=obs(None,status=state,state_evidence=proof,profile="b",family="gemini")
    r=reconcile(result(a,b),["a","b"])
    field=r.eyes["OD"].al
    assert field.data_status.value==state and field.status.value==state
    assert field.verification_status=="cross_checked" and field.value is None
    assert field.candidate_value is None

@pytest.mark.parametrize("second", ["23.63", "2362", "-23.62"])
def test_disagreement_never_votes(second):
    r = reconcile(result(obs(), obs(second, profile="b", family="gemini")), ["a", "b"])
    assert r.eyes["OD"].al.value is None
    assert r.eyes["OD"].al.status == Status.CONFLICT

def test_same_family_and_missing_route_do_not_verify():
    for observations in [(obs(family="paddle"), obs(profile="b", family="paddle")), (obs(),)]:
        r = reconcile(result(*observations), ["a", "b"])
        assert r.eyes["OD"].al.value is None
        assert r.eyes["OD"].al.status == Status.UNVERIFIED

def test_repeated_summary_is_not_majority_vote():
    r = reconcile(result(obs(), obs(), obs("23.61"), obs(profile="b",family="gemini")), ["a", "b"])
    assert r.eyes["OD"].al.status == Status.CONFLICT

def test_individual_measurement_not_merged():
    r = reconcile(result(obs(), obs("25", role="individual"), obs(profile="b",family="gemini")), ["a", "b"])
    assert r.eyes["OD"].al.value == Decimal("23.62")

def test_axis_equivalence_preserves_original():
    a=obs("180",field="k1_axis")
    b=obs("0",field="k1_axis",profile="b",family="gemini")
    a.unit=b.unit="degree"
    r=reconcile(result(a,b), ["a","b"])
    assert r.eyes["OD"].k1.axis.value == 180

def test_equivalent_axes_roundtrip_when_run_order_differs_from_required_profiles():
    a=obs("180",field="k1_axis")
    b=obs("0",field="k1_axis",profile="b",family="gemini")
    a.unit=b.unit="degree"
    r=reconcile(result(b,a), ["a","b"])
    field=r.eyes["OD"].k1.axis
    assert field.status==Status.VERIFIED
    assert field.value==field.candidate_value==Decimal("180")
    restored=ExtractionResult.model_validate_json(r.model_dump_json())
    assert restored.eyes["OD"].k1.axis==field


def test_missing_marks_distinct_from_absent():
    a=obs(None,status="not_measured",field="tk1_power")
    b=obs(None,status="not_measured",field="tk1_power",profile="b",family="gemini")
    a.unit=b.unit="D"
    r=reconcile(result(a,b), ["a","b"])
    assert r.eyes["OD"].tk1.power.status == Status.NOT_MEASURED
    assert r.eyes["OD"].tk2.power.status == Status.NOT_REPORTED

def test_multi_identity_prevents_all_merge():
    a=Observation(field="patient_id",value="001",page_number=1,profile_id="a",family="pdf")
    b=a.model_copy(update={"value":"002", "page_number":2})
    r=reconcile(result(a,b,obs(),obs(profile="b",family="gemini")), ["a","b"])
    assert r.eyes["OD"].al.value is None
    assert any(i.code=="multiple_sessions" for i in r.issues)

def test_unresolved_page_identity_blocks_formal_value():
    r=reconcile(result(obs(flags=["page_identity_missing"]),obs(profile="b",family="gemini")), ["a","b"])
    assert r.eyes["OD"].al.value is None

def test_config_forbids_bad_routes_and_bounds():
    with pytest.raises(ValueError):
        ExtractionConfig(workflows={"bad":Workflow(profiles=["nonexistent"])},workflow="bad")
    with pytest.raises(ValueError):
        Workflow(profiles=["a","a"])
    with pytest.raises(ValueError):
        ExtractionConfig(dpi=2000)

def test_leading_zero_identity_survives_worker_json_roundtrip():
    candidate = Observation(field="patient_id", value="00123", page_number=1)
    assert Observation.model_validate_json(candidate.model_dump_json()).value == "00123"
    final = FieldResult(value="00123", status="verified")
    assert FieldResult.model_validate_json(final.model_dump_json()).value == "00123"

def test_incomplete_required_run_cannot_verify_earlier_pages():
    r = result(obs(),obs(profile="b",family="gemini"))
    r.runs[-1].outcome = "error"
    r = reconcile(r,["a","b"])
    assert r.eyes["OD"].al.status == Status.UNVERIFIED
    assert r.eyes["OD"].al.value is None

def test_same_hash_cannot_be_split_by_group_labels():
    a=obs(family="runtime_a", model_hash="identical", independence_group="a")
    b=obs(profile="b", family="runtime_b", model_hash="identical", independence_group="b")
    assert reconcile(result(a,b),["a","b"]).eyes["OD"].al.value is None

def test_extra_nonrequired_candidate_never_becomes_formal_value():
    r=reconcile(result(obs("99", profile="extra"), obs(), obs(profile="b",family="gemini")),["a","b"])
    assert r.eyes["OD"].al.value == Decimal("23.62")

def test_metadata_only_report_is_not_ready_biometry():
    a=Observation(field="patient_id",value="001",page_number=1,profile_id="a",family="pdf")
    b=a.model_copy(update={"profile_id":"b","family":"gemini"})
    r=reconcile(result(a,b),["a","b"])
    assert r.patient_id.value=="001"
    assert r.verification_state=="needs_review"
    assert any(i.code=="no_biometry" for i in r.issues)


def test_crosspage_missingness_does_not_block_independent_verification():
    unreadable=obs(None,status="unreadable").model_copy(update={"page_number":2})
    absent=obs(None,status="not_reported").model_copy(update={"page_number":3})
    r=reconcile(result(obs(),unreadable,absent,obs(profile="b",family="gemini")),["a","b"])
    assert r.eyes["OD"].al.status==Status.VERIFIED and r.eyes["OD"].al.value==Decimal("23.62")
    assert len(r.eyes["OD"].al.candidates)==4
    assert r.metadata["merge_policy_version"]==POLICY_VERSION


@pytest.mark.parametrize("state",["unreadable","not_reported"])
def test_required_model_with_only_unknown_evidence_does_not_conflict_or_verify(state):
    r=reconcile(result(obs(),obs(None,status=state,profile="b",family="gemini")),["a","b"])
    assert r.eyes["OD"].al.status==Status.UNVERIFIED and r.eyes["OD"].al.value is None


def test_explicit_not_measured_survives_unknown_page_but_conflicts_with_actual_value():
    a=obs(None,status="not_measured",field="tk1_power")
    unreadable=a.model_copy(update={"status":"unreadable","page_number":2})
    b=a.model_copy(update={"profile_id":"b","family":"gemini"})
    r=reconcile(result(a,unreadable,b),["a","b"])
    assert r.eyes["OD"].tk1.power.status==Status.NOT_MEASURED
    conflicting=a.model_copy(update={"status":"present","value":Decimal("44.5"),"page_number":3})
    r=reconcile(result(a,unreadable,conflicting,b),["a","b"])
    assert r.eyes["OD"].tk1.power.status==Status.CONFLICT


def test_unsafe_evidence_still_blocks_formal_acceptance_even_with_good_sibling():
    unsafe=obs(None,status="unreadable",flags=["page_identity_missing"]).model_copy(update={"page_number":2})
    r=reconcile(result(obs(),unsafe,obs(profile="b",family="gemini")),["a","b"])
    assert r.eyes["OD"].al.status==Status.UNVERIFIED and r.eyes["OD"].al.value is None


def test_merged_k_power_and_axis_still_need_a_common_page_per_required_route():
    observations=[]
    for profile,family in [("a","pdf"),("b","gemini")]:
        observations.extend([obs("44.5",profile=profile,family=family,field="k1_power").model_copy(update={"unit":"D"}),
            obs("70",profile=profile,family=family,field="k1_axis").model_copy(update={"unit":"degree","page_number":2})])
    r=reconcile(result(*observations),["a","b"])
    assert r.eyes["OD"].k1.power.status==r.eyes["OD"].k1.axis.status==Status.UNVERIFIED
    assert any(i.code=="unpaired_keratometry" for i in r.issues)


def test_failed_required_route_cannot_formally_confirm_not_measured():
    a=obs(None,status="not_measured",field="tk1_power")
    b=a.model_copy(update={"profile_id":"b","family":"gemini"})
    r=result(a,b)
    r.runs[-1].outcome="error"
    merged=reconcile(r,["a","b"])
    assert merged.eyes["OD"].tk1.power.status==Status.UNVERIFIED
    assert len(merged.eyes["OD"].tk1.power.candidates)==2
