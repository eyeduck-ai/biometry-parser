"""End-to-end worker/orchestrator tests using synthetic provider contracts."""
from PIL import Image
from decimal import Decimal
import os
from biometry_parser.pipeline import extract
from biometry_parser.config import ExtractionConfig, Profile, Workflow
from biometry_parser.models import FieldClaim, ProviderOutput
from biometry_parser.providers import register_provider

class FixtureProvider:
    def __init__(self, config):
        self.options=config.get("options",{})
    def describe(self):
        return {"id":"fixture", "family":self.options.get("family","fixture"), "kind":"claims", "remote":False,
                "test_credential_configured":bool(os.environ.get("BIOMETRY_TEST_KEY"))}
    def recognize(self,page):
        if self.options.get('fail_page')==page.page_number:
            from biometry_parser.providers import ProviderError
            raise ProviderError('invalid_provider_response','fixture page failure')
        if self.options.get("skip_first") and page.page_number==1:
            return ProviderOutput(kind="claims",supported=False,metadata={"usage":{"prompt_tokens":10,"completion_tokens":2}})
        return ProviderOutput(kind="claims",supported=True,claims=[
            FieldClaim(field="patient_id",raw_value="000123"),
            FieldClaim(field="measurement_date",raw_value="2026-09-11"),
            FieldClaim(field="al",eye="OD",raw_value=self.options.get("al","23.62"),unit="mm"),
            FieldClaim(field="k1_power",eye="OD",raw_value="44.91",unit="D"),
            FieldClaim(field="k1_axis",eye="OD",raw_value="165",unit="degree"),
        ],metadata={"usage":{"prompt_tokens":10,"completion_tokens":2},
                    "usage_complete":not self.options.get("incomplete_usage",False)})

def fixture_factory(config):
    return FixtureProvider(config)

def setup(tmp_path):
    register_provider("integration-fixture",fixture_factory,replace=True)
    source=tmp_path/"synthetic.png"
    Image.new("RGB",(100,140),"white").save(source)
    config=ExtractionConfig(profiles={
        "a":Profile(adapter="integration-fixture",model="fixture-a",options={"family":"fixture-a"}),
        "b":Profile(adapter="integration-fixture",model="fixture-b",options={"family":"fixture-b"}),
    },workflows={"dual":Workflow(profiles=["a","b"])},workflow="dual")
    return source,config

def test_real_worker_pipeline_agreement_and_conflict(tmp_path):
    source,config=setup(tmp_path)
    result=extract(source,config)
    assert result.patient_id.value=="000123"
    assert result.eyes["OD"].al.value==Decimal("23.62")
    assert result.eyes["OD"].k1.axis.status.value=="verified"
    assert all(r.outcome=="ok" for r in result.runs)
    config.profiles["b"].options["al"]="23.63"
    result=extract(source,config)
    assert result.eyes["OD"].al.status.value=="conflict"
    assert result.eyes["OD"].al.value is None

def test_missing_real_gemini_key_never_sends_request_or_verifies(tmp_path,monkeypatch):
    source,config=setup(tmp_path)
    monkeypatch.delenv("BIOMETRY_TEST_ABSENT_KEY",raising=False)
    config.profiles["b"]=Profile(adapter="gemini",model="not-called",api_key_env="BIOMETRY_TEST_ABSENT_KEY")
    result=extract(source,config)
    assert result.runs[1].outcome=="unavailable"
    assert result.runs[1].issues[0].code=="credentials_unavailable"
    assert result.eyes["OD"].al.value is None
    assert result.eyes["OD"].al.candidates

def test_irrelevant_page_usage_is_counted(tmp_path):
    source,config=setup(tmp_path)
    image=Image.new("RGB",(100,140),"white")
    source=source.with_suffix(".tiff")
    image.save(source,save_all=True,append_images=[image.copy()])
    config.profiles["a"].options["skip_first"]=True
    config.workflows["one"]=Workflow(profiles=["a"],mode="single")
    result=extract(source,config,workflow="one")
    assert result.runs[0].metadata["usage"]=={"prompt_tokens":20,"completion_tokens":4}
    assert {o.page_number for o in result.runs[0].observations}=={2}

def test_progress_and_credentials_are_scoped_to_child(tmp_path,monkeypatch):
    source,config=setup(tmp_path)
    monkeypatch.delenv("BIOMETRY_TEST_KEY",raising=False)
    events=[]
    result=extract(source,config,progress=events.append,env_overrides={"BIOMETRY_TEST_KEY":"temporary-test-secret"})
    assert "BIOMETRY_TEST_KEY" not in os.environ
    assert all(r.metadata["descriptor"]["test_credential_configured"] for r in result.runs)
    assert "temporary-test-secret" not in result.model_dump_json()
    assert events[0]["event"]=="document_start"
    assert events[-1]["event"]=="document_complete"
    assert len([event for event in events if event["event"]=="page_complete"])==2
    assert result.metadata["document_preparation_seconds"]>=0

def test_progress_failure_does_not_change_extraction(tmp_path):
    source,config=setup(tmp_path)
    def fail(event):
        raise RuntimeError("sensitive callback detail")
    result=extract(source,config,progress=fail)
    assert result.eyes["OD"].al.value==Decimal("23.62")
    assert sum(i.code=="progress_callback_failed" for i in result.issues)==1
    assert "sensitive callback detail" not in result.model_dump_json()


def test_page_selection_keeps_original_evidence_and_progress_position(tmp_path):
    source,config=setup(tmp_path)
    image=Image.new("RGB",(100,140),"white")
    source=source.with_suffix(".tiff")
    image.save(source,save_all=True,append_images=[image.copy()])
    events=[]
    result=extract(source,config,pages=[2],progress=events.append)
    assert result.metadata["selected_pages"]==[2]
    assert result.metadata["pages"]==1
    assert all(o.page_number==2 for r in result.runs for o in r.observations)
    page_events=[e for e in events if e["event"]=="page_complete"]
    assert all((e["page_number"],e["page_index"],e["page_count"])==(2,1,1) for e in page_events)


def test_opt_in_auto_deduplicates_exact_input_but_keeps_original_scope(tmp_path):
    source,config=setup(tmp_path)
    image=Image.new("RGB",(100,140),"white")
    source=source.with_suffix(".tiff")
    image.save(source,save_all=True,append_images=[image.copy()])
    events=[]
    result=extract(source,config,page_strategy="auto",progress=events.append)
    assert result.metadata["selected_pages"]==[1]
    assert result.metadata["source_page_count"]==2
    assert result.metadata["page_plan"]["skipped_pages"]==[2]
    assert result.metadata["page_plan"]["recognition_calls_made"]==0
    assert result.metadata["planning_seconds"]>=0
    assert all(o.page_number==1 for run in result.runs for o in run.observations)
    assert any(e["event"]=="page_plan_complete" for e in events)
    # Default still audits both pages, including their independent recognition.
    complete=extract(source,config)
    assert complete.metadata["selected_pages"]==[1,2]


def test_auto_rejects_explicit_subset(tmp_path):
    import pytest
    source,config=setup(tmp_path)
    with pytest.raises(ValueError,match="explicit pages"):
        extract(source,config,page_strategy="auto",pages=[1])




def test_one_failed_page_can_continue_without_erasing_other_page_candidates(tmp_path):
    source,config=setup(tmp_path)
    with Image.open(source) as first:
        source=source.with_suffix('.tiff')
        first.save(source,save_all=True,append_images=[first.copy()])
    config.profiles['a'].options.update(fail_page=1,continue_on_page_error=True)
    config.workflows['single']=Workflow(profiles=['a'],mode='single')
    result=extract(source,config,workflow='single')
    run=result.runs[0]
    assert run.outcome=='partial' and run.metadata['failed_pages']==[1]
    assert [t['outcome'] for t in run.metadata['page_timings']]==['error','ok']
    assert result.scalar('al','OD').candidate_value==Decimal('23.62')
    assert result.scalar('al','OD').value is None and run.metadata['usage_complete'] is False


def test_required_route_failed_page_still_blocks_formal_crosscheck(tmp_path):
    source,config=setup(tmp_path)
    with Image.open(source) as first:
        source=source.with_suffix('.tiff')
        first.save(source,save_all=True,append_images=[first.copy()])
    config.profiles['a'].options.update(fail_page=2,continue_on_page_error=True)
    result=extract(source,config)
    assert result.scalar('al','OD').data_status.value=='present'
    assert result.scalar('al','OD').candidate_value==Decimal('23.62')
    assert result.scalar('al','OD').value is None
    assert result.scalar('al','OD').verification_status=='unverified'
