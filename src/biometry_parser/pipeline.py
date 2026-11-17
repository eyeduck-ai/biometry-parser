"""Memory-bounded execution and conservative field-level verification."""
from __future__ import annotations
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
import time
from typing import Callable

from .config import ExtractionConfig
from .aggregation import POLICY_VERSION, _effective_status, comparison_key, merge_observations
from .models import (EYE_FIELDS, FIELDS, UNITS, SCHEMA_VERSION, DataStatus, ExtractionResult, FieldResult,
                     Issue, Observation, ProviderOutput, RunRecord, Status)

_UNSAFE = {"invalid_value", "invalid_unit", "eye_unresolved", "page_identity_missing", "ambiguous_date", "report_template_unconfirmed"}

def observation_key(obs: Observation):
    return comparison_key(obs.field, obs.status, obs.value, obs.unit)

def _verify(candidates: list[Observation], required: list[str], mode: str, unit: str | None,
            pages_by_profile: dict[str, list[int]] | None = None) -> FieldResult:
    result = FieldResult(unit=unit, candidates=candidates)
    summaries = [o for o in candidates if o.role == "summary"]
    if not summaries:
        return result
    selected = [o for o in summaries if o.profile_id in required]
    scope = sorted({n for p in required for n in (pages_by_profile or {}).get(p, [])}) or None
    source = merge_observations(selected, summaries[0].field, summaries[0].eye, pages=scope)
    result.data_status = DataStatus(source.status)
    result.candidate_value = source.value if source.status == "present" else None
    by_profile = defaultdict(list)
    for candidate in summaries:
        by_profile[candidate.profile_id].append(candidate)
    keys = set()
    supporters = {}
    incomplete = False
    unreadable = False
    for profile in required:
        observations = by_profile.get(profile, [])
        valid = [o for o in observations if not (_UNSAFE & set(o.flags))]
        merged = merge_observations(valid, summaries[0].field, summaries[0].eye,
                                    pages=(pages_by_profile or {}).get(profile))
        if merged.status == "conflict":
            result.status = Status.CONFLICT
            return result
        decisive = [o for o in valid if _effective_status(o) in {"present", "not_measured", "blank", "absent"}]
        if merged.status not in {"present", "not_measured", "blank", "absent"}:
            decisive = []
        if not decisive:
            incomplete = True
        else:
            keys.add(comparison_key(summaries[0].field, merged.status, merged.value, merged.unit))
            supporters[profile] = decisive
        if any(_UNSAFE & set(o.flags) for o in observations):
            incomplete = True
            unreadable = True
        elif not decisive and merged.status == "unreadable":
            unreadable = True
    # Labels can merge evidence groups, never split a known common lineage.
    parents = {p: p for p in supporters}
    def find(p):
        while parents[p] != p:
            p = parents[p]
        return p
    for left in supporters:
        for right in supporters:
            a, b = supporters[left], supporters[right]
            def shared(attribute):
                return {getattr(o, attribute) for o in a if getattr(o, attribute)} & {getattr(o, attribute) for o in b if getattr(o, attribute)}
            if shared("family") or shared("model_hash") or shared("model_id") or shared("independence_group"):
                parents[find(right)] = find(left)
    group_count = len({find(p) for p in supporters})
    hard_keys = {key for key in keys if key[0] in {"present", "not_measured"}}
    if len(hard_keys) > 1:
        result.status = Status.CONFLICT
        result.data_status = DataStatus.CONFLICT
        result.candidate_value = None
    elif not keys:
        result.status = Status.UNREADABLE if unreadable else Status.UNVERIFIED
    elif incomplete or len(keys) > 1 or mode == "single" or group_count < 2:
        result.status = Status.UNVERIFIED
    else:
        status, value, _ = next(iter(keys))
        if status == "present":
            # Preserve the reported 180 rather than replacing it with its comparison equivalent 0.
            winning_key = next(iter(keys))
            result.value = next(o.value for valid in supporters.values() for o in valid
                                if observation_key(o) == winning_key and o.status == "present")
            # Equivalent axes (0/180) can arrive in a different run order than
            # the required profiles. Preserve the accepted representation in
            # both slots so saved results remain internally consistent.
            result.candidate_value = result.value
            result.status = Status.VERIFIED
        else:
            result.status = Status(status)
        result.verification_status = "cross_checked"
    return result

def reconcile(result: ExtractionResult, required: list[str], mode: str = "crosscheck") -> ExtractionResult:
    """Public deterministic reconciliation; never uses confidence as a vote."""
    result.metadata["merge_policy_version"] = POLICY_VERSION
    if result.schema_version != SCHEMA_VERSION:
        result.metadata.setdefault("legacy_schema_version", result.schema_version)
    result.schema_version = SCHEMA_VERSION
    observations = [o for run in result.runs for o in run.observations]
    pages_by_profile = {}
    for run in result.runs:
        pages = run.metadata.get("selected_pages")
        if pages is None:
            pages = sorted({number for numbers in run.metadata.get("routed_pages", {}).values() for number in numbers})
        if not pages:
            pages = result.metadata.get("selected_pages") or sorted({o.page_number for o in run.observations})
        pages_by_profile[run.profile_id] = pages
    failed_routes = {r.profile_id for r in result.runs if r.outcome in {"error", "unavailable"} or r.metadata.get('failed_pages')}
    identities = {o.value for o in observations if o.field == "patient_id" and o.status == "present" and o.value is not None}
    dates = {o.value for o in observations if o.field == "measurement_date" and o.status == "present" and o.value is not None}
    mixed = len(identities) > 1 or len(dates) > 1
    if mixed:
        result.issues.append(Issue(code="multiple_sessions", message="Conflicting patient identities or measurement dates; split or review the document."))
    for field in FIELDS:
        for eye in (("OD", "OS") if field in EYE_FIELDS else (None,)):
            candidates = [o for o in observations if o.field == field and o.eye == eye]
            final = _verify(candidates, required, mode, UNITS.get(field), pages_by_profile)
            if (failed_routes.intersection(required) and candidates
                    and final.status in {Status.VERIFIED, Status.NOT_MEASURED, Status.NOT_REPORTED, Status.BLANK, Status.ABSENT}):
                final.value = None
                final.status = Status.UNVERIFIED
                final.verification_status = "unverified"
            if mixed and candidates:
                final.value = None
                final.status = Status.CONFLICT
                final.data_status = DataStatus.CONFLICT
                final.candidate_value = None
                final.verification_status = "unverified"
            destination = result.scalar(field, eye)
            for attr in ("value", "unit", "status", "candidates", "data_status", "candidate_value", "verification_status"):
                setattr(destination, attr, getattr(final, attr))
    # A complete K pair must have been observed together by every required route.
    for eye in ("OD", "OS"):
        for group in ("k1", "k2", "tk1", "tk2"):
            power = result.scalar(group + "_power", eye)
            axis = result.scalar(group + "_axis", eye)
            if power.status == axis.status == Status.VERIFIED:
                for profile in required:
                    power_pages = {o.page_number for o in power.candidates if o.profile_id == profile and o.status == "present"}
                    axis_pages = {o.page_number for o in axis.candidates if o.profile_id == profile and o.status == "present"}
                    if not power_pages & axis_pages:
                        power.value = axis.value = None
                        power.status = axis.status = Status.UNVERIFIED
                        power.verification_status = axis.verification_status = "unverified"
                        result.issues.append(Issue(code="unpaired_keratometry", message="Power and axis were not observed together by every route.", eye=eye, field=group))
    scalars = [result.scalar(f, e) for f in FIELDS for e in (("OD", "OS") if f in EYE_FIELDS else (None,))]
    counts = Counter(s.status.value for s in scalars)
    result.completeness = dict(counts)
    result.metadata["source_state_counts"] = dict(Counter(s.data_status.value for s in scalars))
    outcomes = [run.outcome for run in result.runs]
    result.metadata["processing_status"] = ("not_run" if not outcomes else "unavailable" if all(o == "unavailable" for o in outcomes)
        else "failed" if all(o in {"error", "unavailable"} for o in outcomes) else "partial" if any(o != "ok" for o in outcomes) else "succeeded")
    if not observations:
        result.verification_state = "error" if all(r.outcome in {"error", "unavailable"} for r in result.runs) else "needs_review"
    elif not any(o.field in EYE_FIELDS and o.eye in {"OD", "OS"} and o.role == "summary" for o in observations):
        result.verification_state = "needs_review"
        result.issues.append(Issue(code="no_biometry", message="No eye-specific biometry summary was recognized."))
    elif any(s.data_status in {DataStatus.CONFLICT, DataStatus.UNREADABLE, DataStatus.NOT_FOUND} for s in scalars):
        result.verification_state = "needs_review"
    elif any(s.verification_status == "unverified" for s in scalars):
        result.verification_state = "unverified"
    else:
        result.verification_state = "verified"
    return result

def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def extract(source: str | Path, config: ExtractionConfig | None = None, *, workflow: str | None = None,
            progress: Callable[[dict], None] | None = None,
            env_overrides: dict[str, str] | None = None,
            pages: list[int] | None = None, page_strategy: str = "all") -> ExtractionResult:
    """Extract all pages by default, or opt into an inspectable selection plan.

    ``pages`` always means original page numbers. ``page_strategy='auto'``
    conservatively removes repeated native summaries/exact duplicate rasters;
    ``biometry_first`` starts with the richest native eye summary while
    retaining complementary/conflicting evidence and separate IOL pages.
    it cannot be combined with an explicit subset. Skipped pages and planning
    time remain in metadata, so a subset score is never a full-document audit.
    """
    from .documents import open_document, native_text_usable, ensure_page_image
    from .report_templates import inspect_report, parse_registered_page
    from .parsing import parse_page
    from .worker_client import WorkerClient
    from .providers import ProviderError
    from .resource_usage import ResourceSampler
    from .input_transforms import reviewed_page

    path = Path(source).resolve(strict=True)
    config = config or ExtractionConfig()
    if workflow is not None:
        config = ExtractionConfig.model_validate({**config.model_dump(), "workflow": workflow})
    recipe = config.workflows[config.workflow]
    result = ExtractionResult(source_sha256=_hash_file(path), workflow=config.workflow,
                              metadata={"config_sha256": config.fingerprint(), "date_order": config.date_order,
                                        "dpi": config.dpi, "prompt_version": "zeiss-1", "parser_version": "zeiss-markers-2", "pages": 0})
    def emit(event: str, **payload):
        if progress is not None:
            try:
                progress({"event": event, **payload})
            except Exception:
                # Telemetry cannot alter a clinical extraction result.
                if not any(i.code == "progress_callback_failed" for i in result.issues):
                    result.issues.append(Issue(code="progress_callback_failed", message="Progress reporting failed; extraction continued."))
    total_started = time.perf_counter()
    emit("document_start")
    if page_strategy not in {"all", "auto", "biometry_first"}:
        raise ValueError("page_strategy must be all, auto or biometry_first")
    if page_strategy in {"auto", "biometry_first"}:
        from .page_selection import plan_pages
        if pages is not None:
            raise ValueError("planned page selection cannot be combined with explicit pages")
        emit("page_planning")
        plan = plan_pages(path, strategy=page_strategy, dpi=config.dpi, max_page_pixels=config.max_page_pixels,
                          max_pages=config.max_pages, date_order=config.date_order)
        if plan["source_sha256"] != result.source_sha256:
            raise ValueError("source changed during page planning")
        result.metadata["page_plan"] = plan
        result.metadata["planning_seconds"] = plan["planning_seconds"]
        result.metadata["source_page_count"] = plan["source_page_count"]
        pages = plan["selected_pages"]
        emit("page_plan_complete", page_count=len(pages), source_page_count=plan["source_page_count"],
             elapsed_seconds=plan["planning_seconds"])
    result.metadata["page_strategy"] = page_strategy if page_strategy in {"auto", "biometry_first"} else "custom" if pages is not None else "all"
    preparation_started = time.perf_counter()
    with open_document(path, dpi=config.dpi, cache_dir=Path(config.cache_dir) if config.cache_dir else None,
                       max_page_pixels=config.max_page_pixels, max_pages=config.max_pages,
                       page_numbers=pages, render_pages=False) as page_iterator:
        page_items = list(page_iterator)  # Only metadata and file references; not pixel arrays.
        result.metadata["pages"] = len(page_items)
        if pages is None:
            result.metadata["source_page_count"] = len(page_items)
        result.metadata["selected_pages"] = [p.page_number for p in page_items]
        result.metadata["page_selection_explicit"] = pages is not None
        result.metadata["lazy_rendering"] = True
        result.metadata["rasterized_pages"] = [p.page_number for p in page_items if Path(p.image_path).is_file()]
        result.metadata["raster_render_seconds"] = 0.0
        result.metadata["render_seconds"] = time.perf_counter() - preparation_started
        result.metadata["document_preparation_seconds"] = time.perf_counter() - total_started
        emit("document_prepared", page_count=len(page_items), elapsed_seconds=result.metadata["document_preparation_seconds"])
        for route_index, profile_id in enumerate(recipe.profiles):
            profile = config.profiles[profile_id]
            run = RunRecord(profile_id=profile_id, provider_id=profile.adapter, model_id=profile.model)
            run.metadata["selected_pages"] = [page.page_number for page in page_items]
            started = time.perf_counter()
            client = None
            worker_sessions = []
            any_supported = False
            native_sampler = None
            emit("route_start", profile_id=profile_id, page_count=len(page_items))
            try:
                for page_index, page in enumerate(page_items, start=1):
                    emit("page_start", profile_id=profile_id, page_number=page.page_number,
                         page_index=page_index, page_count=len(page_items))
                    tick = time.perf_counter()
                    recognition_seconds = 0.0
                    page_resources = {}
                    native_sampler = None
                    recognition_page = page
                    transform = None
                    routing_started = time.perf_counter()
                    inspection = inspect_report(page.native_spans)
                    prefer_native = route_index == 0 and recipe.native_first
                    native_parse_seconds = 0.0
                    native_parsed = None
                    decision = {"page_number": page.page_number, "requested_adapter": profile.adapter,
                                "native_first": prefer_native, "forced_ocr": profile.adapter != "native_pdf" and not prefer_native,
                                "native_attempted": False, "native_text_usable": inspection["native_quality"]["usable"],
                                "native_parse_supported": None, "report_template": inspection,
                                "reason_codes": [], "fallback_to_recognizer": False}
                    run.metadata.setdefault("routing_decisions", []).append(decision)
                    use_native = profile.adapter == "native_pdf"
                    if use_native or (prefer_native and inspection["native_quality"]["usable"]):
                        native_sampler = ResourceSampler(os.getpid(), children=False)
                        native_sampler.start()
                        native_sampler.begin_scope()
                        decision["native_attempted"] = True
                        native_output = ProviderOutput(kind="spans", spans=page.native_spans)
                        probe_started = time.perf_counter()
                        try:
                            native_parsed = parse_registered_page(page, native_output, inspection=inspection,
                                profile_id=profile_id, model_id="pdfium-text", family="native_pdf", date_order=config.date_order)
                            decision["native_parse_supported"] = native_parsed.supported
                            calculation_intent = inspection["page_kind"] == "iol_calculation" or native_parsed.report_type in {"calculation", "iol_calculation"}
                            readable_iol_rows = [row for table in native_parsed.iol_calculations if table.eye in {"OD", "OS"}
                                                 for row in table.rows if row.iol_power is not None]
                            decision["native_summary_observation_count"] = len(native_parsed.observations)
                            decision["native_iol_table_count"] = len(native_parsed.iol_calculations)
                            decision["native_iol_readable_power_row_count"] = len(readable_iol_rows)
                            calculation_missing = calculation_intent and not readable_iol_rows
                            if calculation_missing:
                                decision["reason_codes"].append("native_calculation_rows_missing")
                            payload = (any(o.eye in {"OD", "OS"} and o.role == "summary" and
                                       _effective_status(o) in {"present", "not_measured"} for o in native_parsed.observations)
                                       or bool(readable_iol_rows))
                            damaged = (any(issue.code in {"invalid_value", "invalid_unit", "eye_unresolved", "ambiguous_date",
                                                          "normalization_failed", "identity_eye_invalid",
                                                          "iol_eye_unresolved", "invalid_iol_cell"} for issue in native_parsed.issues)
                                       or any(o.role == "summary" and o.status == "unreadable" for o in native_parsed.observations)
                                       or any(flag.endswith("_unreadable") or flag.endswith("_out_of_range")
                                              for table in native_parsed.iol_calculations for row in table.rows for flag in row.flags))
                            if profile.adapter != "native_pdf":
                                use_native = native_parsed.supported and payload and not damaged and not calculation_missing
                            decision["reason_codes"].append("explicit_native_only" if profile.adapter == "native_pdf" else
                                                           "usable_native_report" if use_native else "native_parse_requires_fallback")
                        except Exception:
                            if profile.adapter == "native_pdf":
                                raise
                            decision["reason_codes"].append("native_parse_failed")
                            use_native = False
                        native_parse_seconds = time.perf_counter() - probe_started
                    elif prefer_native:
                        decision["reason_codes"].extend(inspection["native_quality"]["reason_codes"])
                        decision["reason_codes"].append("native_text_unusable")
                    else:
                        decision["reason_codes"].append("forced_model_recognition")
                    decision["fallback_to_recognizer"] = prefer_native and not use_native
                    decision["route"] = "native_pdf" if use_native else profile.adapter
                    decision["routing_seconds"] = max(0.0, time.perf_counter() - routing_started - native_parse_seconds)
                    decision["native_probe_seconds"] = native_parse_seconds
                    if use_native:
                        output = ProviderOutput(kind="spans", spans=page.native_spans)
                        family = "native_pdf"
                        provider_id = "native_pdf"
                        model_id = "pdfium-text"
                        run.metadata["native_reader_version"] = version("pypdfium2")
                    else:
                        if native_sampler is not None:
                            decision["native_probe_resources"] = native_sampler.end_scope()
                            native_sampler.close()
                            native_sampler = None
                        if profile.options.get('input_manifest'):
                            recognition_page, transform = reviewed_page(page,result.source_sha256,profile.options['input_manifest'])
                            run.metadata.setdefault('input_transforms',[]).append(transform)
                            run.metadata['excluded_fields'] = transform['excluded_fields']
                        render_started = time.perf_counter()
                        needs_render = not Path(recognition_page.image_path).is_file()
                        recognition_page = ensure_page_image(recognition_page, dpi=config.dpi,
                                                             max_page_pixels=config.max_page_pixels)
                        render_seconds = time.perf_counter() - render_started if needs_render else 0.0
                        decision["image_render_seconds"] = render_seconds
                        if needs_render:
                            result.metadata["raster_render_seconds"] += render_seconds
                            result.metadata["render_seconds"] += render_seconds
                            if page.page_number not in result.metadata["rasterized_pages"]:
                                result.metadata["rasterized_pages"].append(page.page_number)
                        if client is None:
                            client = WorkerClient(profile, profile_id=profile_id, env_overrides=env_overrides)
                            description = client.start()
                            run.family = description.get("family", profile.adapter)
                            run.metadata["descriptor"] = description
                        # Raster recognizers do not receive the text layer or other routes' answers.
                        recognition_started = time.perf_counter()
                        try:
                            output = client.recognize(recognition_page.model_copy(update={"native_spans": []}))
                        except Exception as error:
                            failed_seconds = time.perf_counter() - tick
                            run.page_seconds.append(failed_seconds)
                            code=getattr(error,'code','provider_error')
                            run.metadata.setdefault('page_timings',[]).append({'page_number':page.page_number,
                                'elapsed_seconds':failed_seconds,'recognition_seconds':time.perf_counter()-recognition_started,
                                'outcome':'error','issue_codes':[code],'measurement_scope':'worker_process_tree',**client.last_resources})
                            run.metadata.setdefault('failed_pages',[]).append(page.page_number)
                            run.metadata['usage_complete']=False
                            if profile.options.get('continue_on_page_error',False) and code in {
                                'api_transport_error','invalid_provider_response','api_job_failed',
                                'provider_timeout','provider_failed','worker_failed'}:
                                run.issues.append(Issue(code=code,message='This page failed; remaining pages were attempted.',
                                    profile_id=profile_id,page_number=page.page_number))
                                emit('page_error',profile_id=profile_id,page_number=page.page_number,code=code)
                                if client.process is None or client.process.poll() is not None:
                                    client.close()
                                    worker_sessions.append({'startup_seconds':client.startup_seconds,
                                        'peak_rss_bytes':client.peak_rss_bytes,'cpu_seconds':client.cpu_seconds})
                                    client=None
                                continue
                            raise
                        page_resources = {**client.last_resources,'measurement_scope':'worker_process_tree'}
                        recognition_seconds = time.perf_counter() - recognition_started
                        family = run.family or profile.adapter
                        provider_id = profile.adapter
                        model_id = output.metadata.get("model_id", profile.model)
                    run.metadata.setdefault("routed_pages", {}).setdefault(provider_id, []).append(page.page_number)
                    encoded_output = json.dumps(output.model_dump(mode='json'),ensure_ascii=False)
                    if any(secret and len(secret)>=8 and secret in encoded_output for secret in (env_overrides or {}).values()):
                        raise ProviderError('credential_in_provider_output','Provider output contained a session credential.')
                    parse_started = time.perf_counter()
                    if use_native:
                        parsed = native_parsed
                    else:
                        recognized_inspection = inspect_report(output.spans) if output.kind == "spans" else None
                        if recognized_inspection is not None:
                            decision["recognized_report_template"] = recognized_inspection
                            if recognized_inspection["supported"] or not inspection["supported"]:
                                decision["report_template"] = recognized_inspection
                        active_template = decision["report_template"]
                        if active_template.get("explicit_unsupported_device"):
                            parsed = parse_registered_page(recognition_page, ProviderOutput(kind="spans", spans=[]),
                                profile_id=profile_id, model_id=model_id, family=family, date_order=config.date_order)
                            decision["reason_codes"].append("unsupported_report_device")
                        elif output.kind == "spans" and active_template["supported"]:
                            parsed = parse_registered_page(recognition_page, output, inspection=active_template,
                                profile_id=profile_id, model_id=model_id, family=family, date_order=config.date_order)
                        else:
                            parsed = parse_page(recognition_page, output, profile_id=profile_id, model_id=model_id,
                                                family=family, date_order=config.date_order)
                            if output.kind == "spans" and parsed.supported:
                                decision["reason_codes"].append("report_template_unconfirmed_layout_fallback")
                                parsed.issues.append(Issue(code="report_template_unconfirmed",
                                    message="OCR layout was parsed, but the report device could not be confirmed.",
                                    page_number=page.page_number, profile_id=profile_id))
                                for observation in parsed.observations:
                                    observation.flags.append("report_template_unconfirmed")
                    for observation in parsed.observations:
                        if "image_dash_triplet_v1" in observation.flags:
                            processing = run.metadata.setdefault("postprocessing", {})
                            processing["image_dash_triplet_v1"] = processing.get("image_dash_triplet_v1", 0) + 1
                    run.page_seconds.append(time.perf_counter() - tick)
                    if native_sampler is not None:
                        page_resources = {**native_sampler.end_scope(),'measurement_scope':'coordinator_native'}
                        native_sampler.close()
                        native_sampler = None
                    page_timing = {"page_number": page.page_number,
                        "elapsed_seconds":run.page_seconds[-1],"recognition_seconds": recognition_seconds,
                        "parsing_seconds": time.perf_counter() - parse_started + native_parse_seconds,"supported": parsed.supported,
                        "routing_seconds": decision["routing_seconds"], "image_render_seconds": decision.get("image_render_seconds", 0.0),
                        "outcome":"partial" if not parsed.supported or parsed.issues or output.issues else "ok",**page_resources}
                    for key in ('transport_seconds','preprocessing_seconds','usage','pricing','estimated_cost_usd',
                                'request_bytes','response_bytes','http_requests','billing_tier','throttle_seconds'):
                        if key in output.metadata:
                            page_timing[key] = output.metadata[key]
                    run.metadata.setdefault("page_timings", []).append({"page_number": page.page_number,
                        **page_timing})
                    if output.metadata.get('pricing'):
                        run.metadata['pricing'] = output.metadata['pricing']
                    if output.metadata.get('billing_tier'):
                        run.metadata['billing_tier'] = output.metadata['billing_tier']
                    if profile.options.get('capture_raw_output',False):
                        raw_output=output.model_dump(mode='json')
                        # Never retain an echoed session credential, even from a
                        # misbehaving custom provider. Source text stays local.
                        run.metadata.setdefault('page_outputs',[]).append({'page_number':page.page_number,'output':raw_output})
                    emit("page_complete", profile_id=profile_id, page_number=page.page_number, page_count=len(page_items), page_index=page_index,
                         elapsed_seconds=run.page_seconds[-1], supported=parsed.supported)
                    run.issues.extend(output.issues)
                    run.issues.extend(parsed.issues)
                    # Calls for irrelevant pages are still real, potentially billed calls.
                    for name in ("usage", "estimated_cost", "model_hash", "version", "prompt_sha256", "transport_seconds", "pricing"):
                        if name in output.metadata:
                            run.metadata.setdefault("page_metadata", []).append({"page": page.page_number, name: output.metadata[name]})
                    usage = output.metadata.get("usage")
                    if isinstance(usage, dict):
                        aggregated = run.metadata.setdefault("usage", {})
                        for key, value in usage.items():
                            if isinstance(value, int) and not isinstance(value, bool):
                                aggregated[key] = aggregated.get(key, 0) + value
                    elif output.kind == "claims" or output.metadata.get("usage_complete") is False:
                        run.metadata["usage_complete"] = False
                    if output.metadata.get("usage_complete") is False:
                        run.metadata["usage_complete"] = False
                    if not parsed.supported:
                        continue
                    any_supported = True
                    page_has_id = any(o.field == "patient_id" and o.status == "present" and o.value is not None for o in parsed.observations)
                    page_has_date = any(o.field == "measurement_date" and o.status == "present" and o.value is not None for o in parsed.observations)
                    # In biometry-first mode, a known calculation-only page is
                    # retained for its IOL tables and identity guards. Repeated
                    # eye summaries there are not an additional biometry source.
                    # Complementary/conflicting, uncertain and raster pages
                    # keep their observations; only proven redundant summaries
                    # are excluded from this explicitly selected scope.
                    plan_roles = next((detail.get("selection_roles", []) for detail in
                                       result.metadata.get("page_plan", {}).get("page_details", [])
                                       if detail["page_number"] == page.page_number), [])
                    collect_eye_summary = not (page_strategy == "biometry_first"
                                               and plan_roles == ["iol_calculation"])
                    decision["biometry_summary_collected"] = collect_eye_summary
                    if not collect_eye_summary:
                        run.metadata.setdefault("biometry_summary_excluded_on_iol_only_pages", []).append(page.page_number)
                        suppressed_count = sum(obs.eye is not None for obs in parsed.observations)
                        run.metadata.setdefault("suppressed_biometry_summary_claims", []).append({
                            "page_number": page.page_number, "claim_count": suppressed_count,
                            "selection_roles": plan_roles, "reason": "redundant_biometry_on_iol_only_page"})
                    for calculation in parsed.iol_calculations:
                        calculation.profile_id = profile_id
                        calculation.provider_id = provider_id
                        calculation.family = family
                        calculation.model_id = model_id
                        calculation.model_hash = output.metadata.get("model_hash")
                        if not (page_has_id and page_has_date) and "page_identity_missing" not in calculation.flags:
                            calculation.flags.append("page_identity_missing")
                        run.iol_calculations.append(calculation)
                    for obs in parsed.observations:
                        if obs.eye is not None and not collect_eye_summary:
                            continue
                        obs.profile_id = profile_id
                        obs.provider_id = provider_id
                        obs.family = family
                        obs.independence_group = profile.independence_group
                        obs.model_id = model_id
                        obs.model_hash = output.metadata.get("model_hash")
                        if obs.eye is not None and not (page_has_id and page_has_date):
                            obs.flags.append("page_identity_missing")
                        run.observations.append(obs)
                if not any_supported:
                    run.issues.append(Issue(code="unsupported_document", message="No supported ZEISS report page was recognized.", profile_id=profile_id))
                if run.issues:
                    run.outcome = "partial"
                if run.metadata.get('failed_pages') and not any_supported:
                    run.outcome='error'
            except Exception as exc:
                code = getattr(exc, "code", "provider_error")
                run.outcome = "unavailable" if code in {"dependency_unavailable", "missing_api_key", "credentials_unavailable", "missing_model", "model_not_found", "model_unavailable", "configuration_error", "invalid_configuration"} else "error"
                # Exception messages can include patient content or HTTP auth headers.
                run.issues.append(Issue(code=code, message="Recognition route could not complete; use doctor to check its configuration.", profile_id=profile_id))
                emit("route_error", profile_id=profile_id, code=code)
            finally:
                if native_sampler is not None:
                    native_sampler.close()
                run.metadata["usage_complete"] = run.metadata.get("usage_complete", True) and run.outcome not in {"error", "unavailable"}
                actual_providers = set(run.metadata.get("routed_pages", {}))
                if actual_providers == {"native_pdf"}:
                    run.provider_id = run.family = "native_pdf"
                    run.model_id = "pdfium-text"
                elif len(actual_providers) > 1:
                    run.provider_id = "native_or_" + profile.adapter
                    run.family = "mixed"
                if client is not None:
                    client.close()
                    worker_sessions.append({'startup_seconds':client.startup_seconds,
                        'peak_rss_bytes':client.peak_rss_bytes,'cpu_seconds':client.cpu_seconds})
                if worker_sessions:
                    run.metadata['worker_sessions']=worker_sessions
                    startup=[s['startup_seconds'] for s in worker_sessions]
                    cpus=[s['cpu_seconds'] for s in worker_sessions]
                    run.startup_seconds=sum(startup) if all(v is not None for v in startup) else None
                    run.peak_rss_bytes=max((s['peak_rss_bytes'] for s in worker_sessions if s['peak_rss_bytes'] is not None),default=None)
                    run.metadata['cpu_seconds']=sum(cpus) if all(v is not None for v in cpus) else None
                else:
                    timings = run.metadata.get('page_timings',[])
                    measured_cpu = [t.get('cpu_seconds') for t in timings]
                    if measured_cpu and all(v is not None for v in measured_cpu):
                        run.metadata['cpu_seconds'] = sum(measured_cpu)
                transport = [t['transport_seconds'] for t in run.metadata.get('page_timings',[]) if t.get('transport_seconds') is not None]
                if transport:
                    run.metadata['transport_seconds'] = sum(transport)
                run.elapsed_seconds = time.perf_counter() - started
                emit("route_complete", profile_id=profile_id, outcome=run.outcome, elapsed_seconds=run.elapsed_seconds)
            result.runs.append(run)
    result.metadata["elapsed_seconds"] = time.perf_counter() - total_started
    result.iol_calculations = [calculation for run in result.runs for calculation in run.iol_calculations]
    result.metadata["peak_worker_rss_bytes"] = max((r.peak_rss_bytes for r in result.runs if r.peak_rss_bytes is not None), default=None)
    result.issues.extend(issue for run in result.runs for issue in run.issues)
    result = reconcile(result, recipe.profiles, recipe.mode)
    emit("document_complete", verification_state=result.verification_state, elapsed_seconds=result.metadata["elapsed_seconds"])
    return result
