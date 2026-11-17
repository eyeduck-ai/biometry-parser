from decimal import Decimal
from pathlib import Path

from PIL import Image
import pytest

from biometry_parser.models import Box, IOLCalculation, IOLRow, Observation, ParsedPage, Span
from biometry_parser.page_selection import plan_pages, select_biometry_pages


def key(field, eye, value="1", status="present"):
    return (field, eye, status, value, "unit", ())


def identity(patient="P1", date="2026-11-15"):
    return {key("patient_id", None, patient), key("measurement_date", None, date)}


def test_coverage_beats_cct_only_page_and_keeps_complementary_cct():
    complete = {key(f, e) for e in ("OD", "OS") for f in ("al", "acd", "lt", "k1_power")}
    cct_only = {key("cct", "OD"), key("cct", "OS")}
    selected = select_biometry_pages({1:cct_only | identity(), 2:complete | identity()})
    assert selected["primary_biometry_page"] == 2
    assert selected["biometry_pages"] == [2,1]
    assert selected["native_biometry_ranking"][0]["answered_ocular_fields"] == 8


def test_bilateral_coverage_then_cct_breaks_equal_coverage_tie():
    one_eye = {key(f, "OD") for f in ("al", "acd", "lt", "cct")}
    bilateral = {key(f, e) for e in ("OD", "OS") for f in ("al", "lt")}
    assert select_biometry_pages({1:one_eye,2:bilateral})["primary_biometry_page"] == 2
    with_cct = {key(f,e) for e in ("OD", "OS") for f in ("al","cct")}
    assert select_biometry_pages({1:bilateral,2:with_cct})["primary_biometry_page"] == 2


def test_missing_rich_page_falls_back_to_both_eyes_without_cct():
    od = {key(f,"OD") for f in ("al","acd","lt")}
    os = {key(f,"OS") for f in ("al","acd","lt")}
    selected = select_biometry_pages({1:od | identity(),2:os | identity()})
    assert selected["primary_biometry_page"] == 1
    assert selected["biometry_pages"] == [1,2]
    assert selected["fallback_biometry_pages"] == [2]
    assert all(row["cct_eyes"] == 0 for row in selected["native_biometry_ranking"])


def test_conflicting_values_or_identity_are_not_hidden_by_complete_primary():
    primary = {key(f,e) for e in ("OD","OS") for f in ("al","cct")} | identity()
    for other in ({key("al","OD","2")} | identity(), {key("al","OD")} | identity("P2"),
                  {key("al","OD")} | identity(date="2026-11-14")):
        selected = select_biometry_pages({1:primary,2:other})
        assert selected["primary_biometry_page"] == 1
        assert selected["biometry_pages"] == [1,2]


def test_unknown_slots_are_not_ranked_as_answers_or_a_confirmed_absence():
    unknown = {key(f,e,None,"not_found") for e in ("OD","OS") for f in ("al","cct","acd","lt")}
    actual = {key("al","OD"),key("tk1_power","OD",None,"not_measured")}
    selected = select_biometry_pages({1:unknown,2:actual})
    assert selected["primary_biometry_page"] == 2
    assert selected["native_biometry_ranking"][0]["answered_ocular_fields"] == 2
    assert selected["native_biometry_ranking"][1]["answered_ocular_fields"] == 0


def fake_native_document(tmp_path, monkeypatch, rows, iol_pages=()):
    import pypdfium2 as pdfium
    import biometry_parser.page_selection as module
    source=tmp_path/"pages.pdf";source.write_bytes(b"fake-native-document")
    class Page:
        def __init__(self,number):self.number=number
        def get_size(self):return 100,140
        def close(self):pass
        def render(self,**kwargs):pytest.fail("native plan must not render or recognize")
    class Document(list):
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(pdfium,"PdfDocument",lambda path:Document(Page(n) for n in sorted(rows)))
    monkeypatch.setattr(module,"_native_spans",lambda page:[Span(text=str(page.number),bbox=Box(x0=.1,y0=.1,x1=.2,y1=.2))])
    monkeypatch.setattr(module,"native_text_usable",lambda page:True)
    monkeypatch.setattr(module,"inspect_report",lambda spans:{"supported":True,"page_kind":"iol_calculation" if int(spans[0].text) in iol_pages else "biometry"})
    def parse(page,*args,**kwargs):
        observations=[Observation(field="patient_id",value="P1",page_number=page.page_number),Observation(field="measurement_date",value="2026-11-15",page_number=page.page_number)]
        observations += [Observation(field=f,eye=e,value=Decimal("1"),page_number=page.page_number) for f,e in rows[page.page_number]]
        tables=[IOLCalculation(table_id="source-table",page_number=page.page_number,eye="OD",rows=[IOLRow(iol_power=Decimal("20"))])] if page.page_number in iol_pages else []
        return ParsedPage(supported=True,observations=observations,iol_calculations=tables)
    monkeypatch.setattr(module,"parse_page",parse)
    return source


def test_iol_pages_are_separate_from_primary_biometry_and_all_facts_preserved(tmp_path,monkeypatch):
    basic={(f,e) for e in ("OD","OS") for f in ("al","lt")}
    rich=basic | {("cct","OD"),("cct","OS")}
    source=fake_native_document(tmp_path,monkeypatch,{1:basic,2:rich,3:basic},iol_pages=[1])
    plan=plan_pages(source,strategy="biometry_first")
    assert plan["primary_biometry_page"] == 2
    assert plan["biometry_pages"] == [2]
    assert plan["iol_pages"] == [1]
    assert plan["selected_pages"] == [1,2] and plan["skipped_pages"] == [3]
    assert plan["page_details"][0]["selection_roles"] == ["iol_calculation"]
    assert plan["page_details"][1]["selection_roles"] == ["biometry_primary"]
    assert plan["coverage"]["native_iol_table_count"] == 1
    assert not plan["inference_used"] and plan["recognition_calls_made"] == 0


def test_raster_pages_are_not_claimed_ranked_without_recognition(tmp_path):
    first=Image.new("RGB",(40,60),"white");other=first.copy();other.putpixel((10,10),(0,0,0))
    source=tmp_path/"pages.tiff";first.save(source,save_all=True,append_images=[first.copy(),other])
    plan=plan_pages(source,strategy="biometry_first")
    assert plan["primary_biometry_page"] is None and plan["biometry_pages"] == []
    assert plan["selected_pages"] == [1,3]
    assert plan["raster_pages_requiring_recognition"] == [1,3]
    assert plan["biometry_selection_limitations"] == ["raster_pages_cannot_be_ranked_without_recognition"]
    assert not plan["inference_used"]
    with pytest.raises(ValueError):plan_pages(source,strategy="biometry_first",pages=[1])




