"""IOL source evidence tests; never assert a clinically selected implant."""
from decimal import Decimal
from pathlib import Path

import pytest

from biometry_parser.documents import open_document
from biometry_parser.iol_parsing import parse_iol_tables
from biometry_parser.models import (Box, ExtractionResult, IOLRowClaim, IOLTableClaim,
                                 PageInput, ProviderOutput, Span)
from biometry_parser.parsing import parse_page


def cell(text,x,y,width=.13):
    return Span(text=text,bbox=Box(x0=x,y0=y,x1=x+width,y1=y+.01))


def simple_page():
    spans=[cell("IOLMaster 700",.1,.95),cell("OD",.1,.2),
           cell("AL: 23.62 mm",.1,.3),cell("Target ref.: plano",.1,.33),
           cell("K Lens Example 1",.1,.42,.28),cell("Barrett Universal II",.12,.45,.25),
           cell("LF: +1.96 DF: Default",.12,.47,.25),
           cell("IOL (D)",.12,.5,.08),cell("Ref (D)",.25,.5,.08),
           cell("+19.00",.13,.53,.07),cell("-0.20",.26,.53,.07),
           cell("+18.50",.13,.55,.07),cell("+0.10",.26,.55,.07),
           cell("+18.70",.13,.57,.07),cell("Emmetropia",.24,.57,.12)]
    return PageInput(page_number=1,image_path="not-rendered-native",width=1000,height=1400,native_spans=spans)


def test_calculation_rows_and_summary_biometry_are_separate():
    page=simple_page()
    result=parse_page(page,ProviderOutput(kind="spans",spans=page.native_spans),profile_id="native",family="native_pdf")
    assert len(result.iol_calculations)==1
    table=result.iol_calculations[0]
    assert table.eye=="OD"
    assert table.lens_model=="Lens Example 1"
    assert table.formula=="Barrett Universal II"
    assert table.keratometry_source=="K"
    assert table.target_refraction==Decimal(0)
    assert [(c.name,c.value) for c in table.constants]==[("LF","+1.96"),("DF","Default")]
    assert [(row.iol_power,row.predicted_refraction) for row in table.rows]==[
        (Decimal("19.00"),Decimal("-0.20")),(Decimal("18.50"),Decimal("+0.10")),(Decimal("18.70"),None)]
    assert table.rows[-1].row_kind=="emmetropia"
    assert all(row.selection=="unknown" for row in table.rows)
    assert all(row.highlight_status=="unknown" for row in table.rows)
    assert all(obs.field!="mean_k" for obs in result.observations)
    assert [obs.value for obs in result.observations if obs.field=="al"]==[Decimal("23.62")]


def test_vlm_raw_cells_strict_and_selection_requires_visible_evidence():
    page=simple_page()
    out=ProviderOutput(kind="claims",supported=True,iol_calculations=[IOLTableClaim(
        eye="OS",lens_model="Example toric",formula="Barrett Toric",lens_type="toric",target_refraction="plano",
        rows=[IOLRowClaim(iol_power="+15.50",predicted_refraction="0.00",toric_power="2.50",axis="179",selection="marked"),
              IOLRowClaim(iol_power="+15.00",predicted_refraction="-0.35",selection="marked",selection_evidence="printed checked box beside this row"),
              IOLRowClaim(iol_power="15O",predicted_refraction="0,00",axis="270",highlight_status="highlighted")])])
    tables,issues=parse_iol_tables(page,out,profile_id="vlm",model_id="vision-test",family="test-vlm")
    table=tables[0]
    assert table.target_refraction==0
    assert table.rows[0].iol_power==Decimal("15.50")
    assert table.rows[0].axis==Decimal(179)
    assert table.rows[0].selection=="unknown"
    assert table.rows[1].selection=="marked"
    assert table.rows[2].iol_power is None
    assert table.rows[2].predicted_refraction is None
    assert table.rows[2].axis is None
    assert table.rows[2].highlight_status=="unknown"
    assert "selection_unsubstantiated" in table.rows[0].flags
    assert "axis_out_of_range" in table.rows[2].flags
    assert any(issue.code=="invalid_iol_cell" for issue in issues)
    assert table.verification_status=="unverified"
    assert parse_iol_tables(page,out.model_copy(update={"supported":False}),profile_id="vlm")==([],[])


def test_duplicate_and_conflicting_tables_are_preserved_not_flattened():
    claim=IOLTableClaim(eye="OD",lens_model="Same lens",formula="Formula one",rows=[IOLRowClaim(iol_power="20.00",predicted_refraction="-0.10")])
    out=ProviderOutput(kind="claims",supported=True,iol_calculations=[claim,claim.model_copy(update={"formula":"Formula two"}),claim])
    tables,_=parse_iol_tables(simple_page(),out,profile_id="vlm")
    assert len(tables)==3
    assert len({table.table_id for table in tables})==3
    assert [table.formula for table in tables]==["Formula one","Formula two","Formula one"]


def test_shading_bold_or_nearzero_are_not_explicit_selection_evidence():
    out=ProviderOutput(kind="claims",supported=True,iol_calculations=[IOLTableClaim(eye="OD",rows=[
        IOLRowClaim(iol_power="20.00",predicted_refraction="0.00",selection="marked",selection_evidence="gray shaded bold row",highlight_status="highlighted",highlight_evidence="printed gray row stripe"),
        IOLRowClaim(iol_power="19.50",selection="marked",selection_evidence="this is the best recommendation"),
        IOLRowClaim(iol_power="19.00",selection="marked",selection_evidence="empty checkbox, not selected"),
        IOLRowClaim(iol_power="18.50",selection="marked",selection_evidence="printed Selected label with a tick"),
    ])])
    tables,_=parse_iol_tables(simple_page(),out,profile_id="vlm")
    assert [row.selection for row in tables[0].rows]==["unknown","unknown","unknown","marked"]
    assert tables[0].rows[0].highlight_status=="highlighted"
    assert tables[0].rows[0].predicted_refraction==0
    assert "selection_unsubstantiated" in tables[0].rows[0].flags


def test_unknown_eye_is_not_inferred_from_table_position():
    out=ProviderOutput(kind="claims",supported=True,iol_calculations=[IOLTableClaim(rows=[IOLRowClaim(iol_power="20.00")])])
    tables,issues=parse_iol_tables(simple_page(),out,profile_id="vlm")
    assert tables[0].eye is None
    assert tables[0].flags==["eye_unresolved"]
    assert issues[0].code=="iol_eye_unresolved"


def test_additive_schema_loads_old_results_and_serializes_numeric_iol_cells():
    old=ExtractionResult.model_validate({"source_sha256":"test","workflow":"single"})
    assert old.iol_calculations==[]
    out=ProviderOutput(kind="claims",supported=True,iol_calculations=[IOLTableClaim(eye="OD",rows=[IOLRowClaim(iol_power="20.50",predicted_refraction="-0.10")])])
    tables,_=parse_iol_tables(simple_page(),out,profile_id="vlm")
    old.iol_calculations=tables
    data=old.model_dump(mode="json")
    assert data["iol_calculations"][0]["rows"][0]["iol_power"]==20.5
    assert data["iol_calculations"][0]["rows"][0]["predicted_refraction"]==-.1


def test_unreadable_calculation_table_headers_do_not_become_corneal_mean_k():
    page=simple_page()
    page.native_spans=[cell("ZEISS IOL CALCULATION",.1,.1,.5),cell("OD",.1,.2),
                       cell("IOL SE Res. SE Toric Power Res. Ast.",.1,.5,.7)]
    result=parse_page(page,ProviderOutput(kind="spans",spans=page.native_spans),profile_id="ocr")
    assert result.supported
    assert not result.iol_calculations
    assert all(obs.field!="mean_k" for obs in result.observations)


def test_detailed_toric_cylinder_is_not_an_extra_iol_table():
    # Two actual tables with IOL Cyl (including an OCR I/l shape confusion),
    # compound Ref headers, and adjacent signed values in one OCR text box.
    spans=[cell("IOLMaster 700",.1,.95),cell("OD",.1,.2),cell("OS",.85,.2),
           cell("K Example Toric",.08,.42,.35),cell("K Example Toric",.53,.42,.35),
           cell("Barrett Toric",.08,.45,.35),cell("Barrett Toric",.53,.45,.35)]
    for offset in [0,.45]:
        for text,x,width in [("IOL SE",.09,.045),("IOL CyI",.145,.055),("IOL axis",.21,.06),
                             ("Ref SE",.28,.05),("Ref SphRef Cyl",.34,.10),("Ref Axis",.445,.04)]:
            spans.append(cell(text,x+offset,.5,width))
        for text,x,width in [("+16.00+3.00",.09,.11),("4°",.22,.025),("-0.67",.28,.05),
                             ("-0.50-0.35",.34,.10),("94°",.455,.025)]:
            spans.append(cell(text,x+offset,.53,width))
    page=PageInput(page_number=1,image_path="none",width=1,height=1,native_spans=spans)
    tables,issues=parse_iol_tables(page,ProviderOutput(kind="spans",spans=spans),profile_id="ocr")
    assert len(tables)==2 and not issues
    assert [t.eye for t in tables]==["OD","OS"]
    for table in tables:
        assert len(table.rows)==1
        row=table.rows[0]
        assert row.iol_power==Decimal("16.00")
        assert row.predicted_refraction==Decimal("-0.67")
        assert row.toric_power==Decimal("3.00")
        assert row.axis==4


def test_diagram_integer_is_not_a_dioptric_cell_or_selected_lens():
    spans=[cell("IOLMaster 700",.1,.95),cell("OD",.1,.2),
           cell("K Lens Toric",.08,.42,.35),cell("Barrett Toric K",.08,.45,.35),
           cell("IOL SE",.08,.5,.05),cell("Res. SE",.15,.5,.05),
           cell("Toric Power",.26,.5,.09),cell("Res. Ast.",.40,.5,.07),
           cell("+15.50",.08,.53,.05),cell("-0.33",.15,.53,.05),
           cell("300",.31,.53,.02),cell("3.00",.35,.53,.03),cell("-0.36@94°",.40,.53,.07)]
    page=PageInput(page_number=1,image_path="none",width=1,height=1,native_spans=spans)
    tables,_=parse_iol_tables(page,ProviderOutput(kind="spans",spans=spans),profile_id="ocr")
    assert len(tables)==1
    row=tables[0].rows[0]
    assert row.iol_power==Decimal("15.50")
    assert row.predicted_refraction==Decimal("-0.33")
    assert row.toric_power==Decimal("3.00")
    assert row.axis is None
    assert row.selection=="unknown"


def test_skewed_secondary_columns_do_not_merge_adjacent_power_rows():
    spans=[cell("IOLMaster 700",.1,.95),cell("OD",.1,.2),
           cell("K Lens Toric",.08,.42,.35),cell("Barrett Toric K",.08,.45,.35),
           cell("IOL SE",.08,.5,.05),cell("Res. SE",.15,.5,.05),
           cell("Toric Power",.26,.5,.09),cell("Res. Ast.",.40,.5,.07),
           cell("+15.00",.08,.530,.05),cell("0.00",.15,.530,.05),
           cell("+14.50",.08,.544,.05),cell("+0.34",.15,.544,.05),
           cell("2.50",.35,.537,.03),cell("-0.68@94°",.40,.537,.07)]
    page=PageInput(page_number=1,image_path="none",width=1,height=1,native_spans=spans)
    tables,_=parse_iol_tables(page,ProviderOutput(kind="spans",spans=spans),profile_id="ocr")
    assert [(row.iol_power,row.predicted_refraction) for row in tables[0].rows]==[
        (Decimal("15.00"),Decimal("0.00")),(Decimal("14.50"),Decimal("0.34"))]


def test_toric_diagram_cannot_move_primary_refraction_to_preceding_row():
    # OCR has separate, correctly positioned primary cells. The taller diagram
    # labels bridge page-wide line grouping and used to leave the middle two
    # refractions unresolved despite their unambiguous individual baselines.
    def source(text,x0,y0,x1,y1):
        return Span(text=text,bbox=Box(x0=x0,y0=y0,x1=x1,y1=y1))
    spans=[source("IOLMaster 700",.1,.05,.3,.06),source("OS",.8,.2,.85,.21),
           source("K Example Toric",.506,.251,.63,.262),source("Barrett Toric",.509,.266,.64,.280),
           source("IOL SE",.516,.279,.563,.294),source("Res. SE",.579,.279,.637,.294),
           source("Toric Power",.692,.280,.768,.292),source("Res. Ast.",.839,.280,.896,.292)]
    for power,ref,y0,y1,ry0,ry1 in [
            ("+21.50","-0.61",.29485,.30935,.29571,.30822),
            ("+21.00","-0.28",.30935,.32386,.31049,.32300),
            ("+20.50","+0.06",.32499,.33637,.32386,.33637),
            ("+20.00","+0.39",.33750,.35200,.33864,.35115)]:
        spans.extend([source(power,.513,y0,.561,y1),source(ref,.590,ry0,.628,ry1)])
    spans.extend([
        source("225 2.25",.758,.30196,.816,.31362),source("-0.02 @",.824,.30196,.878,.31362),
        source("89°",.875,.30111,.902,.31362),source("179°",.676,.31305,.721,.33267),
        source("2.00",.783,.31675,.817,.32926),source("-0.18@",.822,.31561,.875,.33011),
        source("89°",.877,.31675,.902,.32812),source("150 1.50",.758,.33239,.816,.34376),
        source("-0.50 @",.823,.33239,.875,.34376),source("89°",.877,.33125,.902,.34376)])
    page=PageInput(page_number=1,image_path="not-rendered",width=100,height=140,native_spans=spans)
    tables,_=parse_iol_tables(page,ProviderOutput(kind="spans",spans=spans),profile_id="ocr")
    assert len(tables)==1 and tables[0].eye=="OS"
    assert [(row.iol_power,row.predicted_refraction) for row in tables[0].rows]==[
        (Decimal("21.50"),Decimal("-.61")),(Decimal("21.00"),Decimal("-.28")),
        (Decimal("20.50"),Decimal(".06")),(Decimal("20.00"),Decimal(".39"))]
    assert "row-aligned source refraction: -0.28" in tables[0].rows[1].raw_text
    assert tables[0].rows[1].bbox.x1 < .65
    assert all(row.selection=="unknown" for row in tables[0].rows)


def test_primary_alignment_requires_unique_overlapping_source_cells():
    from biometry_parser.iol_parsing import _align_primary_refractions
    from biometry_parser.parsing import _Row
    columns=[(.12,"iol","IOL"),(.25,"refraction","Ref"),(.40,"ignored","Res. Ast.")]
    claim=IOLRowClaim(iol_power="21.00")
    power=cell("+21.00",.1,.53,.05)
    # Contradictory cells at the same primary position cannot be resolved by
    # choosing the value nearest the target or by using a residual-diagram cell.
    rows=[_Row([power,cell("-0.20",.23,.53,.05),cell("+0.10",.23,.53,.05),cell("0.00",.38,.53,.05)])]
    assert _align_primary_refractions([claim],rows,columns,0,.5)[0].predicted_refraction is None
    # A refraction outside the power's vertical extent is not a matching row.
    rows=[_Row([power,cell("0.00",.23,.55,.05)])]
    assert _align_primary_refractions([claim],rows,columns,0,.5)[0].predicted_refraction is None
    # Vertically merged numbers in a single source box retain no row mapping.
    rows=[_Row([power,cell("-0.20 +0.10",.23,.53,.05)])]
    assert _align_primary_refractions([claim],rows,columns,0,.5)[0].predicted_refraction is None


def test_literal_compatible_unit_suffix_is_separated_without_value_repair():
    out=ProviderOutput(kind="claims",supported=True,iol_calculations=[IOLTableClaim(
        eye="OD",target_refraction="+0.00 D",rows=[
            IOLRowClaim(iol_power="+15.50 D",predicted_refraction="-0.33 D",toric_power="3.00D",axis="4°"),
            IOLRowClaim(iol_power="+15.00",axis="179 deg"),
            IOLRowClaim(iol_power="+14.50",axis="90 degrees"),
            IOLRowClaim(iol_power="15.00 mm",predicted_refraction="-0.33 D extra",axis="4D"),
            IOLRowClaim(iol_power="20.00 degree",axis="179° extra"),
        ])])
    tables,issues=parse_iol_tables(simple_page(),out,profile_id="vlm")
    table=tables[0]
    assert table.target_refraction==Decimal(0)
    assert "literal_unit_suffix_separated" in table.flags
    assert "+0.00 D" in table.raw_text
    first=table.rows[0]
    assert (first.iol_power,first.predicted_refraction,first.toric_power,first.axis)==(
        Decimal("15.50"),Decimal("-0.33"),Decimal("3.00"),Decimal(4))
    assert first.flags==["literal_unit_suffix_separated"]
    assert "4°" in first.raw_text and "+15.50 D" in first.raw_text
    assert table.rows[1].axis==179 and table.rows[2].axis==90
    assert table.rows[3].iol_power is None and table.rows[3].predicted_refraction is None and table.rows[3].axis is None
    assert table.rows[4].iol_power is None and table.rows[4].axis is None
    assert all(i.code=="invalid_iol_cell" for i in issues)


