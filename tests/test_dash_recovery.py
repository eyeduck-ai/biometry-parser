from PIL import Image, ImageDraw
import pytest

from biometry_parser.dash_recovery import METHOD, find_dash_triplet
from biometry_parser.models import Box, FieldClaim, Observation, PageInput, ProviderOutput, Span
from biometry_parser.parsing import parse_page, recover_tk_observations


LABEL = Box(x0=.05, y0=.20, x1=.20, y1=.24)


def image_with(strokes):
    image = Image.new("RGB", (500, 500), "white")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        draw.rectangle(stroke, fill="black")
    return image


def test_explicit_three_strokes_return_source_coordinates():
    with image_with([(160, 113, 167, 114), (171, 113, 178, 114), (182, 113, 189, 114)]) as image:
        evidence = find_dash_triplet(image, LABEL)
        assert evidence is not None and evidence.method == METHOD
        assert evidence.text == "---"
        assert evidence.bbox.x0 == pytest.approx(.32)
        assert evidence.bbox.x1 == pytest.approx(.38)


@pytest.mark.parametrize("strokes", [
    [],  # Blank is not "not measured".
    [(160, 113, 167, 114)],  # A minus sign is not a missing marker.
    [(160, 113, 250, 114)],  # A table rule or merged blob is not three glyphs.
    [(160, 113, 167, 114), (171, 109, 178, 118), (182, 113, 189, 114)],
    [(160, 113, 167, 114), (171, 113, 178, 114), (182, 125, 189, 126)],
    [(160, 113, 167, 114), (171, 113, 178, 114), (182, 113, 189, 114), (195, 113, 202, 114)],
])
def test_blank_minus_digits_rules_and_other_rows_are_rejected(strokes):
    with image_with(strokes) as image:
        assert find_dash_triplet(image, LABEL) is None


def test_positive_tk_number_and_axis_are_never_recovered_as_missing():
    image = image_with([(160, 113, 167, 114), (171, 113, 178, 114), (182, 113, 189, 114)])
    ImageDraw.Draw(image).text((195, 106), "44.5 D @70", fill="black")
    assert find_dash_triplet(image, LABEL) is None
    image.close()


def test_ocr_omitted_dashes_are_recovered_for_paired_tk_fields(tmp_path):
    path = tmp_path / "page.png"
    with image_with([(160, 113, 167, 114), (171, 113, 178, 114), (182, 113, 189, 114)]) as image:
        image.save(path)
    spans = [Span(text="IOLMaster 700", bbox=Box(x0=.1,y0=.9,x1=.5,y1=.92)),
             Span(text="OD",bbox=Box(x0=.05,y0=.1,x1=.1,y1=.12)),
             Span(text="TK1",bbox=LABEL)]
    page = PageInput(page_number=3, image_path=str(path), width=500, height=500)
    result = parse_page(page,ProviderOutput(kind="spans",spans=spans),profile_id="ocr")
    assert {o.field for o in result.observations} == {"tk1_power", "tk1_axis"}
    assert all(o.status == "not_measured" and o.value is None for o in result.observations)
    assert all(METHOD in o.flags and o.confidence is None for o in result.observations)
    assert all(o.page_number == 3 and o.bbox.x1 == pytest.approx(.38) for o in result.observations)


@pytest.mark.parametrize("marker", ["---", "- - -", "−−−", "———"])
def test_structured_raw_marker_counts_as_not_measured(marker):
    page = PageInput(page_number=1,image_path="unused",width=500,height=500)
    result = parse_page(page,ProviderOutput(kind="claims",supported=True,claims=[
        FieldClaim(field="tk1_power",eye="OD",raw_value=marker),
        FieldClaim(field="tk1_axis",eye="OD",raw_value=marker)]),profile_id="vlm")
    assert all(o.status == "not_measured" for o in result.observations)


def test_marker_without_eye_remains_unreadable():
    page = PageInput(page_number=1,image_path="unused",width=500,height=500)
    result = parse_page(page,ProviderOutput(kind="claims",supported=True,claims=[
        FieldClaim(field="tk1_power",raw_value="---")]),profile_id="vlm")
    assert result.observations[0].status == "unreadable"


def test_replay_keeps_original_evidence_and_positive_measurement(tmp_path):
    path=tmp_path/"page.png"
    with image_with([(160,113,167,114),(171,113,178,114),(182,113,189,114)]) as image:
        image.save(path)
    page=PageInput(page_number=1,image_path=str(path),width=500,height=500)
    original=[Observation(field="tk1_power",eye="OD",status="unreadable",page_number=1,bbox=LABEL,raw_label="TK1"),
              Observation(field="tk2_power",eye="OD",status="present",value="44.5",unit="D",page_number=1,bbox=LABEL)]
    replayed=recover_tk_observations(page,original)
    assert replayed[0].status=="not_measured" and METHOD in replayed[0].flags
    assert original[0].status=="unreadable" and original[0].flags==[]
    assert replayed[1].status=="present" and replayed[1].value=="44.5"
