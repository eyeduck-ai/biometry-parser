from pathlib import Path

from PIL import Image
import pytest

from biometry_parser.documents import _display_box, native_text_usable, open_document, ensure_page_image
from biometry_parser.models import Box, PageInput, Span


def test_image_frames_are_lazy_and_cleaned(tmp_path):
    path = tmp_path / "two.tiff"
    first, second = Image.new("RGB", (120, 80), "white"), Image.new("RGB", (80, 120), "black")
    first.save(path, save_all=True, append_images=[second])
    first.close()
    second.close()
    with open_document(path, cache_dir=tmp_path) as pages:
        one = next(pages)
        assert (one.width, one.height, one.page_number) == (120, 80, 1)
        assert Path(one.image_path).exists()
        assert not (Path(one.image_path).parent / "page-0002.png").exists()
        two = next(pages)
        assert (two.width, two.height, two.page_number) == (80, 120, 2)
        assert Path(one.image_path).exists()
        with pytest.raises(StopIteration):
            next(pages)
    assert not Path(one.image_path).exists()
    assert not Path(two.image_path).exists()


@pytest.mark.parametrize("extension,format",[("png","PNG"),("jpg","JPEG"),("tif","TIFF"),("bmp","BMP"),("webp","WEBP")])
def test_screenshot_image_formats_can_be_imported(tmp_path,extension,format):
    source=tmp_path/f"screenshot.{extension}"
    with Image.new("RGB",(120,80),"white") as image:
        image.save(source,format=format)
    with open_document(source) as pages:
        page=next(pages)
        assert (page.width,page.height,page.page_number)==(120,80,1)
        assert page.native_spans==[]


def test_transparent_png_uses_same_white_pixels_for_recognition_and_page_hash(tmp_path):
    from biometry_parser.page_selection import _pixels
    source=tmp_path/"screenshot.png"
    with Image.new("RGBA",(100,120),(0,0,0,0)) as image:
        image.putpixel((20,20),(0,0,0,255))
        image.save(source)
        original_hash=_pixels(image)
    with open_document(source) as pages:
        page=next(pages)
        with Image.open(page.image_path) as rendered:
            assert rendered.getpixel((0,0))==(255,255,255)
            assert rendered.getpixel((20,20))==(0,0,0)
            assert _pixels(rendered)==original_hash


def test_pixel_and_page_limits(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (100, 100)).save(path)
    with open_document(path, max_page_pixels=99) as pages:
        with pytest.raises(ValueError, match="max_page_pixels"):
            next(pages)
    first, second = Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))
    multi = tmp_path / "multi.tiff"
    first.save(multi, save_all=True, append_images=[second])
    with pytest.raises(ValueError, match="max_pages"):
        with open_document(multi, max_pages=1):
            pass


@pytest.mark.parametrize("rotation,expected", [
    (0, (.1, .6, .3, .9)),
    (90, (.1, .1, .4, .3)),
    (180, (.7, .1, .9, .4)),
    (270, (.6, .7, .9, .9)),
])
def test_rotation_maps_original_pdf_coordinates(rotation, expected):
    box = _display_box((20, 10, 40, 40), (10, 0, 110, 100), rotation)
    assert (box.x0, box.y0, box.x1, box.y1) == pytest.approx(expected)


def test_nonempty_text_is_not_enough_for_native_route():
    p = PageInput(page_number=1, image_path="unused", width=100, height=100,
                  native_spans=[Span(text="This is a scanned hospital page with a text watermark only.", bbox=Box(x0=0,y0=0,x1=1,y1=.1))])
    assert not native_text_usable(p)


def test_selected_frame_preserves_original_number_and_skips_other_render(tmp_path):
    path=tmp_path/"selection.tiff"
    Image.new("RGB",(100,120),"white").save(path,save_all=True,append_images=[Image.new("RGB",(80,140),"black")])
    with open_document(path,page_numbers=[2]) as iterator:
        pages=list(iterator)
        assert len(pages)==1
        assert (pages[0].page_number,pages[0].width,pages[0].height)==(2,80,140)
        assert not (Path(pages[0].image_path).parent/"page-0001.png").exists()


@pytest.mark.parametrize("selection",[[],[0],[-1],[True],[1,1],[3]])
def test_invalid_selected_pages_are_rejected(tmp_path,selection):
    path=tmp_path/"one.png"
    Image.new("RGB",(80,100),"white").save(path)
    with pytest.raises(ValueError):
        with open_document(path,page_numbers=selection) as iterator:
            list(iterator)




