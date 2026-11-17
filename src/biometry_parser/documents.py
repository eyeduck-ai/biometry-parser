"""Bounded, lazy PDF/image ingestion with original-page coordinates.

Rendered files belong to the context and are removed when it exits. Adapters
must finish using the files before leaving :func:`open_document`.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from PIL import Image, ImageOps

from .models import Box, PageInput, Span


def display_rgb(image: Image.Image) -> Image.Image:
    """Normalize the displayed image; transparent report pixels use white paper.

    Discarding alpha with convert('RGB') can turn a transparent screenshot's
    background black. The preview, page hash and recognizer must see the same
    composited pixels. The caller owns the returned image.
    """
    if "A" not in image.getbands() and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    try:
        background = Image.new("RGB", rgba.size, "white")
        with rgba.getchannel("A") as alpha:
            background.paste(rgba, mask=alpha)
        return background
    finally:
        rgba.close()


def _display_box(rect, crop, rotation: int) -> Box:
    left, bottom, right, top = crop
    width, height = right - left, top - bottom
    x0, y0, x1, y1 = rect
    points = [(x0 - left, top - y0), (x1 - left, top - y1)]
    if rotation == 90:
        points = [(height - y, x) for x, y in points]
        width, height = height, width
    elif rotation == 180:
        points = [(width - x, height - y) for x, y in points]
    elif rotation == 270:
        points = [(y, width - x) for x, y in points]
        width, height = height, width
    xs, ys = zip(*points)
    clip = lambda n: max(0.0, min(1.0, n))
    return Box(x0=clip(min(xs) / width), y0=clip(min(ys) / height),
               x1=clip(max(xs) / width), y1=clip(max(ys) / height))


def _native_spans(page) -> list[Span]:
    """Retain words (rather than reading order supplied by a PDF producer)."""
    text = page.get_textpage()
    crop = page.get_bbox()
    rotation = page.get_rotation()
    spans: list[Span] = []
    chars: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []

    def flush():
        if chars and boxes:
            rect = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes))
            spans.append(Span(text="".join(chars), bbox=_display_box(rect, crop, rotation)))
        chars.clear()
        boxes.clear()

    try:
        for i in range(text.count_chars()):
            char = text.get_text_range(i, 1)
            if not char or char.isspace() or char == "\x00":
                flush()
                continue
            rect = text.get_charbox(i)
            if rect[2] <= rect[0] or rect[3] <= rect[1]:
                flush()
                continue
            if boxes:
                last = boxes[-1]
                char_height = max(last[3] - last[1], rect[3] - rect[1])
                # Missing literal spaces are common in PDF text objects.
                if abs((rect[1] + rect[3]) - (last[1] + last[3])) > char_height or rect[0] - last[2] > char_height * .65:
                    flush()
            chars.append(char)
            boxes.append(rect)
        flush()
    finally:
        text.close()
    return spans


def native_text_usable(page: PageInput) -> bool:
    """Require a registered device, identity and measurement/table anchors."""
    from .report_templates import inspect_report
    return inspect_report(page.native_spans)["native_quality"]["usable"]


def ensure_page_image(page: PageInput, *, dpi: int = 300,
                      max_page_pixels: int = 20_000_000) -> PageInput:
    """Materialize one deferred original PDF raster only when it is required.

    The destination belongs to ``open_document`` and must be used within that
    context. Native parsing can use page metadata without creating this PNG.
    """
    destination = Path(page.image_path)
    if destination.is_file():
        return page
    source = Path(page.source_path)
    if source.suffix.lower() != ".pdf" or not source.is_file():
        raise FileNotFoundError("a deferred page needs its original PDF")
    if not 72 <= dpi <= 600:
        raise ValueError("dpi must be between 72 and 600")
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(source) as document:
        if not 1 <= page.page_number <= len(document):
            raise ValueError("page number is outside the original PDF")
        original = document[page.page_number - 1]
        try:
            width, height = original.get_size()
            scale = dpi / 72
            if width * height * scale * scale > max_page_pixels:
                raise ValueError("page exceeds max_page_pixels")
            bitmap = original.render(scale=scale)
            try:
                image = bitmap.to_pil()
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image.save(destination, format="PNG")
                    return page.model_copy(update={"width": image.width, "height": image.height})
                finally:
                    image.close()
            finally:
                bitmap.close()
        finally:
            original.close()


@contextmanager
def open_document(source: str | Path, *, dpi: int = 300,
                  cache_dir: Path | None = None, max_page_pixels: int = 20_000_000,
                  max_pages: int = 50, page_numbers: list[int] | None = None,
                  render_pages: bool = True) -> Iterator[Iterator[PageInput]]:
    """Yield pages lazily; keep only one decoded image in memory at a time.

    ``max_page_pixels`` is a per-page safety limit, not an instruction to silently
    downsample an unreadable input. PDF render DPI is bounded by the same limit.
    EXIF orientation is applied to image inputs; evidence coordinates refer to
    that displayed orientation, as they do to PDF page rotation.
    With ``render_pages=False``, PDF PNGs are deferred until
    :func:`ensure_page_image` is called; native spans remain available immediately.
    """
    path = Path(source).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not 72 <= dpi <= 600:
        raise ValueError("dpi must be between 72 and 600")
    if max_page_pixels < 1 or max_pages < 1:
        raise ValueError("max_page_pixels and max_pages must be positive")
    if page_numbers is not None and (not isinstance(page_numbers, list) or not page_numbers
            or any(type(n) is not int or n < 1 for n in page_numbers)
            or len(set(page_numbers)) != len(page_numbers)):
        raise ValueError("page_numbers must contain distinct positive page numbers")
    def selected_indices(count):
        if page_numbers is None:
            return range(count)
        if max(page_numbers) > count:
            raise ValueError("requested page is outside the document")
        return [n-1 for n in sorted(page_numbers)]
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="biometry-", dir=cache_dir) as temp:
        directory = Path(temp)
        if path.suffix.lower() == ".pdf":
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(path)
            if len(doc) > max_pages:
                doc.close()
                raise ValueError("document exceeds max_pages")
            try:
                indices = selected_indices(len(doc))
            except Exception:
                doc.close()
                raise

            def pdf_pages():
                for i in indices:
                    page = doc[i]
                    try:
                        width, height = page.get_size()
                        scale = dpi / 72
                        if render_pages and width * height * scale * scale > max_page_pixels:
                            raise ValueError(f"page {i + 1} exceeds max_page_pixels at {dpi} DPI")
                        spans = _native_spans(page)
                        dest = directory / f"page-{i + 1:04}.png"
                        if not render_pages:
                            result = PageInput(page_number=i + 1, image_path=str(dest),
                                               width=math.ceil(width * scale), height=math.ceil(height * scale),
                                               native_spans=spans, source_path=str(path))
                        else:
                            bitmap = page.render(scale=scale)
                            try:
                                image = bitmap.to_pil()
                                try:
                                    image.save(dest, format="PNG")
                                    result = PageInput(page_number=i + 1, image_path=str(dest),
                                                       width=image.width, height=image.height,
                                                       native_spans=spans, source_path=str(path))
                                finally:
                                    image.close()
                            finally:
                                bitmap.close()
                    finally:
                        page.close()
                    yield result

            pages = pdf_pages()
            try:
                yield pages
            finally:
                pages.close()
                doc.close()
        else:
            image = Image.open(path)
            if getattr(image, "n_frames", 1) > max_pages:
                image.close()
                raise ValueError("document exceeds max_pages")
            try:
                indices = selected_indices(getattr(image, "n_frames", 1))
            except Exception:
                image.close()
                raise

            def image_pages():
                for i in indices:
                    image.seek(i)
                    if image.width * image.height > max_page_pixels:
                        raise ValueError(f"page {i + 1} exceeds max_page_pixels")
                    displayed = ImageOps.exif_transpose(image)
                    rgb = display_rgb(displayed)
                    try:
                        dest = directory / f"page-{i + 1:04}.png"
                        rgb.save(dest, format="PNG")
                        result = PageInput(page_number=i + 1, image_path=str(dest),
                                           width=rgb.width, height=rgb.height, source_path=str(path))
                    finally:
                        rgb.close()
                        if displayed is not image:
                            displayed.close()
                    yield result

            pages = image_pages()
            try:
                yield pages
            finally:
                pages.close()
                image.close()
