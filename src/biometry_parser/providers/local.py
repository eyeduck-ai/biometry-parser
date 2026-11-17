"""Optional local OCR adapters. Dependencies and weights load only on selection."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

from ..model_store import ModelStoreError, file_sha256, resolve_model
from ..models import Box, PageInput, ProviderOutput, Span
from . import ProviderError


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _box(points, page: PageInput) -> Box:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    clamp = lambda v: max(0.0, min(1.0, v))
    return Box(x0=clamp(min(xs)/page.width), y0=clamp(min(ys)/page.height), x1=clamp(max(xs)/page.width), y1=clamp(max(ys)/page.height))


def _model_paths(config: dict) -> dict:
    options = config.get("options", {})
    if options.get("det_model_path") or options.get("rec_model_path"):
        result = {}
        hashes = []
        for role in ("det", "rec"):
            path = Path(options.get(f"{role}_model_path", ""))
            if not path.is_file():
                raise ProviderError("model_unavailable", "Both explicit local detection and recognition models are required.")
            actual = file_sha256(path)
            expected = options.get(f"{role}_sha256")
            if expected and expected != actual:
                raise ProviderError("model_hash_mismatch", "An explicit local model failed its SHA-256 check.")
            result[role] = str(path.resolve())
            hashes.append(actual)
        result["hash"] = hashlib.sha256("".join(hashes).encode()).hexdigest()
        result["version"] = options.get("ocr_version", "PP-OCRv6")
        result["size"] = options.get("model_type", "small")
        result["recognition_language"] = options.get("recognition_language", "ch")
        return result
    try:
        return resolve_model(config.get("model") or "ppocr-v6-small", options.get("cache_dir"))
    except ModelStoreError as exc:
        raise ProviderError(exc.code, str(exc)) from None


class NativePDFProvider:
    def __init__(self, config: dict):
        self.config = config

    def describe(self):
        return dict(id="native_pdf", family="native_pdf", kind="spans", remote=False, version=_version("pypdfium2"), model_id="pdfium-text-layer")

    def recognize(self, page: PageInput) -> ProviderOutput:
        return ProviderOutput(kind="spans", spans=page.native_spans, metadata=self.describe())


class RapidOCRProvider:
    def __init__(self, config: dict):
        self.config = config
        self.options = config.get("options", {})
        self.rec_batch_num = self.options.get("rec_batch_num", 1)
        if type(self.rec_batch_num) is not int or not 1 <= self.rec_batch_num <= 32:
            raise ProviderError("invalid_configuration", "rec_batch_num must be an integer between 1 and 32.")
        self.paths = _model_paths(config)
        try:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import EngineType, LangRec, ModelType, OCRVersion
        except ImportError:
            raise ProviderError("dependency_unavailable", "Install the ocr extra in the selected worker environment.") from None
        version = OCRVersion(self.paths["version"])
        size = ModelType(self.paths["size"])
        self.recognition_language = self.paths.get("recognition_language", "ch")
        try:
            recognition_language = LangRec("ch" if self.recognition_language == "multilingual" else self.recognition_language)
        except ValueError:
            raise ProviderError("invalid_configuration", "Unsupported RapidOCR recognition_language.") from None
        params = {
            "Global.use_cls": False,
            "Global.text_score": float(self.options.get("text_score", 0.0)),
            "Global.max_side_len": int(self.options.get("max_side_len", 3508)),
            "Global.log_level": "error",
            "Det.model_path": self.paths["det"],
            "Rec.model_path": self.paths["rec"],
            "Rec.rec_batch_num": self.rec_batch_num,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Det.ocr_version": version,
            "Rec.ocr_version": version,
            "Det.model_type": size,
            "Rec.model_type": size,
            "Rec.lang_type": recognition_language,
            "Det.limit_side_len": int(self.options.get("det_limit_side_len", 1600)),
            "EngineConfig.onnxruntime.intra_op_num_threads": int(self.options.get("threads", 2)),
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.use_dml": False,
            "EngineConfig.onnxruntime.use_cann": False,
            "EngineConfig.onnxruntime.use_coreml": False,
        }
        # Both files are already validated; classifier and implicit model selection are disabled.
        self.engine = RapidOCR(params=params)

    def describe(self):
        return dict(id="rapidocr", family="paddle_ppocr", kind="spans", remote=False, version=_version("rapidocr"), model_id=self.config.get("model") or "ppocr-v6-small", model_hash=self.paths["hash"], recognition_language=self.recognition_language, execution_provider="CPUExecutionProvider", rec_batch_num=self.rec_batch_num, enable_cpu_mem_arena=False)

    def recognize(self, page: PageInput) -> ProviderOutput:
        result = self.engine(page.image_path, use_cls=False)
        spans = []
        if result.txts is not None and result.boxes is not None:
            scores = result.scores if result.scores is not None else [None] * len(result.txts)
            for points, text, confidence in zip(result.boxes, result.txts, scores):
                if text.strip():
                    spans.append(Span(text=text, bbox=_box(points, page), confidence=None if confidence is None else float(confidence), confidence_scale="0..1 (uncalibrated)"))
        return ProviderOutput(kind="spans", spans=spans, metadata=self.describe())
