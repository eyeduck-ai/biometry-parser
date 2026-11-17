"""Minimal provider contract, not an additional OCR engine.

It only returns any native spans already supplied by the caller. It belongs to
native_pdf's evidence family, so it cannot count as independent confirmation.
Install your real adapter in an importable application module before registration.
"""
from biometry_parser import PageInput, ProviderOutput


class NativeSpanExample:
    def __init__(self, config: dict):
        self.config = config

    def describe(self) -> dict:
        return {"id": "native_span_example", "family": "native_pdf",
                "kind": "spans", "remote": False}

    def recognize(self, page: PageInput) -> ProviderOutput:
        return ProviderOutput(kind="spans", spans=page.native_spans,
                              metadata={"model_id": "native-span-example"})


def create_provider(config: dict) -> NativeSpanExample:
    return NativeSpanExample(config)
