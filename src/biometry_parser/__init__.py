"""Reusable ophthalmic parsing, without loading OCR runtimes or making requests."""
from .client import BiometryParser, CloudRecognitionDisabled, extract
from .config import ExtractionConfig, Profile, Workflow, load_config
from .models import (
    Box, DataStatus, ExtractionResult, FieldClaim, FieldResult, IOLCalculation,
    IOLConstantClaim, IOLRow, IOLRowClaim, IOLTableClaim, Issue, Observation,
    PageInput, ParsedPage, ProviderOutput, RunRecord, Span,
)

__version__ = "0.1.0"


def register_provider(name, factory, *, replace=False):
    from .providers import register_provider as register
    return register(name, factory, replace=replace)


def list_providers():
    from .providers import list_providers as inventory
    return inventory()


def register_report_template(template, *, replace=False):
    from .report_templates import register_report_template as register
    return register(template, replace=replace)


def list_report_templates():
    from .report_templates import list_report_templates as inventory
    return inventory()


def plan_pages(source, **options):
    from .page_selection import plan_pages as plan
    return plan(source, **options)


__all__ = [
    "BiometryParser", "CloudRecognitionDisabled", "extract", "load_config",
    "ExtractionConfig", "Profile", "Workflow", "ExtractionResult", "FieldResult",
    "DataStatus", "ProviderOutput", "PageInput", "ParsedPage", "Span", "Box",
    "FieldClaim", "Observation", "RunRecord", "Issue", "IOLCalculation", "IOLRow",
    "IOLTableClaim", "IOLRowClaim", "IOLConstantClaim", "plan_pages",
    "register_provider", "list_providers", "register_report_template", "list_report_templates",
]
