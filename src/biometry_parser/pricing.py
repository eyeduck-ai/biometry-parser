"""Dated list-price references. Estimates are never provider billing records."""
from datetime import datetime, timezone
from importlib.resources import files
import json
import math


def price_for(adapter: str, model: str, *, as_of: str | None = None) -> dict | None:
    catalog=json.loads(files('biometry_parser').joinpath('schemas/api-pricing.json').read_text('utf-8'))
    if adapter == 'paddleocr_api':
        return dict(catalog['paddleocr_api'])
    if adapter != 'gemini' or model not in catalog['gemini']:
        return None
    price=dict(catalog['gemini'][model])
    date=as_of or datetime.now(timezone.utc).date().isoformat()
    for future in price.pop('future_prices',[]):
        if date >= future['effective_from']:
            price.update(input_per_million_tokens=future['text_image_video_input'],
                         output_per_million_tokens=future['output_including_thinking'],
                         effective_from=future['effective_from'],valid_through=None)
    price['actual_charge_usd']=None
    return price


def estimate_reference_cost(pricing: dict | None, usage: dict | None):
    if not pricing or not isinstance(usage,dict):
        return None
    incoming=usage.get('promptTokenCount',usage.get('input_tokens',usage.get('prompt_tokens')))
    outgoing=usage.get('candidatesTokenCount',usage.get('output_tokens',usage.get('completion_tokens')))
    if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in (incoming,outgoing)):
        return None
    if 'candidatesTokenCount' in usage:
        thoughts=usage.get('thoughtsTokenCount',0)
        if not isinstance(thoughts,int) or isinstance(thoughts,bool) or thoughts < 0:
            return None
        if isinstance(outgoing,int):outgoing+=thoughts
    rates=[pricing.get('input_per_million_tokens'),pricing.get('output_per_million_tokens')]
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<0 for v in rates):
        return None
    return (incoming*rates[0]+outgoing*rates[1])/1_000_000
