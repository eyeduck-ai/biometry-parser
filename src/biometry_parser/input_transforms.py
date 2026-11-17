"""Load explicitly reviewed, integrity-checked page copies for remote studies."""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

from .model_store import file_sha256
from .providers import ProviderError


def reviewed_page(page, source_sha256: str, manifest_path: str):
    try:
        manifest_path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text('utf-8'))
        cases = [c for c in manifest['cases'] if c['source_sha256'] == source_sha256]
        if len(cases) != 1:
            raise ValueError()
        case = cases[0]
        count=case.get('source_page_count')
        if type(count) is not int or count < 1:
            raise ValueError()
        if any(type(p.get('page_number')) is not int or not 1 <= p['page_number'] <= count for p in case['pages']):
            raise ValueError()
        entries = [p for p in case['pages'] if p['page_number'] == page.page_number]
        if len(entries) != 1:
            raise ValueError()
        entry = entries[0]
        if not isinstance(case['excluded_fields'],list) or any(f not in {'patient_id','gender','measurement_date'} for f in case['excluded_fields']):
            raise ValueError()
        if entry.get('reviewed') is not True:
            raise ValueError()
        image_path = (manifest_path.parent / entry['image_path']).resolve(strict=True)
        if not image_path.is_relative_to(manifest_path.parent):
            raise ValueError()
        if file_sha256(image_path) != entry['image_sha256']:
            raise ValueError()
        with Image.open(image_path) as image:
            if image.format != 'PNG' or getattr(image,'n_frames',1) != 1 or set(image.info)-{'dpi'}:
                raise ValueError()
            if image.size != (page.width,page.height) or image.size != (entry['width'],entry['height']):
                raise ValueError()
            image.verify()
        metadata = {'kind':manifest.get('kind','reviewed_deidentification'),
            'source_sha256':source_sha256,'page_number':page.page_number,
            'image_sha256':entry['image_sha256'],'masks':entry.get('masks',[]),
            'excluded_fields':case['excluded_fields'],'reviewed':True,
            'synthetic_header':bool(entry.get('synthetic_header') or entry.get('header_added')),
            'header_added':entry.get('header_added'),'manifest_sha256':file_sha256(manifest_path)}
        return page.model_copy(update={'image_path':str(image_path),'native_spans':[]}), metadata
    except (OSError,ValueError,KeyError,StopIteration,TypeError):
        raise ProviderError('input_transform_invalid','Reviewed remote input copy is missing or failed integrity validation.') from None
