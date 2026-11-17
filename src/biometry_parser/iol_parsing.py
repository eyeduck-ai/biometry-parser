"""IOL calculation source tables, kept separate from measured biometry.

No recommended/selected implant is inferred from closeness to zero, row order,
bold text or an Emmetropia row. Printed highlighting is a separate source fact.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import re
from statistics import median

from .models import (Box, IOLCalculation, IOLConstantClaim, IOLRow,
                     IOLRowClaim, IOLTableClaim, Issue, PageInput,
                     ProviderOutput, Span)

_NUM = re.compile(r"(?<![A-Za-z.])(?:[+\-−]\d+(?:\.\d+)?|(?<![0-9])\d+(?:\.\d+)?)(?![A-Za-z0-9.])")
_FORMULA = re.compile(r"Barrett|Haigis|Hoffer|Holladay|SRK\s*/?\s*T|Kane|Hill.?RBF|EVO|Olsen|PEARL|Castrop", re.I)
_COLUMN = re.compile(
    r"\b(?P<iol_axis>IOL\s*Axis)\b|\b(?P<toric>Toric\s*Power|RL\s*Cyl|IOL\s*Cy[lI])\b|"
    r"(?P<ignored>Ref\s*Sph|Ref\s*Cyl|Ref\s*Axis|Res\.?\s*Ast\.?)|"
    r"(?P<refraction>Res\.?\s*SE|Ref\.?\s*SE|Ref\s*\(D\)|\bRef\b)|"
    r"\b(?P<iol>IOL(?:\s*\(D\)|\s+SE)?)\b", re.I)
_CONSTANT = re.compile(r"(?<![A-Za-z])(?P<name>LF|DF|A[- ]?const(?:ant)?|pACD|SF)\s*:?\s*(?P<value>Default|[+\-−]?\d+(?:\.\d+)?)", re.I)


def _hull(spans: list[Span]) -> Box | None:
    if not spans:
        return None
    return Box(x0=min(s.bbox.x0 for s in spans), y0=min(s.bbox.y0 for s in spans),
               x1=max(s.bbox.x1 for s in spans), y1=max(s.bbox.y1 for s in spans))


def _words(spans: list[Span]) -> list[Span]:
    """Approximate token positions inside OCR line boxes; retain raw line proof."""
    result = []
    for span in spans:
        for match in re.finditer(r"\S+", span.text):
            if len(match.group()) == len(span.text):
                result.append(span)
                continue
            width = span.bbox.x1-span.bbox.x0
            result.append(Span(text=match.group(), bbox=Box(
                x0=span.bbox.x0+width*match.start()/len(span.text),
                x1=span.bbox.x0+width*match.end()/len(span.text),
                y0=span.bbox.y0, y1=span.bbox.y1),
                confidence=span.confidence, confidence_scale=span.confidence_scale))
    return result


def _at(text: str, intervals, start: int, end: int) -> float:
    matching = [(a,b,s) for a,b,s in intervals if b > start and a < end]
    if not matching:
        return 0
    xs = []
    for a,b,s in matching:
        width = s.bbox.x1-s.bbox.x0
        xs.extend([s.bbox.x0+width*max(0,start-a)/max(1,b-a),
                   s.bbox.x0+width*min(b-a,end-a)/max(1,b-a)])
    return (min(xs)+max(xs))/2


def _eye_markers(rows, before: float):
    found = []
    for row in rows:
        if row.y >= before:
            continue
        for span in row.spans:
            match = re.fullmatch(r"(?:(?:right|left)\s+)?(OD|OS)(?:\s+(?:right|left))?", span.text.strip(), re.I)
            if match:
                found.append((row.y, (span.bbox.x0+span.bbox.x1)/2, match.group(1).upper()))
    if not found:
        return {}
    y = max(item[0] for item in found)
    return {eye:x for row_y,x,eye in found if abs(row_y-y)<.025}


def _eye_at(x: float, centers):
    if len(centers)==1:
        return next(iter(centers))
    if len(centers)==2:
        left,right=sorted(centers,key=centers.get)
        return left if x < (centers[left]+centers[right])/2 else right
    return None


def _columns(row):
    text, intervals = row.string()
    cols = []
    for match in _COLUMN.finditer(text):
        kind = match.lastgroup
        # RefSph/RefCyl/RefAxis without a space must be consumed as other
        # columns, rather than letting a generic Ref match steal the cell.
        cols.append((_at(text,intervals,match.start(),match.end()),kind,match.group()))
    return cols


def _segment(row, left: float, right: float):
    return [s for s in row.spans if left <= (s.bbox.x0+s.bbox.x1)/2 < right]


def _text(spans):
    return " ".join(s.text for s in sorted(spans,key=lambda s:s.bbox.x0))


def _target_refractions(rows, centers):
    values = {}
    for row in rows:
        text, intervals = row.string()
        for match in re.finditer(r"Target\s+ref\.?\s*:?\s*(plano|[+\-−]?\d+(?:\.\d+)?)",text,re.I):
            eye=_eye_at(_at(text,intervals,match.start(),match.end()),centers)
            raw="0" if match.group(1).lower()=="plano" else match.group(1)
            values.setdefault(eye,set()).add(raw)
    return {eye:next(iter(items)) for eye,items in values.items() if len(items)==1}


def _header_context(rows, header_index: int, left: float, right: float, previous_header_y: float):
    nearby = [(i,_segment(row,left,right)) for i,row in enumerate(rows[:header_index])
              if previous_header_y < row.y and rows[header_index].y-row.y < .16]
    formula_index = next((i for i,spans in reversed(nearby) if _FORMULA.search(_text(spans))),None)
    formula = None
    lens_parts = []
    source = "unknown"
    used=[]
    if formula_index is not None:
        formula_spans=_segment(rows[formula_index],left,right)
        formula=_text(formula_spans).strip(" -–—") or None
        # A trailing K/TK on compact printouts is the measurement source.
        if formula:
            marker=re.search(r"\s+(TK|K)\s*$",formula)
            if marker:
                source=marker.group(1)
                formula=formula[:marker.start()].strip()
        used.extend(formula_spans)
        for i,spans in reversed(nearby):
            if i>=formula_index:
                continue
            content=_text(spans).strip()
            if not content:
                continue
            if rows[formula_index].y-rows[i].y > .055:
                break
            if re.search(r"(?:\b(?:AL|ACD|LT|CCT|WTW|K1|K2|TK1|TK2|SE|TSE)\s*:|[Δ∆]\s*TK\s*:|Biometric|Eye status|Target ref|Emmetropia|[+\-]\d+\.\d+)",content,re.I):
                break
            if re.search(r"\bCALCULATION\b|flipped\s+axis",content,re.I) or content.upper()=="IOL":
                break
            if re.search(r"\b(?:OD|OS)\b",content):
                break
            if _CONSTANT.search(content) or _FORMULA.search(content):
                break
            marker=re.match(r"^(TK|K)\b\s*",content)
            if marker:
                source=marker.group(1)
                content=content[marker.end():]
            content=re.sub(r"\s+(?:TK|K)$","",content).strip()
            if content:
                lens_parts.insert(0,content)
                used.extend(spans)
    constants=[]
    for i,spans in nearby:
        if formula_index is not None and i<formula_index:
            continue
        content=_text(spans)
        for match in _CONSTANT.finditer(content):
            constants.append(IOLConstantClaim(name=match.group("name"),value=match.group("value"),raw_text=match.group(),bbox=_hull(spans)))
            used.extend(spans)
    lens=" ".join(lens_parts).strip() or None
    return lens,formula,source,constants,used


def _raw_row(row, columns, left, right, *, primary_only=False):
    spans=_segment(row,left,right)
    text=_text(spans)
    if not spans or not columns:
        return None
    # A new lens/formula/header is not a numerical calculation row.
    if _FORMULA.search(text) or _CONSTANT.search(text) or re.search(r"IOL|Ref|Comment|Borderline",text,re.I):
        return None
    allowed=re.sub(r"Emmetropia|selected|\b[DTNK]\b","",text,flags=re.I)
    if re.search(r"[A-Za-z]",allowed):
        return None
    values={}
    ordered_numbers=[]
    columns=sorted(columns)
    for span in spans:
        for match in _NUM.finditer(span.text):
            # Diagram angular ticks and alignment labels are not lens powers.
            is_angle="°" in span.text[match.end():match.end()+2]
            x=span.bbox.x0+(span.bbox.x1-span.bbox.x0)*(match.start()+match.end())/(2*max(1,len(span.text)))
            closest=min(columns,key=lambda c:abs(c[0]-x))
            kind=closest[1]
            if is_angle and not any(col[1]=="iol_axis" for col in columns):
                continue
            # ZEISS calculation diopters are printed with a sign or decimal.
            # Bare integer labels alongside toric diagrams (e.g. model/cylinder
            # labels with a dropped dot) cannot establish a dioptric cell.
            if not is_angle and kind!="iol_axis" and "." not in match.group() and not match.group().startswith(("+","-","−")):
                continue
            ordered_numbers.append((x,match.group()))
            values.setdefault(kind,[]).append(match.group())
    # Some printouts right-align numeric cells far from centered headings
    # (especially Toric Power). When every declared column has exactly one
    # value, left-to-right order is stronger evidence than heading distance.
    if len(ordered_numbers)==len(columns):
        values={}
        for (_,raw),(_,kind,_) in zip(sorted(ordered_numbers),columns):
            values.setdefault(kind,[]).append(raw)
    power=values.get("iol",[])
    if not power:
        return None
    claim=IOLRowClaim(raw_text=text,bbox=_hull(spans))
    for kind,field in [("iol","iol_power"),("refraction","predicted_refraction"),("toric","toric_power"),("iol_axis","axis")]:
        if primary_only and kind not in {"iol","refraction"}:
            continue
        found=values.get(kind,[])
        if len(found)==1:
            setattr(claim,field,found[0])
    if "emmetropia" in text.lower():
        claim.row_kind="emmetropia"
    if re.search(r"☑|✓|✔|\[x\]|\bselected\b",text,re.I):
        claim.selection="marked"
        claim.selection_evidence=text
    elif "☐" in text:
        claim.selection="not_marked"
        claim.selection_evidence="printed empty checkbox"
    return claim


def _split_power_rows(row, columns, left, right):
    """Split globally merged OCR rows using actual power-column baselines.

    Skewed diagram text can bridge two adjacent rows during page-wide line
    grouping. The power/refraction columns retain separate baselines. Other
    columns are left unresolved in this case rather than borrowed from a
    neighboring row or a diagram.
    """
    from .parsing import _Row
    spans=_segment(row,left,right)
    anchors=[]
    for span in spans:
        for match in _NUM.finditer(span.text):
            if "." not in match.group() and not match.group().startswith(("+","-","−")):
                continue
            x=span.bbox.x0+(span.bbox.x1-span.bbox.x0)*(match.start()+match.end())/(2*max(1,len(span.text)))
            if columns and min(columns,key=lambda c:abs(c[0]-x))[1]=="iol":
                anchors.append((span.bbox.y0+span.bbox.y1)/2)
    height=median(s.bbox.y1-s.bbox.y0 for s in spans) if spans else 0
    if len(anchors)<2 or max(anchors)-min(anchors)<max(.002,height*.45):
        return [(row,False)]
    centers=[]
    for y in sorted(anchors):
        if not centers or y-centers[-1]>max(.001,height*.3):
            centers.append(y)
    groups=[[] for _ in centers]
    for span in spans:
        x=(span.bbox.x0+span.bbox.x1)/2
        if min(columns,key=lambda c:abs(c[0]-x))[1] not in {"iol","refraction"}:
            continue
        y=(span.bbox.y0+span.bbox.y1)/2
        closest=min(range(len(centers)),key=lambda i:abs(centers[i]-y))
        groups[closest].append(span)
    return [(_Row(group),True) for group in groups if group]


def _align_primary_refractions(claims, rows, columns, left, right):
    """Resolve primary cells from their own baselines, independent of diagrams.

    Page-wide row grouping can put one refraction into the preceding row when
    a tilted toric diagram bridges their text boxes. Only a unique power cell
    and a uniquely nearest, vertically overlapping refraction cell establish
    the pairing. Secondary diagram columns never participate in this repair.
    Ambiguous, repeated or vertically merged cells remain unresolved.
    """
    if sum(kind == "iol" for _, kind, _ in columns) != 1 or sum(kind == "refraction" for _, kind, _ in columns) != 1:
        return claims
    cells = {"iol": [], "refraction": []}
    seen = set()
    for row in rows:
        for span in _segment(row, left, right):
            matches = list(_NUM.finditer(span.text))
            # A single source box containing multiple numbers does not retain
            # enough vertical evidence to align its individual values safely.
            if len(matches) != 1:
                continue
            match = matches[0]
            raw = match.group()
            if "°" in span.text[match.end():match.end()+2] or ("." not in raw and not raw.startswith(("+", "-", "−"))):
                continue
            x = span.bbox.x0 + (span.bbox.x1-span.bbox.x0)*(match.start()+match.end())/(2*max(1,len(span.text)))
            kind = min(columns, key=lambda c: abs(c[0]-x))[1]
            key = (kind, raw, span.bbox.model_dump_json())
            if kind in cells and key not in seen:
                cells[kind].append((Decimal(raw.replace("−", "-")), span))
                seen.add(key)
    if not cells["iol"] or not cells["refraction"]:
        return claims
    assigned = {}
    for value, ref in cells["refraction"]:
        y = (ref.bbox.y0+ref.bbox.y1)/2
        nearest = sorted(enumerate(cells["iol"]), key=lambda item: abs((item[1][1].bbox.y0+item[1][1].bbox.y1)/2-y))
        index, (_, power) = nearest[0]
        delta = abs((power.bbox.y0+power.bbox.y1)/2-y)
        if len(nearest) > 1:
            other = nearest[1][1][1]
            if abs((other.bbox.y0+other.bbox.y1)/2-y)-delta <= .001:
                continue
        height = min(ref.bbox.y1-ref.bbox.y0, power.bbox.y1-power.bbox.y0)
        overlap = min(ref.bbox.y1,power.bbox.y1)-max(ref.bbox.y0,power.bbox.y0)
        if height <= 0 or overlap < height*.5 or delta > max(.001,height*.5):
            continue
        assigned.setdefault(index, []).append((value, ref))
    result = []
    for claim in claims:
        if claim.predicted_refraction is not None or claim.iol_power is None or claim.row_kind == "emmetropia":
            result.append(claim)
            continue
        power_matches = [(i, span) for i, (value, span) in enumerate(cells["iol"])
                         if value == Decimal(claim.iol_power.replace("−", "-"))]
        if len(power_matches) != 1:
            result.append(claim)
            continue
        index, power = power_matches[0]
        refs = assigned.get(index, [])
        if len(refs) != 1:
            result.append(claim)
            continue
        _, ref = refs[0]
        updated = claim.model_copy(deep=True)
        updated.predicted_refraction = _NUM.search(ref.text).group()
        updated.raw_text += " | row-aligned source refraction: " + ref.text
        if updated.toric_power is None and updated.axis is None:
            updated.bbox = _hull([power, ref])
        result.append(updated)
    return result


def _table_claims(page: PageInput, spans: list[Span]):
    from .parsing import _rows
    rows=_rows(_words(spans))
    headers=[]
    for i,row in enumerate(rows):
        cols=_columns(row)
        if any(kind=="iol" for _,kind,_ in cols) and any(kind=="refraction" for _,kind,_ in cols):
            headers.append((i,cols))
    tables=[]
    for header_number,(index,cols) in enumerate(headers):
        anchor_cols=[col for col in cols if col[1]=="iol"]
        anchors=[c[0] for c in anchor_cols]
        step=median(b-a for a,b in zip(anchors,anchors[1:])) if len(anchors)>1 else .4
        ref_gaps=[ref[0]-anchor for anchor in anchors for ref in cols
                  if ref[1]=="refraction" and ref[0]>anchor and not any(anchor<x<ref[0] for x in anchors)]
        padding=min(.08,step*.30,min(ref_gaps)*.70 if ref_gaps else .06)
        boundaries=[max(0,x-padding) for x in anchors]
        boundaries.append(min(1,max(anchors[-1]+step-padding,max(c[0] for c in cols)+padding)))
        centers=_eye_markers(rows,rows[index].y)
        if len(centers)==1 and len(anchors)>=4:
            centers={}
        if len(centers)==2:
            eye_boundary=sum(centers.values())/2
            for boundary_index in range(1,len(anchors)):
                if _eye_at(anchors[boundary_index-1],centers)!=_eye_at(anchors[boundary_index],centers):
                    # Keep the last OD axis cell out of the first OS power
                    # column on wide bilateral toric tables.
                    boundaries[boundary_index]=eye_boundary
        targets=_target_refractions(rows[:index],centers)
        stop=headers[header_number+1][0] if header_number+1<len(headers) else len(rows)
        previous_y=rows[headers[header_number-1][0]].y if header_number else 0
        for n,anchor in enumerate(anchors):
            left,right=boundaries[n:n+2]
            local_cols=[col for col in cols if left<=col[0]<right]
            lens,formula,source,constants,context=_header_context(rows,index,left,right,previous_y)
            claims=[]
            numeric_rows=[]
            all_spans=list(context)+_segment(rows[index],left,right)
            for row in rows[index+1:stop]:
                content=_text(_segment(row,left,right))
                # Constants on newer compact printouts are below the rows.
                matches=list(_CONSTANT.finditer(content))
                if matches:
                    for match in matches:
                        constants.append(IOLConstantClaim(name=match.group("name"),value=match.group("value"),raw_text=match.group(),bbox=_hull(_segment(row,left,right))))
                    if claims:
                        all_spans.extend(_segment(row,left,right))
                        break
                    continue
                if claims and (_FORMULA.search(content) or re.search(r"Comment|Borderline",content,re.I)):
                    break
                numeric_rows.append(row)
                done=False
                for data_row,primary_only in _split_power_rows(row,local_cols,left,right):
                    claim=_raw_row(data_row,local_cols,left,right,primary_only=primary_only)
                    if claim:
                        if primary_only:
                            claim.raw_text=content
                            claim.bbox=_hull(_segment(row,left,right))
                        claims.append(claim)
                        all_spans.extend(_segment(row,left,right))
                        if claim.row_kind=="emmetropia":
                            done=True
                            break
                if done:
                    break
            claims=_align_primary_refractions(claims,numeric_rows,local_cols,left,right)
            if not claims:
                continue
            eye=_eye_at(anchor,centers)
            table=IOLTableClaim(eye=eye,lens_model=lens,formula=formula,
                                lens_type="toric" if any(c[1]=="toric" for c in local_cols) or formula and re.search(r"\btoric\b",formula,re.I) else "unknown",
                                keratometry_source=source,target_refraction=targets.get(eye),
                                constants=constants,rows=claims,bbox=_hull(all_spans),
                                raw_text="\n".join(_text(_segment(row,left,right)) for row in rows
                                                   if all_spans and min(s.bbox.y0 for s in all_spans)<=row.y<=max(s.bbox.y1 for s in all_spans)))
            tables.append(table)
    _recover_highlights(page,tables)
    return tables


def _recover_highlights(page: PageInput, tables: list[IOLTableClaim]):
    """Recognize a uniform shaded stripe; never translate it to selection."""
    from pathlib import Path
    if not Path(page.image_path).is_file():
        return
    from PIL import Image, ImageStat
    try:
        with Image.open(page.image_path) as original:
            image=original.convert("L")
            try:
                for table in tables:
                    if not table.bbox:
                        continue
                    backgrounds=[]
                    for row in table.rows:
                        if not row.bbox or row.row_kind=="emmetropia":
                            backgrounds.append(None)
                            continue
                        # Leave the text interior in the crop: a uniform source
                        # stripe still has a stable median despite ink glyphs.
                        b=row.bbox
                        crop=image.crop((int(table.bbox.x0*image.width)+3,int(b.y0*image.height),
                                         int(table.bbox.x1*image.width)-3,max(int(b.y1*image.height),int(b.y0*image.height)+1)))
                        try:
                            backgrounds.append(ImageStat.Stat(crop).median[0])
                        finally:
                            crop.close()
                    known=[v for v in backgrounds if v is not None]
                    if len(known)<3 or max(known)<240:
                        continue
                    for row,background in zip(table.rows,backgrounds):
                        if background is not None and 160<=background<=235 and max(known)-background>=18:
                            row.highlight_status="highlighted"
                            row.highlight_evidence="source image contains a shaded row stripe; surgical selection unknown"
            finally:
                image.close()
    except (OSError,ValueError):
        return


def parse_iol_tables(page: PageInput, output: ProviderOutput, *, profile_id: str,
                     model_id: str="", family: str="") -> tuple[list[IOLCalculation],list[Issue]]:
    """Normalize source tables; never choose or merge a preferred implant."""
    if output.supported is False or output.kind=="claims" and output.supported is not True:
        return [],[]
    claims=output.iol_calculations if output.kind=="claims" else _table_claims(page,output.spans)
    result,issues=[],[]
    provider_id=str(output.metadata.get("provider_id",output.metadata.get("id",profile_id)))
    for ordinal,claim in enumerate(claims):
        flags=[]
        if claim.eye is None:
            flags.append("eye_unresolved")
            issues.append(Issue(code="iol_eye_unresolved",message="IOL table has no reliable eye heading.",page_number=page.page_number,profile_id=profile_id))
        def number(raw,field,row_flags):
            if raw is None or not raw.strip():
                return None
            text=raw.strip().replace("−","-")
            if field=="target_refraction" and text.lower()=="plano":
                return Decimal(0)
            suffix = r"(?:°|deg(?:ree)?s?)" if field=="axis" else r"D"
            literal=re.fullmatch(r"([+\-]?\d+(?:\.\d+)?)\s*"+suffix,text,re.I)
            if literal:
                text=literal.group(1)
                if "literal_unit_suffix_separated" not in row_flags:
                    row_flags.append("literal_unit_suffix_separated")
            if not re.fullmatch(r"[+\-]?\d+(?:\.\d+)?",text):
                row_flags.append(field+"_unreadable")
                issues.append(Issue(code="invalid_iol_cell",message="A calculation cell is not an unambiguous decimal.",field=field,eye=claim.eye,page_number=page.page_number,profile_id=profile_id))
                return None
            try:
                value=Decimal(text)
            except InvalidOperation:
                return None
            if field=="axis" and not 0<=value<=180:
                row_flags.append("axis_out_of_range")
                return None
            return value
        rows=[]
        for raw in claim.rows:
            row_flags=[]
            values={key:number(getattr(raw,key),key,row_flags) for key in ("iol_power","predicted_refraction","toric_power","axis")}
            if values["iol_power"] is None:
                row_flags.append("iol_power_unreadable")
            selection=raw.selection
            evidence=raw.selection_evidence.strip()
            positive_marker=bool(re.search(r"☑|✓|✔|\[x\]|\b(?:selected|chosen|checked|checkbox|tick(?:ed)?)\b|已選|選定|勾選",evidence,re.I))
            negative_marker=bool(re.search(r"☐|\b(?:unchecked|empty\s+checkbox|not\s+(?:selected|chosen|checked|marked))\b|未選|未勾",evidence,re.I))
            explicit_marker=positive_marker and not negative_marker if selection=="marked" else negative_marker
            if selection!="unknown" and (not evidence or not explicit_marker):
                selection="unknown"
                row_flags.append("selection_unsubstantiated")
            highlight=raw.highlight_status
            if highlight!="unknown" and not raw.highlight_evidence.strip():
                highlight="unknown"
                row_flags.append("highlight_unsubstantiated")
            raw_text=raw.raw_text
            if "literal_unit_suffix_separated" in row_flags:
                literal_cells=[f"{field}={getattr(raw,field)}" for field in values
                               if getattr(raw,field) and not re.fullmatch(r"[+\-−]?\d+(?:\.\d+)?",getattr(raw,field).strip())]
                # This is explicitly a provider transcription, not a fabricated
                # quote of unseen source text; retain all original unit glyphs.
                raw_text += (" | " if raw_text else "")+"provider cell transcription: "+"; ".join(literal_cells)
            rows.append(IOLRow(**values,row_kind=raw.row_kind,selection=selection,
                               selection_evidence=raw.selection_evidence,highlight_status=highlight,
                               highlight_evidence=raw.highlight_evidence,raw_text=raw_text,
                               bbox=raw.bbox,flags=row_flags))
        fingerprint=f"{profile_id}:{page.page_number}:{ordinal}:{claim.model_dump_json()}"
        target_refraction=number(claim.target_refraction,"target_refraction",flags)
        table_raw_text=claim.raw_text
        if "literal_unit_suffix_separated" in flags:
            table_raw_text += (" | " if table_raw_text else "")+"provider cell transcription: target_refraction="+str(claim.target_refraction)
        result.append(IOLCalculation(table_id=hashlib.sha256(fingerprint.encode()).hexdigest()[:24],
                      eye=claim.eye,lens_model=claim.lens_model,lens_type=claim.lens_type,
                      formula=claim.formula,keratometry_source=claim.keratometry_source,
                      target_refraction=target_refraction,
                      constants=claim.constants,rows=rows,page_number=page.page_number,
                      bbox=claim.bbox,raw_text=table_raw_text,provider_id=provider_id,
                      profile_id=profile_id,model_id=model_id,model_hash=output.metadata.get("model_hash"),
                      family=family,flags=flags))
    return result,issues


def without_iol_table_rows(spans: list[Span], tables: list[IOLCalculation]) -> list[Span]:
    """Keep column headers and numeric IOL rows out of measured SE/P fields."""
    boxes=[table.bbox for table in tables if table.bbox is not None]
    return [span for span in spans if not any(
        box.x0-.005 <= (span.bbox.x0+span.bbox.x1)/2 <= box.x1+.005 and
        box.y0-.003 <= (span.bbox.y0+span.bbox.y1)/2 <= box.y1+.003 for box in boxes)]
