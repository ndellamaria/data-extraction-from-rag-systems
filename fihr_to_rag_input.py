#!/usr/bin/env python3
"""
fhir_to_rag.py — Convert FHIR JSON to RAG/chunking-friendly JSONL.

Input:
  - Files or directories containing:
      * Single FHIR resource JSON
      * Bundle JSON (resourceType="Bundle")
      * NDJSON/JSONL with one FHIR resource per line

Output:
  - JSONL with fields:
      doc_id, chunk_id, resource_type, fhir_id, source_file, metadata, text

Usage:
  - Default (no args): process all JSON files under datasets/fhir and write
    datasets/fhir/rag_chunks.jsonl

  - Custom:
      python fhir_to_rag.py INPUT_PATH [INPUT_PATH ...] \
        --out out.jsonl \
        --max-chars 2000 \
        --overlap 200
"""

import argparse
import json
import os
import sys
import html
import re
from typing import Any, Dict, Iterable, List, Tuple, Union

# -------------- Utilities --------------

HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

def strip_html(s: str) -> str:
    # Unescape HTML entities and drop tags
    s = html.unescape(s)
    s = HTML_TAG_RE.sub(" ", s)
    return WS_RE.sub(" ", s).strip()

def is_primitive(val: Any) -> bool:
    return isinstance(val, (str, int, float, bool))

def join_nonempty(parts: Iterable[str], sep: str = " ") -> str:
    return sep.join(p for p in parts if p and isinstance(p, str))

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# -------------- FHIR-aware text extraction --------------

def extract_text_from_resource(res: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (text, metadata) for a FHIR resource.
    Metadata is intentionally small and stable; text is a human-readable block.
    """
    rtype = res.get("resourceType", "Unknown")
    rid = res.get("id")
    meta: Dict[str, Any] = {"resourceType": rtype}
    if rid:
        meta["id"] = rid

    # Pull common identifiers/names to metadata for filtering
    if rtype == "Patient":
        name = _best_human_name(res.get("name"))
        if name:
            meta["patient_name"] = name
        gender = res.get("gender")
        if gender:
            meta["patient_gender"] = gender
        birthDate = res.get("birthDate")
        if birthDate:
            meta["patient_birthDate"] = birthDate

    if "code" in res:
        meta["code_display"] = _codeable_concept_to_text(res["code"])

    if "status" in res and isinstance(res["status"], str):
        meta["status"] = res["status"]

    # Build main text body (FHIR-aware + generic recursive)
    chunks: List[str] = []

    # Narrative div
    narrative = safe_get(res, "text", "div")
    if isinstance(narrative, str) and narrative.strip():
        chunks.append(strip_html(narrative))

    # Titles/headlines for common resource types
    title = _resource_title(res)
    if title:
        chunks.append(title)

    # Summaries for common structures
    chunks.extend(_summaries_for_common_resources(res))

    # Generic recursive walk for leftover human-readable bits
    generic_bits = _generic_human_bits(res)
    if generic_bits:
        chunks.append(generic_bits)

    text = _normalize_text("\n\n".join(p for p in chunks if p))
    return text, meta

def _resource_title(res: Dict[str, Any]) -> str:
    rt = res.get("resourceType")
    rid = res.get("id")
    parts = [rt or "Resource"]
    if rid:
        parts.append(f"({rid})")
    # Try a human label
    label = safe_get(res, "title") or safe_get(res, "name")  # e.g., List.title, Questionnaire.name
    if isinstance(label, str):
        parts.append(f"- {label}")
    return " ".join(parts)

def _summaries_for_common_resources(res: Dict[str, Any]) -> List[str]:
    rt = res.get("resourceType")
    out: List[str] = []

    # Patient
    if rt == "Patient":
        name = _best_human_name(res.get("name"))
        gender = res.get("gender")
        birth = res.get("birthDate")
        lines = []
        if name:
            lines.append(f"Patient: {name}")
        if gender or birth:
            lines.append(join_nonempty([f"Gender: {gender}" if gender else "", f"DOB: {birth}" if birth else ""], " | "))
        if addr := _addresses_to_text(res.get("address")):
            lines.append(f"Address: {addr}")
        if telecom := _telecom_to_text(res.get("telecom")):
            lines.append(f"Contact: {telecom}")
        if lines:
            out.append("\n".join(lines))

    # Observation
    if rt == "Observation":
        code = _codeable_concept_to_text(res.get("code"))
        eff = res.get("effectiveDateTime") or res.get("effectivePeriod")
        val = _observation_value_to_text(res)
        intp = _codeable_concept_to_text(res.get("interpretation"))
        comps = res.get("component") if isinstance(res.get("component"), list) else None
        lines = []
        if code:
            lines.append(f"Observation: {code}")
        if eff:
            lines.append(f"Effective: {json.dumps(eff, ensure_ascii=False)}")
        if val:
            lines.append(f"Value: {val}")
        if intp:
            lines.append(f"Interpretation: {intp}")
        if comps:
            for c in comps:
                ccode = _codeable_concept_to_text(c.get("code"))
                cval = _quantity_like_to_text(c.get("valueQuantity")) or \
                       c.get("valueString") or \
                       _codeable_concept_to_text(c.get("valueCodeableConcept"))
                if ccode or cval:
                    lines.append(f"Component - {join_nonempty([ccode, cval], ': ')}")
        if lines:
            out.append("\n".join(lines))

    # Condition
    if rt == "Condition":
        code = _codeable_concept_to_text(res.get("code"))
        onset = res.get("onsetDateTime") or res.get("onsetPeriod")
        verification = res.get("verificationStatus", {}).get("text") or _codeable_concept_to_text(res.get("verificationStatus"))
        clinical = res.get("clinicalStatus", {}).get("text") or _codeable_concept_to_text(res.get("clinicalStatus"))
        lines = []
        if code:
            lines.append(f"Condition: {code}")
        if clinical:
            lines.append(f"Clinical Status: {clinical}")
        if verification:
            lines.append(f"Verification Status: {verification}")
        if onset:
            lines.append(f"Onset: {json.dumps(onset, ensure_ascii=False)}")
        if lines:
            out.append("\n".join(lines))

    # MedicationStatement / MedicationRequest
    if rt in ("MedicationStatement", "MedicationRequest"):
        med = _medication_to_text(res.get("medicationCodeableConcept") or res.get("medication"))
        dosage = _dosage_to_text(res.get("dosageInstruction"))
        when = res.get("authoredOn") or res.get("effectiveDateTime") or res.get("effectivePeriod")
        parts = []
        if med:
            parts.append(f"Medication: {med}")
        if dosage:
            parts.append(f"Dosage: {dosage}")
        if when:
            parts.append(f"When: {json.dumps(when, ensure_ascii=False)}")
        if parts:
            out.append("\n".join(parts))

    # Procedure
    if rt == "Procedure":
        code = _codeable_concept_to_text(res.get("code"))
        perf = _reference_to_text(res.get("performer"))
        period = res.get("performedDateTime") or res.get("performedPeriod")
        if code or perf or period:
            out.append("\n".join([
                f"Procedure: {code}" if code else "",
                f"Performed by: {perf}" if perf else "",
                f"When: {json.dumps(period, ensure_ascii=False)}" if period else "",
            ]).strip())

    return [s for s in out if s]

def _best_human_name(names: Any) -> str:
    if not isinstance(names, list):
        return ""
    # Prefer official or usual
    def to_str(n):
        parts = []
        if isinstance(n, dict):
            if n.get("text"):
                return n["text"]
            given = " ".join(n.get("given", [])) if isinstance(n.get("given"), list) else n.get("given")
            family = n.get("family")
            parts = [p for p in [given, family] if p]
        return " ".join(parts)
    preferred = sorted(names, key=lambda n: (n.get("use") not in ("official", "usual"), 1))
    for n in preferred:
        s = to_str(n)
        if s:
            return s
    return ""

def _addresses_to_text(addresses: Any) -> str:
    if not isinstance(addresses, list):
        return ""
    parts = []
    for a in addresses:
        line = ", ".join(a.get("line", [])) if isinstance(a.get("line"), list) else a.get("line")
        city = a.get("city")
        state = a.get("state")
        postal = a.get("postalCode")
        country = a.get("country")
        parts.append(", ".join([p for p in [line, city, state, postal, country] if p]))
    return "; ".join(filter(None, parts))

def _telecom_to_text(telecom: Any) -> str:
    if not isinstance(telecom, list):
        return ""
    return "; ".join([join_nonempty([t.get("system"), t.get("value")], ": ") for t in telecom if isinstance(t, dict)])

def _codeable_concept_to_text(cc: Any) -> str:
    if isinstance(cc, list):
        return "; ".join(filter(None, (_codeable_concept_to_text(x) for x in cc)))
    if not isinstance(cc, dict):
        return ""
    parts = []
    if "text" in cc and isinstance(cc["text"], str):
        parts.append(cc["text"])
    codings = cc.get("coding")
    if isinstance(codings, list):
        for c in codings:
            disp = c.get("display")
            code = c.get("code")
            system = c.get("system")
            if disp:
                parts.append(disp)
            elif code:
                parts.append(code if not system else f"{code} ({system})")
    return "; ".join(dict.fromkeys([p for p in parts if p]))  # de-dup, keep order

def _reference_to_text(ref: Any) -> str:
    if isinstance(ref, list):
        return "; ".join(filter(None, (_reference_to_text(x) for x in ref)))
    if isinstance(ref, dict):
        return ref.get("display") or ref.get("reference") or ""
    if isinstance(ref, str):
        return ref
    return ""

def _quantity_like_to_text(q: Any) -> str:
    if not isinstance(q, dict):
        return ""
    val = q.get("value")
    unit = q.get("unit") or q.get("code")
    if val is None and not unit:
        return ""
    return f"{val} {unit}".strip()

def _observation_value_to_text(obs: Dict[str, Any]) -> str:
    if "valueQuantity" in obs:
        return _quantity_like_to_text(obs["valueQuantity"])
    for k in ("valueString", "valueTime", "valueDateTime", "valueInteger", "valueBoolean"):
        if k in obs:
            return str(obs[k])
    if "valueCodeableConcept" in obs:
        return _codeable_concept_to_text(obs["valueCodeableConcept"])
    return ""

def _dosage_to_text(dosage: Any) -> str:
    if not isinstance(dosage, list):
        return ""
    out = []
    for d in dosage:
        text = d.get("text")
        route = _codeable_concept_to_text(d.get("route"))
        dose = _quantity_like_to_text(d.get("doseQuantity"))
        freq = ""
        if "timing" in d and isinstance(d["timing"], dict):
            rep = d["timing"].get("repeat", {})
            times = rep.get("frequency")
            per = rep.get("period")
            per_unit = rep.get("periodUnit")
            if times and per and per_unit:
                freq = f"{times} times per {per} {per_unit}"
        parts = [text, route, dose, freq]
        line = ", ".join([p for p in parts if p])
        if line:
            out.append(line)
    return "; ".join(out)

def _medication_to_text(m: Any) -> str:
    if isinstance(m, dict):
        if "text" in m:
            return m["text"] if isinstance(m["text"], str) else _codeable_concept_to_text(m["text"])
        if "coding" in m or "code" in m:
            return _codeable_concept_to_text(m)
    return _codeable_concept_to_text(m)

def _normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _generic_human_bits(node: Any, path: str = "", depth: int = 0) -> str:
    """
    Conservative recursive extraction of human-friendly strings while
    skipping highly technical keys and very deep recursion.
    """
    if depth > 6:
        return ""
    SKIP_KEYS = {
        "meta", "implicitRules", "modifierExtension", "extension", "contained",
        "identifier", "id", "resourceType", "language", "versionId", "lastUpdated",
        "security", "tag", "statusHistory", "note", "hash", "signature", "data"
    }
    PREFER_KEYS = {"text", "display", "description", "summary", "title", "name", "value", "reason", "comment"}

    out: List[str] = []

    if isinstance(node, dict):
        # Prefer readable fields first
        for k, v in node.items():
            if k in SKIP_KEYS:
                continue
            if isinstance(v, str) and k == "div" and (node.get("text") is node):  # avoid double narrative
                continue
            if k in ("text",) and isinstance(v, dict) and "div" in v:
                out.append(strip_html(v["div"]))
        # Preferred key strings
        for k, v in node.items():
            if k in SKIP_KEYS:
                continue
            if k in PREFER_KEYS and isinstance(v, str):
                out.append(v)
            elif k in {"code", "category", "type", "method", "bodySite", "reasonCode"}:
                out.append(_codeable_concept_to_text(v))
            elif k in {"valueCodeableConcept"}:
                out.append(_codeable_concept_to_text(v))
            elif k == "valueQuantity":
                out.append(_quantity_like_to_text(v))
            elif k in {"reference", "subject", "encounter", "requester", "performer"}:
                out.append(_reference_to_text(v))

        # Generic descent
        for k, v in node.items():
            if k in SKIP_KEYS:
                continue
            out.append(_generic_human_bits(v, f"{path}.{k}" if path else k, depth + 1))

    elif isinstance(node, list):
        for i, item in enumerate(node):
            out.append(_generic_human_bits(item, f"{path}[{i}]", depth + 1))
    elif isinstance(node, str):
        out.append(node if len(node) <= 5000 else node[:5000] + " …")
    elif is_primitive(node):
        out.append(str(node))

    combined = "\n".join(s for s in out if isinstance(s, str) and s.strip())
    return _normalize_text(combined)

# -------------- Chunking --------------

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    if not text:
        return []
    if max_chars <= 0:
        return [text]
    out = []
    start = 0
    n = len(text)
    step = max(1, max_chars - overlap)
    while start < n:
        end = min(n, start + max_chars)
        # try to break at nearest newline or sentence end for readability
        slice_ = text[start:end]
        break_pt = max(slice_.rfind("\n"), slice_.rfind(". "), slice_.rfind("; "))
        if break_pt < int(0.5 * len(slice_)):
            break_pt = len(slice_)
        chunk = slice_[:break_pt].rstrip()
        if not chunk:
            chunk = slice_
        out.append(chunk)
        start += step
    return out

# -------------- IO --------------

def iter_fhir_resources_from_file(path: str) -> Iterable[Tuple[Dict[str, Any], int]]:
    """
    Yields (resource, line_no) pairs.
    For JSON files: line_no = 0; for NDJSON: actual 1-based line numbers.
    """
    _, ext = os.path.splitext(path.lower())
    if ext in (".ndjson", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield from _extract_resources(obj, line_no=i)
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        yield from _extract_resources(obj, line_no=0)

def _extract_resources(obj: Dict[str, Any], line_no: int) -> Iterable[Tuple[Dict[str, Any], int]]:
    if not isinstance(obj, dict):
        return
    if obj.get("resourceType") == "Bundle" and isinstance(obj.get("entry"), list):
        for e in obj["entry"]:
            res = e.get("resource")
            if isinstance(res, dict):
                yield res, line_no
    else:
        if obj.get("resourceType"):
            yield obj, line_no

def write_jsonl(records: Iterable[Dict[str, Any]], out_path: str):
    with (open(out_path, "w", encoding="utf-8") if out_path != "-" else sys.stdout) as w:
        first = True
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False)
            if out_path != "-" and not first:
                w.write("\n")
            elif out_path == "-" and not first:
                w.write("\n")
            w.write(line)
            first = False

# -------------- Main pipeline --------------

def process_paths(paths: List[str], out_path: str, max_chars: int, overlap: int) -> int:
    recs: List[Dict[str, Any]] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith((".json", ".ndjson", ".jsonl")):
                        recs.extend(process_file(os.path.join(root, f), max_chars, overlap))
        else:
            recs.extend(process_file(p, max_chars, overlap))
    write_jsonl(recs, out_path)
    return len(recs)

def process_file(path: str, max_chars: int, overlap: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for res, line_no in iter_fhir_resources_from_file(path):
        text, meta = extract_text_from_resource(res)
        if not text:
            continue
        rtype = res.get("resourceType", "Unknown")
        rid = res.get("id") or ""
        doc_id = f"{os.path.basename(path)}::{rtype}/{rid or 'no-id'}::{line_no}"
        chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
        for i, ch in enumerate(chunks):
            out.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}::chunk-{i+1}",
                "resource_type": rtype,
                "fhir_id": rid,
                "source_file": path,
                "line_no": line_no,
                "metadata": meta,
                "text": ch
            })
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert FHIR JSON/NDJSON to RAG-ready JSONL with overlapping chunks.")
    ap.add_argument("inputs", nargs="*", help="Input files or directories (FHIR JSON, Bundles, or NDJSON). If omitted, defaults to datasets/fhir.")
    ap.add_argument("--out", default=None, help="Output JSONL path (default: datasets/fhir/rag_chunks.jsonl; use '-' for stdout).")
    ap.add_argument("--max-chars", type=int, default=2000, help="Max characters per chunk (default: 2000).")
    ap.add_argument("--overlap", type=int, default=200, help="Character overlap between chunks (default: 200).")
    args = ap.parse_args()

    # Determine defaults relative to this file, so it works from anywhere
    repo_root = os.path.abspath(os.path.dirname(__file__))
    default_input_dir = os.path.join(repo_root, "datasets", "fhir")
    default_out_path = os.path.join(default_input_dir, "rag_chunks.jsonl")

    inputs = args.inputs if args.inputs else [default_input_dir]
    out_path = default_out_path if args.out is None else args.out

    if args.overlap >= args.max_chars:
        print("--overlap must be smaller than --max-chars", file=sys.stderr)
        sys.exit(2)

    total = process_paths(inputs, out_path, args.max_chars, args.overlap)
    if out_path != "-":
        print(f"Wrote {total} chunks to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()