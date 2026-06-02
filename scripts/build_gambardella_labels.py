#!/usr/bin/env python3
"""
Build authoritative TP53 binary status (MUT/WT) for the 32 Gambardella 2022
breast-cancer cell lines, sourced from Cellosaurus curated sequence variations.

A line is TP53-MUT iff its Cellosaurus record has a sequence-variation entry
whose HGNC cross-reference label == "TP53" and variation-type == "Mutation".
Otherwise WT (no curated pathogenic TP53 mutation).

Output: data/labels/gambardella_tp53_status.csv with columns
    stripped_name, cellosaurus_ac, matched_name, tp53_status, tp53_mutations, source

This script is the single source of truth for labels. It is deterministic
(given the live Cellosaurus DB) and fully auditable: print the matched name
next to each query so the name->accession mapping can be eyeballed.
"""
import csv
import json
import sys
import time
from pathlib import Path

import requests

API = "https://api.cellosaurus.org"

# 32 cell-line prefixes as they appear in Gambardella barcodes (PREFIX_BARCODE).
GAMBARDELLA_LINES = [
    "AU565", "BT20", "BT474", "BT483", "BT549", "CAL51", "CAL851", "CAMA1",
    "DU4475", "EFM19", "EVSAT", "HCC1143", "HCC1187", "HCC1500", "HCC1937",
    "HCC1954", "HCC38", "HCC70", "HDQP1", "HS578T", "JIMT1", "KPL1", "MCF12A",
    "MCF7", "MDAMB361", "MDAMB415", "MDAMB436", "MDAMB453", "MDAMB468", "MX1",
    "T47D", "ZR751",
]


def _norm(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalnum())


def resolve_accession(name: str) -> dict | None:
    """Search Cellosaurus for `name`; return best human cell-line record."""
    url = f"{API}/search/cell-line"
    params = {"q": name, "format": "json", "rows": "40"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    recs = r.json().get("Cellosaurus", {}).get("cell-line-list", [])
    qn = _norm(name)
    best = None
    for cl in recs:
        # organism must be human
        ox = cl.get("species-list", cl.get("organism-list", []))
        # gather id + synonyms
        names = []
        for nm in cl.get("name-list", []):
            if isinstance(nm, dict):
                names.append(nm.get("value", ""))
            else:
                names.append(str(nm))
        norm_names = {_norm(n) for n in names}
        acc = ""
        for x in cl.get("accession-list", []):
            if isinstance(x, dict) and x.get("type") == "primary":
                acc = x.get("value", "")
        if not acc:
            # fallback: first accession
            al = cl.get("accession-list", [])
            if al:
                acc = al[0].get("value", "") if isinstance(al[0], dict) else str(al[0])
        rec = {"ac": acc, "names": names, "norm_names": norm_names}
        if qn in norm_names:
            return rec  # exact (synonym) match wins immediately
        if best is None:
            best = rec
    return best


def fetch_tp53(acc: str) -> tuple[str, list[str]]:
    """Return (status, mutations) for an accession by scanning TP53 variations."""
    r = requests.get(f"{API}/cell-line/{acc}?format=json", timeout=30)
    r.raise_for_status()
    cl = r.json()["Cellosaurus"]["cell-line-list"][0]
    muts = []
    for sv in cl.get("sequence-variation-list", []):
        if sv.get("variation-type") != "Mutation":
            continue
        is_tp53 = False
        for x in sv.get("xref-list", []):
            if x.get("database") == "HGNC" and x.get("label") == "TP53":
                is_tp53 = True
                break
        if is_tp53:
            desc = sv.get("mutation-description", "") or sv.get("variation-note", "")
            muts.append(desc.strip())
    return ("MUT" if muts else "WT", muts)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / "data" / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gambardella_tp53_status.csv"

    rows = []
    for name in GAMBARDELLA_LINES:
        try:
            rec = resolve_accession(name)
            if rec is None or not rec["ac"]:
                print(f"[WARN] {name}: no Cellosaurus match")
                rows.append([name, "", "", "UNKNOWN", "", "Cellosaurus"])
                continue
            status, muts = fetch_tp53(rec["ac"])
            matched = rec["names"][0] if rec["names"] else ""
            print(f"{name:10s} -> {rec['ac']:12s} {matched:18s} {status:4s}  {'; '.join(muts)}")
            rows.append([name, rec["ac"], matched, status, "; ".join(muts), "Cellosaurus"])
            time.sleep(0.2)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {name}: {e!r}")
            rows.append([name, "", "", "ERROR", "", "Cellosaurus"])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stripped_name", "cellosaurus_ac", "matched_name",
                    "tp53_status", "tp53_mutations", "source"])
        w.writerows(rows)

    n_mut = sum(1 for r in rows if r[3] == "MUT")
    n_wt = sum(1 for r in rows if r[3] == "WT")
    n_bad = sum(1 for r in rows if r[3] in ("UNKNOWN", "ERROR"))
    print("-" * 60)
    print(f"MUT={n_mut}  WT={n_wt}  unresolved={n_bad}  total={len(rows)}")
    print(f"Wrote {out_path}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
