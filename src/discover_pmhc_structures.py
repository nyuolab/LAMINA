#!/usr/bin/env python3
"""Find class-I MHC heavy chains paired with 8-14 residue peptides.

Run this on a local copy of the weekly wwPDB mmCIF archive. The output has one
JSON line per candidate heavy-chain entity / peptide entity. Actual chain
pairing and physical contact are handled by build_pmhc_distance_matrices.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from pmhc_mhci_common import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_PDB_ROOT,
    clean_sequence,
    extract_hla_alleles,
    iter_structure_paths,
    parse_mmcif_categories,
    pdb_id_from_path,
)


DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "pmhc_mhci_candidates.jsonl"
METADATA_CATEGORIES = {
    "_entry", "_struct", "_struct_keywords", "_entity", "_entity_poly",
    "_struct_asym",
}


def first_value(categories: dict, category: str, field: str) -> str:
    return next((row[field] for row in categories.get(category, []) if row.get(field)), "")


def build_entities(categories: dict) -> dict[str, dict]:
    """Join entity descriptions, polymer sequences, and label chain IDs."""
    entities = {
        row["id"]: {
            "entity_id": row["id"],
            "description": row.get("pdbx_description", ""),
            "sequence": "",
            "poly_type": "",
            "asym_ids": [],
        }
        for row in categories.get("_entity", [])
        if row.get("id")
    }
    for row in categories.get("_entity_poly", []):
        entity_id = row.get("entity_id")
        if entity_id:
            entity = entities.setdefault(entity_id, {"entity_id": entity_id})
            entity["sequence"] = clean_sequence(row.get("pdbx_seq_one_letter_code_can"))
            entity["poly_type"] = row.get("type", "")

    chains: dict[str, list[str]] = defaultdict(list)
    for row in categories.get("_struct_asym", []):
        if row.get("entity_id") and row.get("id"):
            chains[row["entity_id"]].append(row["id"])
    for entity_id, entity in entities.items():
        entity["asym_ids"] = chains.get(entity_id, [])
    return entities


def is_mhc_i(entity: dict) -> bool:
    """Use the archive annotation or the conserved HLA N terminus."""
    description = entity.get("description", "")
    sequence = entity.get("sequence", "")
    annotated = any((
        extract_hla_alleles(description),
        re.search(r"\bHLA[-_\s]?[ABC]\b", description, re.IGNORECASE),
        re.search(r"\bMHC\b.*\bCLASS\s*I\b|\bCLASS\s*I\b.*\bMHC\b", description, re.IGNORECASE),
        re.search(r"\bH-2[-_\s]?[A-Z0-9]+", description, re.IGNORECASE),
    ))
    sequence_signature = "GSHSMRY" in sequence[:30] and 170 <= len(sequence) <= 380
    return "polypeptide" in entity.get("poly_type", "").lower() and bool(annotated or sequence_signature)


def is_peptide(entity: dict) -> bool:
    sequence = entity.get("sequence", "")
    return (
        "polypeptide" in entity.get("poly_type", "").lower()
        and 8 <= len(sequence) <= 14
        and all(residue in "ACDEFGHIKLMNPQRSTVWYUXO" for residue in sequence)
    )


def discover_one(path_string: str) -> list[dict]:
    """Read only metadata; the much larger atom table is skipped here."""
    path = Path(path_string)
    categories = parse_mmcif_categories(
        path, METADATA_CATEGORIES, stop_before_categories={"_atom_site"}
    )
    pdb_id = first_value(categories, "_entry", "id") or pdb_id_from_path(path)
    title = first_value(categories, "_struct", "title")
    keywords = " ".join(filter(None, (
        first_value(categories, "_struct_keywords", "pdbx_keywords"),
        first_value(categories, "_struct_keywords", "text"),
    )))
    entities = build_entities(categories)
    mhc_entities = [entity for entity in entities.values() if is_mhc_i(entity)]
    peptides = [entity for entity in entities.values() if is_peptide(entity)]

    records = []
    for mhc in mhc_entities:
        allele_text = " ".join((mhc.get("description", ""), title, keywords))
        for peptide in peptides:
            if peptide["entity_id"] != mhc["entity_id"]:
                records.append({
                    "pdb_id": pdb_id.upper(),
                    "mhc": {
                        "entity_id": mhc["entity_id"],
                        "sequence": mhc["sequence"],
                        "asym_ids": mhc["asym_ids"],
                        "allele_candidates": extract_hla_alleles(allele_text),
                    },
                    "peptide": {
                        "entity_id": peptide["entity_id"],
                        "sequence": peptide["sequence"],
                        "asym_ids": peptide["asym_ids"],
                    },
                })
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-root", type=Path, default=DEFAULT_PDB_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="mmCIF files parsed in parallel",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [str(path) for path in iter_structure_paths(args.pdb_root)]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # imap preserves the sorted archive order, so JSONL line numbers remain
    # stable and can be used to join the later SASA and FoldX tables.
    if args.workers == 1:
        results = map(discover_one, paths)
        pool = None
    else:
        pool = Pool(args.workers)
        results = pool.imap(discover_one, paths, chunksize=64)

    candidates = 0
    try:
        with args.output.open("w", encoding="utf-8") as output:
            for records in tqdm(
                results, total=len(paths), desc="Scanning mmCIF files", unit="file"
            ):
                for record in records:
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
                    candidates += 1
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"scanned {len(paths):,} mmCIF files; wrote {candidates:,} candidates to {args.output}")


if __name__ == "__main__":
    main()
