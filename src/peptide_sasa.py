#!/usr/bin/env python3
"""Compute bound, free, and buried SASA for every peptide residue.

The free calculation uses the peptide coordinates from the bound structure
without relaxation. Thus buried SASA is a purely geometric quantity:
``free peptide SASA - bound peptide SASA``.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from Bio import __version__ as BIOPYTHON_VERSION
from Bio.PDB import MMCIFParser, ShrakeRupley
from tqdm import tqdm

from io_utils import file_sha256, read_jsonl, source_code_identity
from pmhc_mhci_common import (
    AA3_TO_1,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_PDB_ROOT,
    STANDARD_AA,
    path_for_pdb_id,
    normalize_missing,
)


DEFAULT_INPUT = DEFAULT_ARTIFACT_DIR / "pmhc_mhci_distance_matrices.jsonl"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "peptide_sasa.jsonl"

# Maximum accessible areas (A^2) from Tien et al. These put residues of
# different sizes on the common relative-SASA scale used in the paper.
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}


def is_paper_complex(record: dict) -> bool:
    """Select fully mapped canonical-sequence complexes before SASA."""
    pseudo_mapping = record.get("pseudosequence_residue_mappings", [])
    screen = record.get("screen_flags", {})
    sequence = record.get("pseudosequence", "") + record.get("peptide", "")
    return (
        len(pseudo_mapping) == 34
        and all(row.get("mapped") for row in pseudo_mapping)
        and all(residue in STANDARD_AA for residue in sequence)
        and not screen.get("peptide_unbound_to_mhc")
        and not screen.get("pseudoseq_pocket_missed")
    )


SASA_ALGORITHM_VERSION = "label-polymer-heavy-v2"
CANONICAL_RESIDUE_NAMES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)


def record_id(record: dict) -> str:
    return "|".join((
        str(record.get("pdb_id", "")),
        str((record.get("mhc") or {}).get("asym_id", "")),
        str((record.get("peptide_chain") or {}).get("asym_id", "")),
        str(record.get("peptide", "")),
    ))


def sasa_code_identity() -> dict:
    """Identify the SASA calculation and its local code dependencies."""
    return source_code_identity(
        Path(__file__).with_name(name)
        for name in ("peptide_sasa.py", "pmhc_mhci_common.py", "io_utils.py")
    )


def scientific_record_sha256(record: dict) -> str:
    """Hash structure-selection content while ignoring machine-local source paths."""
    portable = {key: value for key, value in record.items() if key not in {"source_path", "line_index"}}
    text = json.dumps(portable, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(text.encode()).hexdigest()


def structure_path(record: dict, pdb_root: str | Path) -> Path:
    """Resolve from the requested mirror so its provenance is authoritative."""
    return path_for_pdb_id(pdb_root, str(record["pdb_id"]))


def peptide_mapping(record: dict) -> list[dict]:
    mappings = record.get("peptide_residue_mappings")
    if not mappings:
        raise ValueError("Exact peptide residue mappings are required for SASA")
    return sorted(mappings, key=lambda row: int(row["peptide_index"]))


def mapped_chain(mappings: list[dict], fallback: str) -> str:
    """Return a label asym ID; author chain IDs occupy a different namespace."""
    labels = {
        normalize_missing(row.get("label_asym_id"))
        for row in mappings if normalize_missing(row.get("label_asym_id"))
    }
    fallback = normalize_missing(fallback)
    if len(labels) > 1 or (labels and fallback and labels != {fallback}):
        raise ValueError("Conflicting label chain IDs in residue mappings")
    return next(iter(labels), fallback)


def residue_lookup(chain) -> dict[int, list[Any]]:
    """Index only deposited label sequence IDs, including modified polymers."""
    lookup: dict[int, list[Any]] = {}
    for residue in chain:
        if residue.is_disordered() == 2:
            raise ValueError(f"Ambiguous polymer residue identity at {chain.id}:{residue.id}")
        lookup.setdefault(int(residue.id[1]), []).append(residue)
    return lookup


def find_residue(chain, mapping: dict):
    """Resolve one exact label identity; never infer it from peptide order."""
    label_chain = normalize_missing(mapping.get("label_asym_id"))
    if label_chain and label_chain != str(chain.id):
        raise ValueError("Residue mapping refers to a different label chain")
    explicit = normalize_missing(mapping.get("label_seq_id"))
    legacy = normalize_missing(mapping.get("structure_seq_id"))
    if explicit and legacy and explicit != legacy:
        raise ValueError("Conflicting label sequence IDs in residue mapping")
    label_seq_id = explicit or legacy
    if not label_seq_id:
        raise ValueError("Residue mapping has no deposited label sequence ID")
    try:
        sequence_number = int(label_seq_id)
    except ValueError as error:
        raise ValueError(f"Invalid label sequence ID {label_seq_id!r}") from error
    candidates = residue_lookup(chain).get(sequence_number, [])
    insertion_field = next(
        (name for name in ("auth_insertion_code", "insertion_code") if name in mapping),
        None,
    )
    if insertion_field is not None:
        insertion = normalize_missing(mapping[insertion_field])
        candidates = [r for r in candidates if normalize_missing(r.id[2]) == insertion]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous label residue {chain.id}:{label_seq_id}")
    if not candidates:
        return None
    residue = candidates[0]
    expected_name = normalize_missing(mapping.get("residue_name")).upper()
    if expected_name and expected_name != residue.resname.strip().upper():
        raise ValueError(f"Component identity mismatch at {chain.id}:{label_seq_id}")
    return residue


def keep_chains(model, chain_ids: set[str]):
    """Keep polymer heavy atoms, using the distance builder's altloc policy.

    Parsing with auth_residues=False already excludes waters/nonpolymers that
    have no label_seq_id. HETATM polymer residues (e.g. phosphoserine) remain.
    """
    result = copy.deepcopy(model)
    for chain in list(result):
        if chain.id not in chain_ids:
            result.detach_child(chain.id)
            continue
        for residue in list(chain):
            if residue.is_disordered() == 2:
                raise ValueError(f"Ambiguous polymer residue identity at {chain.id}:{residue.id}")
            for atom in list(residue):
                variants = atom.disordered_get_list() if atom.is_disordered() == 2 else [atom]
                allowed = [
                    candidate for candidate in variants
                    if normalize_missing(candidate.altloc) in {"", "A", "1"}
                    and str(candidate.element or "").strip().upper() not in {"H", "D"}
                ]
                if len(allowed) > 1:
                    raise ValueError(f"Ambiguous accepted atom conformers at {chain.id}:{residue.id}:{atom.id}")
                if not allowed:
                    residue.detach_child(atom.id)
                elif atom.is_disordered() == 2:
                    # Replace the wrapper so no rejected conformer can enter SASA.
                    chosen = copy.deepcopy(allowed[0])
                    residue.detach_child(atom.id)
                    residue.add(chosen)
            if not len(residue):
                chain.detach_child(residue.id)
        if not len(chain):
            result.detach_child(chain.id)
    return result


def residue_sasa(residue) -> float:
    return sum(float(atom.sasa) for atom in residue)


def compute_peptide_sasa(
    record: dict,
    pdb_root: str | Path = DEFAULT_PDB_ROOT,
    probe_radius: float = 1.4,
    n_points: int = 200,
    clip_negative_buried_sasa: bool = False,
) -> dict:
    """Compute both states from exact first-model polymer heavy-atom identities."""
    path = structure_path(record, pdb_root)
    mappings = peptide_mapping(record)
    mhc_chain_id = mapped_chain(
        record.get("pseudosequence_residue_mappings", []),
        (record.get("mhc") or {}).get("asym_id", ""),
    )
    peptide_chain_id = mapped_chain(mappings, (record.get("peptide_chain") or {}).get("asym_id", ""))
    if not mhc_chain_id or not peptide_chain_id or mhc_chain_id == peptide_chain_id:
        raise ValueError("Distinct deposited label MHC and peptide chains are required")
    parser = MMCIFParser(QUIET=True, auth_chains=False, auth_residues=False)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as handle:
        model = parser.get_structure(str(record.get("pdb_id", "pmhc")), handle)[0]
    bound_model = keep_chains(model, {mhc_chain_id, peptide_chain_id})
    if mhc_chain_id not in bound_model or peptide_chain_id not in bound_model:
        raise ValueError(f"Polymer chains {mhc_chain_id!r}/{peptide_chain_id!r} not found in {path}")
    bound_peptide = bound_model[peptide_chain_id]
    ShrakeRupley(probe_radius=probe_radius, n_points=n_points).compute(bound_model, level="A")
    free_peptide = copy.deepcopy(bound_peptide)
    ShrakeRupley(probe_radius=probe_radius, n_points=n_points).compute(free_peptide, level="A")

    rows, excluded, seen_indices, seen_residues = [], [], set(), set()
    for mapping in mappings:
        index = int(mapping["peptide_index"])
        if index in seen_indices or not 0 <= index < len(record["peptide"]):
            raise ValueError(f"Invalid or duplicate peptide index {index}")
        seen_indices.add(index)
        if not mapping.get("mapped", False):
            excluded.append({"peptide_index": index, "reason": "unmapped_coordinate_residue"})
            continue
        bound_residue = find_residue(bound_peptide, mapping)
        if bound_residue is None:
            excluded.append({"peptide_index": index, "reason": "label_coordinate_residue_missing"})
            continue
        if bound_residue.id in seen_residues:
            raise ValueError("Multiple peptide positions resolve to the same coordinate residue")
        seen_residues.add(bound_residue.id)
        free_residue = free_peptide[bound_residue.id]
        expected = str(mapping.get("expected_residue") or record["peptide"][index]).upper()
        residue_name = bound_residue.resname.strip().upper()
        canonical = residue_name in CANONICAL_RESIDUE_NAMES
        residue = AA3_TO_1.get(residue_name, "X")
        if canonical and expected in MAX_ASA and expected != residue:
            raise ValueError(f"Sequence/component mismatch at peptide position {index}: {expected}/{residue_name}")
        bound, free = residue_sasa(bound_residue), residue_sasa(free_residue)
        buried = free - bound
        if clip_negative_buried_sasa:
            buried = max(0.0, buried)
        rows.append({
            "peptide_index": index,
            "residue": residue,
            "expected_residue": expected,
            "residue_name": residue_name,
            "modified_residue": not canonical,
            "label_asym_id": peptide_chain_id,
            "label_seq_id": str(bound_residue.id[1]),
            "auth_seq_id": str(mapping.get("auth_seq_id", "")),
            "auth_insertion_code": normalize_missing(bound_residue.id[2]),
            "coordinate_residue_id": list(bound_residue.id),
            "atom_count": len(bound_residue),
            "bound_sasa": bound,
            "free_peptide_sasa": free,
            "buried_sasa": buried,
            "bound_relative_sasa": bound / MAX_ASA[residue] if canonical else None,
            "normalization_exclusion_reason": None if canonical else "no_canonical_max_asa_for_modified_residue",
        })
    for index in sorted(set(range(len(record["peptide"]))) - seen_indices):
        excluded.append({"peptide_index": index, "reason": "residue_mapping_missing"})
    if not rows:
        raise ValueError("No mapped peptide residues produced SASA values")
    return {
        "record_id": record_id(record),
        "pdb_id": record.get("pdb_id"),
        "allele": record.get("allele"),
        "peptide": record.get("peptide"),
        "source_path": str(path),
        "source_sha256": file_sha256(path),
        "distance_record_sha256": scientific_record_sha256(record),
        "sasa_algorithm_version": SASA_ALGORITHM_VERSION,
        "sasa_code": sasa_code_identity(),
        "biopython_version": BIOPYTHON_VERSION,
        "coordinate_policy": "first_model;label_asym_id+label_seq_id;polymer_heavy_atoms;altloc_blank_A_1",
        "mhc_chain_id": mhc_chain_id,
        "peptide_chain_id": peptide_chain_id,
        "probe_radius": probe_radius,
        "n_points": n_points,
        "clip_negative_buried_sasa": clip_negative_buried_sasa,
        "excluded_residues": excluded,
        "residues": rows,
        "bound_sasa": [row["bound_sasa"] for row in rows],
        "free_peptide_sasa": [row["free_peptide_sasa"] for row in rows],
        "buried_sasa": [row["buried_sasa"] for row in rows],
        "bound_relative_sasa": [row["bound_relative_sasa"] for row in rows],
    }


def _worker(task: tuple[int, dict, str]) -> dict:
    line_index, record, pdb_root = task
    try:
        result = compute_peptide_sasa(record, pdb_root)
        result["line_index"] = line_index
        return {"ok": True, "result": result}
    except Exception as error:
        return {
            "ok": False,
            "line_index": line_index,
            "record_id": record_id(record),
            "pdb_id": record.get("pdb_id"),
            "peptide": record.get("peptide"),
            "error": f"{type(error).__name__}: {error}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pdb-root", type=Path, default=DEFAULT_PDB_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors-output", type=Path, help="Default: beside --output")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.errors_output = args.errors_output or args.output.with_name("peptide_sasa_errors.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.errors_output.parent.mkdir(parents=True, exist_ok=True)
    complexes = residues = errors = 0

    # Count selected complexes for the progress bar.
    total = sum(
        is_paper_complex(record) for _, record in read_jsonl(args.records)
    )
    tasks = (
        (line_index, record, str(args.pdb_root))
        for line_index, record in read_jsonl(args.records)
        if is_paper_complex(record)
    )

    # Each run overwrites the output.
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    results = executor.map(_worker, tasks, chunksize=4) if executor else map(_worker, tasks)
    try:
        with args.output.open("w", encoding="utf-8") as output, args.errors_output.open(
            "w", encoding="utf-8"
        ) as error_output, tqdm(
            results, total=total, desc="Peptide SASA", unit="complex"
        ) as progress:
            for item in progress:
                if item["ok"]:
                    result = item["result"]
                    output.write(json.dumps(result, separators=(",", ":")) + "\n")
                    complexes += 1
                    residues += len(result["residues"])
                else:
                    errors += 1
                    error_output.write(json.dumps(item, separators=(",", ":")) + "\n")
                    tqdm.write(f"SASA failed for line {item['line_index']} ({item['pdb_id']}): {item['error']}")
                progress.set_postfix(residues=f"{residues:,}", errors=errors)
    finally:
        if executor is not None:
            executor.shutdown()

    print(
        f"wrote {residues:,} peptide residues from {complexes:,} complexes "
        f"to {args.output}; {errors:,} structures failed (see {args.errors_output})"
    )


if __name__ == "__main__":
    main()
