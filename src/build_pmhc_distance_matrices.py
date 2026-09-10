#!/usr/bin/env python3
"""Map each candidate to a 34 x peptide heavy-atom distance matrix.

The two screen flags written here are the fixed exclusions in the paper:
peptides detached from the MHC and peptides contacting the MHC outside the 34
pseudosequence residues. The resulting JSONL is the input to attribution,
SASA, and FoldX analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from io_utils import file_sha256, read_jsonl
from hla_residue_mapping import HlaMappingError, REFERENCE_PATH, align_hla_to_reference, validate_coordinate_residue
from pmhc_mhci_common import (
    AA3_TO_1,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_PDB_ROOT,
    DEFAULT_PSEUDOSEQ_PATH,
    MHC_I_PSEUDO_POSITIONS,
    clean_sequence,
    load_pseudoseqs,
    lookup_pseudoseq,
    normalize_missing,
    parse_float,
    parse_mmcif_categories,
    path_for_pdb_id,
)


DEFAULT_INPUT = DEFAULT_ARTIFACT_DIR / "pmhc_mhci_candidates.jsonl"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "pmhc_mhci_distance_matrices.jsonl"
DETAIL_CATEGORIES = {
    "_entity_poly", "_struct_asym", "_pdbx_poly_seq_scheme", "_atom_site",
}

# Fixed paper screens, in Angstroms or fractions of peptide positions.
CONTACT_CUTOFF = 4.0
UNBOUND_DISTANCE = 6.0
PSEUDO_FAR_DISTANCE = 5.5
PSEUDO_DISTANCE_GAP = 2.0
MIN_MHC_CONTACT_FRACTION = 0.20
MAX_PSEUDO_CONTACT_FRACTION = 0.15


def entity_sequences(categories: dict) -> dict[str, str]:
    return {
        row["entity_id"]: clean_sequence(row.get("pdbx_seq_one_letter_code_can"))
        for row in categories.get("_entity_poly", [])
        if row.get("entity_id")
    }


def entity_chains(categories: dict) -> dict[str, list[str]]:
    chains: dict[str, list[str]] = defaultdict(list)
    for row in categories.get("_struct_asym", []):
        if row.get("entity_id") and row.get("id"):
            chains[row["entity_id"]].append(row["id"])
    return dict(chains)


def scheme_index(categories: dict) -> dict[tuple[str, str], dict]:
    """Index polymer numbering rows by label chain and label residue number."""
    return {
        (row["asym_id"], row["seq_id"]): row
        for row in categories.get("_pdbx_poly_seq_scheme", [])
        if row.get("asym_id") and row.get("seq_id")
    }


def atom_indexes(categories: dict):
    """Collect first-model heavy atoms by label chain and residue number."""
    coordinates: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    residue_names: dict[tuple[str, str], str] = {}
    first_model = None
    for row in categories.get("_atom_site", []):
        if normalize_missing(row.get("group_PDB")).upper() not in {"ATOM", "HETATM"}:
            continue
        model = normalize_missing(row.get("pdbx_PDB_model_num")) or "1"
        if first_model is None:
            first_model = model
        if model != first_model:
            continue
        alt = normalize_missing(row.get("label_alt_id"))
        element = normalize_missing(row.get("type_symbol")).upper()
        if alt not in {"", "A", "1"} or element in {"H", "D"}:
            continue
        chain = normalize_missing(row.get("label_asym_id"))
        seq_id = normalize_missing(row.get("label_seq_id"))
        xyz = tuple(parse_float(row.get(name)) for name in ("Cartn_x", "Cartn_y", "Cartn_z"))
        if not chain or not seq_id or any(value is None for value in xyz):
            continue
        key = chain, seq_id
        coordinates[key].append(xyz)
        residue_names[key] = normalize_missing(
            row.get("label_comp_id") or row.get("auth_comp_id")
        ).upper()
    return dict(coordinates), residue_names


def residue_letter(scheme_row: dict | None, atom_residue_name: str) -> str:
    name = ""
    if scheme_row:
        name = normalize_missing(
            scheme_row.get("mon_id")
            or scheme_row.get("pdb_mon_id")
            or scheme_row.get("auth_mon_id")
        ).upper()
    return AA3_TO_1.get(name or atom_residue_name, "X")


def make_mhc_mapping(
    chain: str,
    sequence: str,
    pseudosequence: str,
    canonical_to_deposited: dict[int, int | None],
    scheme: dict,
    coordinates: dict,
    residue_names: dict,
):
    mappings, atom_lists = [], []
    for pseudo_index, canonical_position in enumerate(MHC_I_PSEUDO_POSITIONS):
        sequence_position = canonical_to_deposited[canonical_position]
        if sequence_position is None:
            raise HlaMappingError("missing_canonical_pseudosequence_positions", canonical_positions=[canonical_position])
        seq_id = str(sequence_position)
        key = chain, seq_id
        row = scheme.get(key)
        atoms = coordinates.get(key, [])
        if atoms:
            validate_coordinate_residue(pseudosequence[pseudo_index], residue_names.get(key, ""), chain, seq_id, canonical_position)
        mappings.append({
            "pseudo_index": pseudo_index,
            "canonical_position": canonical_position,
            "structure_seq_id": seq_id,
            "label_asym_id": chain,
            "label_seq_id": seq_id,
            "auth_seq_id": row.get("auth_seq_num", "") if row else "",
            "auth_asym_id": row.get("pdb_strand_id", "") if row else "",
            "auth_insertion_code": normalize_missing(row.get("pdb_ins_code")) if row else "",
            "residue_name": residue_names.get(key, ""),
            "expected_residue": pseudosequence[pseudo_index],
            "sequence_residue": sequence[sequence_position - 1],
            "structure_residue": residue_letter(row, residue_names.get(key, "")),
            "mapped": bool(atoms),
        })
        atom_lists.append(atoms)
    return mappings, atom_lists


def make_peptide_mapping(
    chain: str,
    sequence: str,
    scheme: dict,
    coordinates: dict,
    residue_names: dict,
):
    mappings, atom_lists = [], []
    for index, expected in enumerate(sequence):
        seq_id = str(index + 1)
        key = chain, seq_id
        row = scheme.get(key)
        atoms = coordinates.get(key, [])
        mappings.append({
            "peptide_index": index,
            "structure_seq_id": seq_id,
            "label_asym_id": chain,
            "label_seq_id": seq_id,
            "auth_seq_id": row.get("auth_seq_num", "") if row else "",
            "auth_asym_id": row.get("pdb_strand_id", "") if row else "",
            "auth_insertion_code": normalize_missing(row.get("pdb_ins_code")) if row else "",
            "residue_name": residue_names.get(key, ""),
            "expected_residue": expected,
            "structure_residue": residue_letter(row, residue_names.get(key, "")),
            "mapped": bool(atoms),
        })
        atom_lists.append(atoms)
    return mappings, atom_lists


def minimum_distance(left_atoms: list, right_atoms: list) -> float | None:
    if not left_atoms or not right_atoms:
        return None
    squared = min(
        (lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2
        for lx, ly, lz in left_atoms
        for rx, ry, rz in right_atoms
    )
    return math.sqrt(squared)


def distance_matrix(left: list[list], right: list[list]) -> list[list[float | None]]:
    return [[minimum_distance(a, b) for b in right] for a in left]


def structural_screen(
    mhc_chain: str,
    mhc_mapping: list[dict],
    pseudo_matrix: list[list],
    peptide_atoms: list[list],
    coordinates: dict,
) -> tuple[dict, dict]:
    """Apply the two fixed structural screens described in Methods."""
    pseudo_keys = {(mhc_chain, row["structure_seq_id"]) for row in mhc_mapping}
    full_mhc_atoms = [
        atoms for (chain, _), atoms in coordinates.items() if chain == mhc_chain
    ]
    other_mhc_atoms = [
        atoms for key, atoms in coordinates.items()
        if key[0] == mhc_chain and key not in pseudo_keys
    ]
    full_matrix = distance_matrix(full_mhc_atoms, peptide_atoms)
    other_matrix = distance_matrix(other_mhc_atoms, peptide_atoms)

    def matrix_min(matrix: list[list]) -> float | None:
        values = [value for row in matrix for value in row if value is not None]
        return min(values) if values else None

    def contacts(matrix: list[list]) -> list[bool]:
        return [
            any(row[index] is not None and row[index] <= CONTACT_CUTOFF for row in matrix)
            for index in range(len(peptide_atoms))
        ]

    full_min = matrix_min(full_matrix)
    pseudo_min = matrix_min(pseudo_matrix)
    other_min = matrix_min(other_matrix)
    mhc_contacts = contacts(full_matrix)
    pseudo_contacts = contacts(pseudo_matrix)
    other_contacts = contacts(other_matrix)
    mhc_fraction = sum(mhc_contacts) / len(mhc_contacts)
    pseudo_fraction = sum(pseudo_contacts) / len(pseudo_contacts)
    other_fraction = sum(other_contacts) / len(other_contacts)
    gap = pseudo_min - full_min if pseudo_min is not None and full_min is not None else None

    unbound = full_min is not None and full_min >= UNBOUND_DISTANCE
    pocket_missed = (
        any(mhc_contacts)
        and any(other_contacts)
        and not unbound
        and mhc_fraction >= MIN_MHC_CONTACT_FRACTION
        and pseudo_min is not None
        and pseudo_min >= PSEUDO_FAR_DISTANCE
        and (
            not any(pseudo_contacts)
            or (
                pseudo_fraction <= MAX_PSEUDO_CONTACT_FRACTION
                and gap is not None
                and gap >= PSEUDO_DISTANCE_GAP
            )
        )
    )
    return (
        {
            "peptide_unbound_to_mhc": bool(unbound),
            "pseudoseq_pocket_missed": bool(pocket_missed),
        },
        {
            "full_mhc_min_distance": full_min,
            "pseudoseq_min_distance": pseudo_min,
            "nonpseudoseq_min_distance": other_min,
            "pseudoseq_distance_gap": gap,
            "mhc_contact_fraction": mhc_fraction,
            "pseudoseq_contact_fraction": pseudo_fraction,
            "nonpseudoseq_contact_fraction": other_fraction,
        },
    )


def build_records(candidate: dict, categories: dict, pseudoseqs: dict,
                  max_pair_min_distance: float = 12.0) -> list[dict]:
    """Expand one entity-level candidate into its deposited chain pairs."""
    sequences = entity_sequences(categories)
    chains = entity_chains(categories)
    scheme = scheme_index(categories)
    coordinates, residue_names = atom_indexes(categories)
    mhc, peptide = candidate["mhc"], candidate["peptide"]
    mhc_sequence = clean_sequence(sequences.get(mhc["entity_id"]) or mhc["sequence"])
    peptide_sequence = clean_sequence(
        sequences.get(peptide["entity_id"]) or peptide["sequence"]
    )

    allele, allele_pseudosequence = lookup_pseudoseq(
        mhc.get("allele_candidates", []), pseudoseqs
    )
    alignment = align_hla_to_reference(mhc_sequence)
    pseudosequence = alignment["pseudosequence"]

    mhc_chains = mhc.get("asym_ids") or chains.get(mhc["entity_id"], [])
    peptide_chains = peptide.get("asym_ids") or chains.get(peptide["entity_id"], [])
    records = []
    for mhc_chain in mhc_chains:
        mhc_mapping, mhc_atoms = make_mhc_mapping(
            mhc_chain, mhc_sequence, pseudosequence, alignment["canonical_to_deposited"],
            scheme, coordinates, residue_names,
        )

        # These mapping checks only reject candidates that cannot produce a
        # meaningful 34-row matrix. Complete 34/34 mapping is applied later
        # when the paper cohort is selected.
        if sum(row["mapped"] for row in mhc_mapping) < 30:
            continue

        for peptide_chain in peptide_chains:
            peptide_mapping, peptide_atoms = make_peptide_mapping(
                peptide_chain, peptide_sequence, scheme, coordinates, residue_names
            )
            if not any(row["mapped"] for row in peptide_mapping):
                continue
            # Store three decimals, as in the released table, and use that
            # same table for the fixed pseudosequence screen.
            matrix = [
                [round(value, 3) if value is not None else None for value in row]
                for row in distance_matrix(mhc_atoms, peptide_atoms)
            ]
            values = [value for row in matrix for value in row if value is not None]
            if not values or (max_pair_min_distance and min(values) > max_pair_min_distance):
                continue
            screen_flags, screen_metrics = structural_screen(
                mhc_chain, mhc_mapping, matrix, peptide_atoms, coordinates
            )
            records.append({
                "pdb_id": candidate["pdb_id"],
                "allele": allele,
                "pseudosequence": pseudosequence,
                "pseudosequence_source": "aligned_structure_sequence",
                "pseudosequence_position_offset": alignment["metadata"]["pseudosequence_position_offset"],
                "hla_reference_alignment": alignment["metadata"],
                "allele_table_pseudosequence": allele_pseudosequence,
                "quality_flags": (
                    (["deposited_pseudosequence_differs_from_named_allele"]
                     if allele_pseudosequence is not None and pseudosequence != allele_pseudosequence else [])
                    + (["incomplete_pseudosequence_coordinates"]
                       if not all(row["mapped"] for row in mhc_mapping) else [])
                ),
                "peptide": peptide_sequence,
                "distance_min_heavy_atom": matrix,
                "mhc": {"entity_id": mhc["entity_id"], "asym_id": mhc_chain},
                "peptide_chain": {
                    "entity_id": peptide["entity_id"], "asym_id": peptide_chain
                },
                "pseudosequence_residue_mappings": mhc_mapping,
                "peptide_residue_mappings": peptide_mapping,
                "screen_flags": screen_flags,
                "screen_metrics": screen_metrics,
            })
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_INPUT, help="Entity discovery JSONL")
    parser.add_argument("--pdb-root", type=Path, default=DEFAULT_PDB_ROOT)
    parser.add_argument("--pseudoseqs", type=Path, default=DEFAULT_PSEUDOSEQ_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors-output", type=Path,
                        help="Rejected candidate reasons; default beside the output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pseudoseqs = load_pseudoseqs(args.pseudoseqs)
    candidates = [record for _, record in read_jsonl(args.candidates)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors_path = args.errors_output or args.output.with_name(args.output.stem + "_errors.jsonl")
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.unlink(missing_ok=True)
    candidate_count = record_count = failed_count = 0

    cached_path, categories = None, None
    with args.output.open("w", encoding="utf-8") as output, errors_path.open("w", encoding="utf-8") as errors:
        for candidate in tqdm(
            candidates,
            total=len(candidates),
            desc="Building distance matrices",
            unit="candidate",
        ):
            candidate_count += 1
            try:
                path = path_for_pdb_id(args.pdb_root, candidate["pdb_id"])
                if path != cached_path:
                    categories = parse_mmcif_categories(path, DETAIL_CATEGORIES)
                    cached_path = path
                records = build_records(candidate, categories, pseudoseqs)
                failed_count += not records
                if not records:
                    errors.write(json.dumps({"pdb_id": candidate.get("pdb_id"), "error": "no_usable_coordinate_mapping"}) + "\n")
                for record in records:
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
                    record_count += 1
            except Exception as error:
                # A handful of archive entries have incomplete polymer tables;
                # they are not usable distance records and are simply counted.
                failed_count += 1
                errors.write(json.dumps({
                    "pdb_id": candidate.get("pdb_id"),
                    "error": getattr(error, "code", type(error).__name__),
                    "details": getattr(error, "details", {"message": str(error)}),
                }) + "\n")
                tqdm.write(f"skipping {candidate.get('pdb_id')}: {error}")

    def file_identity(path):
        return {"path": str(path.resolve()), "sha256": file_sha256(path)}

    source_dir = Path(__file__).parent
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": file_identity(args.candidates),
        "pseudoseqs": file_identity(args.pseudoseqs),
        "reference": file_identity(REFERENCE_PATH),
        "pdb_root": str(args.pdb_root.resolve()),
        "outputs": {"distances": file_identity(args.output), "errors": file_identity(errors_path)},
        "code": {name: file_identity(source_dir / name) for name in (
            "build_pmhc_distance_matrices.py", "hla_residue_mapping.py", "pmhc_mhci_common.py", "io_utils.py")},
        "counts": {"input_candidates": candidate_count, "accepted_chain_pairs": record_count,
                   "rejected_candidates": failed_count},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"read {candidate_count:,} candidates; wrote {record_count:,} chain pairs "
        f"to {args.output}; {failed_count:,} candidates had no usable mapping"
    )


if __name__ == "__main__":
    main()
