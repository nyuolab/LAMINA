#!/usr/bin/env python3
"""Run the peptide single-mutant FoldX scan used in the paper.

For each usable pMHC complex this script writes a temporary two-chain PDB,
runs RepairPDB, measures the wild-type interface, builds one mutant per
peptide position (alanine, or glycine for wild-type alanine), and measures
every mutant interface. A flat CSV and its input/coordinate hash manifest
provide the persistent outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

from io_utils import file_sha256, read_jsonl
from pmhc_mhci_common import (
    AA3_TO_1,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_PDB_ROOT,
    STANDARD_AA,
    normalize_missing,
    parse_float,
    parse_mmcif_categories,
    path_for_pdb_id,
)


DEFAULT_INPUT = DEFAULT_ARTIFACT_DIR / "pmhc_mhci_distance_matrices.jsonl"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "foldx_peptide_scan.csv"
DEFAULT_FOLDX_BIN = DEFAULT_DATA_DIR / "FoldX" / "foldx"
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

OUTPUT_COLUMNS = [
    "line_index", "pdb_id", "allele", "peptide", "peptide_index",
    "peptide_position", "wild_type", "mutant", "mutation",
    "wt_interaction_energy", "wt_interface_residues",
    "mutant_interaction_energy", "mutant_interface_residues",
    "delta_interaction_energy",
]


def is_foldx_complex(record: dict) -> bool:
    """Select screened complexes whose whole peptide has coordinates."""
    pseudo_mapping = record.get("pseudosequence_residue_mappings", [])
    peptide_mapping = record.get("peptide_residue_mappings", [])
    screen = record.get("screen_flags", {})
    sequence = record.get("pseudosequence", "") + record.get("peptide", "")
    return (
        len(pseudo_mapping) == 34
        and all(row.get("mapped") for row in pseudo_mapping)
        and len(peptide_mapping) == len(record.get("peptide", ""))
        and 8 <= len(record.get("peptide", "")) <= 14
        and len(record.get("pseudosequence", "")) == 34
        and all(row.get("mapped") for row in peptide_mapping)
        and all(row.get("structure_residue") in STANDARD_AA for row in peptide_mapping)
        and all(residue in STANDARD_AA for residue in sequence)
        and not screen.get("peptide_unbound_to_mhc")
        and not screen.get("pseudoseq_pocket_missed")
    )


def mapping_chain(mapping: dict) -> str:
    return normalize_missing(mapping.get("auth_asym_id") or mapping.get("asym_id"))


def mapping_residue_number(mapping: dict) -> str:
    return normalize_missing(mapping.get("auth_seq_id") or mapping.get("structure_seq_id"))


def mapping_residue(mapping: dict) -> str:
    return normalize_missing(mapping.get("structure_residue") or mapping.get("expected_residue")).upper()


def first_mapped_chain(mappings: list[dict], label: str) -> str:
    chains = {mapping_chain(row) for row in mappings if row.get("mapped") and mapping_chain(row)}
    if len(chains) != 1:
        raise ValueError(f"Expected one mapped {label} chain, found {sorted(chains)}")
    return next(iter(chains))


def output_chain_map(mhc_auth_chain: str, peptide_auth_chain: str) -> dict[str, str]:
    """FoldX/PDB chain IDs are exactly one character."""
    result, used = {}, set()
    for raw_chain in (mhc_auth_chain, peptide_auth_chain):
        if len(raw_chain) == 1 and raw_chain not in used:
            result[raw_chain] = raw_chain
            used.add(raw_chain)
            continue
        result[raw_chain] = next(chain for chain in PDB_CHAIN_IDS if chain not in used)
        used.add(result[raw_chain])
    return result


def pdb_atom_name(atom_name: str, element: str) -> str:
    atom_name = atom_name[:4]
    if len(atom_name) == 4:
        return atom_name
    return f" {atom_name:<3}" if len(element) == 1 and atom_name[:1].isalpha() else f"{atom_name:>4}"


def write_foldx_pdb(record: dict, mmcif_path: Path, output_path: Path) -> dict:
    """Write only the HLA alpha and peptide chains, with heavy atoms."""
    mhc_label = normalize_missing((record.get("mhc") or {}).get("asym_id"))
    peptide_label = normalize_missing((record.get("peptide_chain") or {}).get("asym_id"))
    mhc_auth = first_mapped_chain(record.get("pseudosequence_residue_mappings", []), "MHC")
    peptide_auth = first_mapped_chain(record.get("peptide_residue_mappings", []), "peptide")
    if not mhc_label or not peptide_label:
        raise ValueError("Record is missing label chain IDs")
    chain_map = output_chain_map(mhc_auth, peptide_auth)
    label_to_output = {mhc_label: chain_map[mhc_auth], peptide_label: chain_map[peptide_auth]}
    atoms = parse_mmcif_categories(mmcif_path, {"_atom_site"}).get("_atom_site", [])
    if not atoms:
        raise ValueError(f"No atoms in {mmcif_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_model, previous_chain, serial, written = None, None, 1, 0
    with output_path.open("w", encoding="utf-8") as output:
        for row in atoms:
            if normalize_missing(row.get("group_PDB")).upper() != "ATOM":
                continue
            model = normalize_missing(row.get("pdbx_PDB_model_num")) or "1"
            if first_model is None:
                first_model = model
            if model != first_model:
                continue
            alt = normalize_missing(row.get("label_alt_id") or row.get("auth_alt_id"))
            element = normalize_missing(row.get("type_symbol")).upper()
            label_chain = normalize_missing(row.get("label_asym_id"))
            if alt not in {"", "A", "1"} or element in {"H", "D"} or label_chain not in label_to_output:
                continue
            resname = normalize_missing(row.get("auth_comp_id") or row.get("label_comp_id")).upper()
            if resname not in AA3_TO_1:
                continue
            resname = "MET" if resname == "MSE" else resname
            residue_number = normalize_missing(row.get("auth_seq_id") or row.get("label_seq_id"))
            if not re.fullmatch(r"-?\d+", residue_number):
                continue
            xyz = [parse_float(row.get(field)) for field in ("Cartn_x", "Cartn_y", "Cartn_z")]
            atom_name = normalize_missing(row.get("auth_atom_id") or row.get("label_atom_id"))
            if any(value is None for value in xyz) or not atom_name:
                continue
            chain = label_to_output[label_chain]
            if previous_chain is not None and chain != previous_chain:
                output.write("TER\n")
            previous_chain = chain
            insertion = normalize_missing(row.get("pdbx_PDB_ins_code"))[:1] or " "
            occupancy = parse_float(row.get("occupancy")) or 1.0
            bfactor = parse_float(row.get("B_iso_or_equiv")) or 0.0
            output.write(
                f"ATOM  {serial:5d} {pdb_atom_name(atom_name, element)} {resname:>3} "
                f"{chain}{int(residue_number):4d}{insertion}   "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occupancy:6.2f}{bfactor:6.2f}"
                f"          {element:>2}\n"
            )
            serial += 1
            written += 1
        output.write("TER\nEND\n")
    if not written:
        raise ValueError("No MHC/peptide atoms were written")
    return {
        "mhc_label_asym_id": mhc_label,
        "peptide_label_asym_id": peptide_label,
        "mhc_auth_chain": mhc_auth,
        "peptide_auth_chain": peptide_auth,
        "mhc_foldx_chain": chain_map[mhc_auth],
        "peptide_foldx_chain": chain_map[peptide_auth],
        "chain_map": chain_map,
        "written_atoms": written,
    }



def foldx_command(binary: Path, command: str, **arguments: object) -> list[str]:
    command_line = [str(binary), f"--command={command}",
                    f"--rotabaseLocation={binary.parent / 'rotabase.txt'}"]
    command_line += [
        f"--{name.replace('_', '-')}={value}"
        for name, value in arguments.items()
    ]
    return command_line


def run_foldx(binary: Path, work: Path, command: str, **arguments: object) -> None:
    result = subprocess.run(
        foldx_command(binary, command, **arguments),
        cwd=work,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError((result.stdout or "") + (result.stderr or ""))


def parse_interaction(output_dir: Path) -> tuple[float, float]:
    """Read the two columns used from FoldX's Interaction_*.fxout table."""
    path = next(output_dir.glob("Interaction_*.fxout"))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Pdb\t"):
            header = line.split("\t")
            values = lines[index + 1].split("\t")
            return (
                float(values[header.index("Interaction Energy")]),
                float(values[header.index("Interface Residues")]),
            )
    raise ValueError(f"No interaction table in {path}")


def analyse_complex(
    binary: Path,
    work: Path,
    pdb_name: str,
    pdb_dir: str,
    output_dir: str,
    chains: str,
) -> tuple[float, float]:
    (work / output_dir).mkdir(parents=True, exist_ok=True)
    run_foldx(
        binary,
        work,
        "AnalyseComplex",
        pdb=pdb_name,
        pdb_dir=pdb_dir,
        analyseComplexChains=chains,
        output_dir=output_dir,
    )
    return parse_interaction(work / output_dir)


def natural_key(path: Path):
    return [int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", path.name)]


def scan_complex(
    line_index: int,
    record: dict,
    pdb_root: Path,
    foldx_binary: Path,
) -> list[dict]:
    """Run all four FoldX stages and return one row per peptide mutant."""
    with tempfile.TemporaryDirectory(prefix=f"foldx_{record['pdb_id']}_") as directory:
        work = Path(directory)
        mmcif_path = path_for_pdb_id(pdb_root, record["pdb_id"])
        pdb_info = write_foldx_pdb(record, mmcif_path, work / "input.pdb")
        chains = f"{pdb_info['mhc_foldx_chain']},{pdb_info['peptide_foldx_chain']}"

        # 1) Repair the deposited two-chain complex.
        run_foldx(
            foldx_binary, work, "RepairPDB",
            pdb="input.pdb", pdb_dir=".", output_dir=".",
        )

        # 2) Measure the repaired wild-type interface.
        wt_energy, wt_interface = analyse_complex(
            foldx_binary, work, "input_Repair.pdb", ".", "analysis_wt", chains
        )

        # 3) Build one mutant for every peptide position. Alanine itself is
        # changed to glycine so every row represents a real perturbation.
        mutations = []
        for mapping in sorted(record["peptide_residue_mappings"], key=lambda row: row["peptide_index"]):
            index = int(mapping["peptide_index"])
            wild_type = record["peptide"][index]
            if mapping_residue(mapping) != wild_type:
                raise ValueError(f"Peptide coordinate identity differs at position {index + 1}")
            mutant = "G" if wild_type == "A" else "A"
            chain = pdb_info["chain_map"][mapping_chain(mapping)]
            number = mapping_residue_number(mapping)
            if not re.fullmatch(r"-?\d+", number):
                raise ValueError(f"FoldX requires integer residue numbering: {number}")
            mutations.append((index, wild_type, mutant, f"{wild_type}{chain}{number}{mutant};"))
        (work / "individual_list.txt").write_text(
            "".join(f"{token}\n" for _, _, _, token in mutations),
            encoding="utf-8",
        )
        (work / "mutants").mkdir()
        run_foldx(
            foldx_binary, work, "BuildModel",
            pdb="input_Repair.pdb", pdb_dir=".",
            mutant_file="individual_list.txt", numberOfRuns=1,
            output_dir="mutants",
        )
        mutant_pdbs = sorted(
            (
                path for path in (work / "mutants").glob("*.pdb")
                if not path.name.startswith("WT_")
            ),
            key=natural_key,
        )
        if len(mutant_pdbs) != len(mutations):
            raise RuntimeError(
                f"FoldX made {len(mutant_pdbs)} structures for {len(mutations)} mutations"
            )

        # 4) Measure each mutant interface and subtract the wild-type energy.
        rows = []
        for (index, wild_type, mutant, token), mutant_pdb in zip(mutations, mutant_pdbs):
            energy, interface = analyse_complex(
                foldx_binary,
                work,
                mutant_pdb.name,
                "mutants",
                f"analysis_mutant_{index}",
                chains,
            )
            rows.append({
                "line_index": line_index,
                "pdb_id": record["pdb_id"],
                "allele": record.get("allele"),
                "peptide": record["peptide"],
                "peptide_index": index,
                "peptide_position": index + 1,
                "wild_type": wild_type,
                "mutant": mutant,
                "mutation": token.removesuffix(";"),
                "wt_interaction_energy": wt_energy,
                "wt_interface_residues": wt_interface,
                "mutant_interaction_energy": energy,
                "mutant_interface_residues": interface,
                "delta_interaction_energy": energy - wt_energy,
            })
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pdb-root", type=Path, default=DEFAULT_PDB_ROOT)
    parser.add_argument("--foldx-bin", type=Path, default=DEFAULT_FOLDX_BIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.foldx_bin = args.foldx_bin.expanduser().resolve()
    rotabase = args.foldx_bin.parent / "rotabase.txt"
    if not args.foldx_bin.is_file():
        raise FileNotFoundError(f"Install the licensed FoldX executable: {args.foldx_bin}")
    if not rotabase.is_file():
        raise FileNotFoundError(f"Keep FoldX rotabase.txt beside its executable: {rotabase}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    complexes = mutants = errors = 0
    records_sha256 = file_sha256(args.records)
    source_sha256 = {}
    manifest_path = args.output.with_suffix(".manifest.json")
    # A failed/interrupted fresh scan must not retain an earlier manifest.
    manifest_path.unlink(missing_ok=True)

    # Count selected complexes for the progress bar.
    total = sum(
        is_foldx_complex(record) for _, record in read_jsonl(args.records)
    )
    paper_records = (
        (line_index, record)
        for line_index, record in read_jsonl(args.records)
        if is_foldx_complex(record)
    )

    # Each run overwrites the CSV; its manifest records the completed results.
    with args.output.open("w", encoding="utf-8", newline="") as handle, tqdm(
        paper_records, total=total, desc="FoldX scan", unit="complex"
    ) as progress:
        writer = csv.DictWriter(handle, OUTPUT_COLUMNS)
        writer.writeheader()
        for line_index, record in progress:
            try:
                source_path = path_for_pdb_id(args.pdb_root, record["pdb_id"])
                source_hash = file_sha256(source_path)
                pdb_id = str(record["pdb_id"]).upper()
                if pdb_id in source_sha256 and source_sha256[pdb_id] != source_hash:
                    raise ValueError(f"Coordinate file changed during this run: {source_path}")
                rows = scan_complex(
                    line_index, record, args.pdb_root, args.foldx_bin.resolve()
                )
                if file_sha256(source_path) != source_hash:
                    raise ValueError(f"Coordinate file changed while scanning: {source_path}")
            except Exception as error:
                errors += 1
                tqdm.write(
                    f"FoldX failed for line {line_index} ({record['pdb_id']}): {error}"
                )
                progress.set_postfix(mutants=f"{mutants:,}", errors=errors)
                continue
            writer.writerows(rows)
            source_sha256[pdb_id] = source_hash
            complexes += 1
            mutants += len(rows)
            progress.set_postfix(mutants=f"{mutants:,}", errors=errors)

    if file_sha256(args.records) != records_sha256:
        raise ValueError("Distance-record input changed during the FoldX run; no manifest written")
    manifest = {
        "schema_version": 1,
        "script_sha256": file_sha256(Path(__file__)),
        "foldx_binary_sha256": file_sha256(args.foldx_bin),
        "rotabase_sha256": file_sha256(rotabase),
        "records_sha256": records_sha256,
        "results_sha256": file_sha256(args.output),
        "source_sha256": source_sha256,
        "records_path": str(args.records.resolve()),
        "results_path": str(args.output.resolve()),
        "selected_complexes": total,
        "completed_complexes": complexes,
        "failed_complexes": errors,
        "mutants": mutants,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {mutants:,} mutant energies from {complexes:,} complexes "
        f"to {args.output}; {errors:,} complexes failed; manifest: {manifest_path}"
    )
    if errors:
        raise SystemExit("FoldX scan is incomplete; resolve the reported failures and rerun before analysis")


if __name__ == "__main__":
    main()
