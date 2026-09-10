#!/usr/bin/env python3
"""Reproduce the paper's structural attribution, FoldX, and SASA analyses."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from attribution import ResidueAttribution
from io_utils import file_sha256, read_jsonl, source_code_identity
from lamina import AA_TO_ID, load_model
from peptide_sasa import (
    SASA_ALGORITHM_VERSION, BIOPYTHON_VERSION, record_id,
    sasa_code_identity, scientific_record_sha256, structure_path,
)


def read_records(path):
    return [record for _, record in read_jsonl(path)]


def usable_record(record):
    peptide = record.get("peptide", "")
    pseudo = record.get("pseudosequence", "")
    mapping = record.get("pseudosequence_residue_mappings", [])
    flags = record.get("screen_flags")
    if flags is None:
        raise ValueError("Distance records must include structural screen flags")
    return (
        8 <= len(peptide) <= 14 and len(pseudo) == 34
        and all(aa in AA_TO_ID for aa in peptide + pseudo)
        and np.asarray(record.get("distance_min_heavy_atom")).shape == (34, len(peptide))
        and len(mapping) == 34 and all(row.get("mapped") for row in mapping)
        and not flags.get("pseudoseq_pocket_missed")
        and not flags.get("peptide_unbound_to_mhc")
    )


def validate_sasa_cache(path, records, pdb_root):
    """Check every saved result before selecting the analysis cohort."""
    current = {record_id(record): record for record in records}
    source_hashes, seen = {}, set()
    code_identity = sasa_code_identity()
    for result in read_records(path):
        key = result.get("record_id")
        record = current.get(key)
        if record is None or key in seen:
            raise ValueError("Foreign or duplicate SASA cache entry")
        seen.add(key)
        source = structure_path(record, pdb_root)
        if source not in source_hashes:
            source_hashes[source] = file_sha256(source)
        if not all((
            result.get("sasa_algorithm_version") == SASA_ALGORITHM_VERSION,
            result.get("sasa_code") == code_identity,
            result.get("biopython_version") == BIOPYTHON_VERSION,
            result.get("distance_record_sha256") == scientific_record_sha256(record),
            result.get("source_sha256") == source_hashes[source],
            result.get("probe_radius") == 1.4, result.get("n_points") == 200,
            result.get("clip_negative_buried_sasa") is False,
        )):
            raise ValueError(f"Stale SASA result for {key}; rerun src/peptide_sasa.py")


def load_foldx_results(path, structures, records, attributions, pdb_root):
    """Validate a local scan against its structure, coordinate, and result hashes."""
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("records_sha256") != file_sha256(Path(structures)) or manifest.get("results_sha256") != file_sha256(path):
        raise ValueError("FoldX CSV or structure input changed; regenerate the scan for these inputs")
    if "selected_complexes" in manifest:
        from run_foldx_peptide_scan import is_foldx_complex

        expected = sum(is_foldx_complex(record) for record in records)
        if (manifest["selected_complexes"] != expected
                or manifest.get("completed_complexes") != expected
                or manifest.get("failed_complexes") != 0):
            raise ValueError("FoldX scan is incomplete for the selected cohort; resolve scan errors and rerun before analysis")
    records_by_line = {record["line_index"]: record for record in records}
    source_hashes, rows = {}, []
    for row in pd.read_csv(path).itertuples():
        line = int(row.line_index)
        record = records_by_line.get(line)
        if record is None:
            continue
        source = structure_path(record, pdb_root)
        if source not in source_hashes:
            source_hashes[source] = file_sha256(source)
        if manifest.get("source_sha256", {}).get(record["pdb_id"].upper()) != source_hashes[source]:
            raise ValueError("FoldX coordinate input changed or has no recorded hash")
        index = int(row.peptide_index)
        if (str(row.pdb_id).upper() != record["pdb_id"].upper() or row.peptide != record["peptide"]
                or not 0 <= index < len(record["peptide"]) or row.wild_type != record["peptide"][index]):
            raise ValueError("FoldX residue identity does not match corrected structure")
        wt, energy = float(row.wt_interaction_energy), float(row.mutant_interaction_energy)
        wt_interface, interface = float(row.wt_interface_residues), float(row.mutant_interface_residues)
        if not all(np.isfinite(value) for value in (wt, energy, wt_interface, interface)):
            continue
        if abs(wt) <= 1e-8 or wt > -5 or wt_interface < 10 or abs(energy) <= 1e-8 or interface < 1:
            continue
        rows.append({"line_index": line, "pdb_id": record["pdb_id"], "peptide": record["peptide"],
                     "peptide_index": index, "contribution": float(attributions[line][index]),
                     "foldx_delta": energy - wt, "wt_interaction_energy": wt,
                     "wt_interface_residues": wt_interface, "mutant_interface_residues": interface})
    return pd.DataFrame(rows), manifest


def sasa_table(records, attribution_by_line, sasa_path, pdb_root):
    # run_analysis validates every saved record before selecting the cohort.
    cache = {item["record_id"]: item for item in read_records(sasa_path)}
    rows, missing = [], []
    for record_index, record in enumerate(records):
        result = cache.get(record_id(record))
        if result is None:
            missing.append(record_id(record))
            continue
        raw = attribution_by_line[record["line_index"]]
        scale = float(np.std(raw))
        z = (raw - float(np.mean(raw))) / scale if scale > 0 else np.zeros_like(raw)
        for residue in result["residues"]:
            index = int(residue["peptide_index"])
            if not 0 <= index < len(raw):
                raise ValueError("SASA peptide index is out of range")
            relative = residue.get("bound_relative_sasa")
            relative = float(np.clip(relative, 0, 1)) if relative is not None and np.isfinite(relative) else np.nan
            rows.append({**residue, "line_index": record["line_index"], "record_index": record_index,
                         "pdb_id": record["pdb_id"], "peptide": record["peptide"],
                         "buried_sasa": max(0, float(residue["buried_sasa"])),
                         "bound_relative_sasa": relative,
                         "contribution_raw": float(raw[index]), "contribution": float(z[index]),
                         "positive": bool(relative < 0.5) if np.isfinite(relative) else None})
    return pd.DataFrame(rows), missing


def metric_summary(foldx, sasa):
    raw = sasa.replace([np.inf, -np.inf], np.nan).dropna(subset=["contribution", "buried_sasa"])
    normalized = raw.dropna(subset=["bound_relative_sasa", "positive"])
    per_auc, per_rho, foldx_rho = [], [], []
    if len(foldx):
        for _, group in foldx.groupby("line_index"):
            if len(group) >= 3 and group.contribution.nunique() > 1 and group.foldx_delta.nunique() > 1:
                foldx_rho.append(float(spearmanr(group.contribution, group.foldx_delta).statistic))
    for _, group in raw.groupby("line_index"):
        if len(group) >= 3 and group.contribution.nunique() > 1 and group.buried_sasa.nunique() > 1:
            per_rho.append(float(spearmanr(group.contribution, group.buried_sasa).statistic))
    for _, group in normalized.groupby("line_index"):
        if group.positive.nunique() == 2:
            per_auc.append(float(roc_auc_score(group.positive.astype(bool), group.contribution)))
    summary = {
        "foldx_residues": len(foldx), "foldx_complexes": int(foldx.line_index.nunique()) if len(foldx) else 0,
        "foldx_spearman": float(spearmanr(foldx.contribution, foldx.foldx_delta).statistic) if len(foldx) > 1 else None,
        "foldx_mean_individual_spearman": float(np.mean(foldx_rho)) if foldx_rho else None,
        "foldx_median_individual_spearman": float(np.median(foldx_rho)) if foldx_rho else None,
        "sasa_residues": len(raw), "sasa_complexes": int(raw.line_index.nunique()),
        "buried_sasa_pooled_spearman": float(spearmanr(raw.contribution, raw.buried_sasa).statistic),
        "buried_sasa_mean_individual_spearman": float(np.mean(per_rho)) if per_rho else None,
        "buried_sasa_median_individual_spearman": float(np.median(per_rho)) if per_rho else None,
        "normalized_sasa_residues": len(normalized), "normalization_excluded_residues": len(raw) - len(normalized),
        "pooled_auc": float(roc_auc_score(normalized.positive.astype(bool), normalized.contribution)) if normalized.positive.nunique() == 2 else None,
        "individual_auc_complexes": len(per_auc),
        "mean_individual_auc": float(np.mean(per_auc)) if per_auc else None,
        "median_individual_auc": float(np.median(per_auc)) if per_auc else None,
        "positive_definition": "bound relative SASA < 0.5; unsupported normalization excluded",
        "contribution_normalization": "raw for FoldX; within-complex z-score for SASA",
    }
    return summary, raw, normalized, per_auc


def plot_results(foldx, sasa):
    """Return four analysis figures for notebook display without saving images."""
    import matplotlib.pyplot as plt
    summary, raw, normalized, per_auc = metric_summary(foldx, sasa)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})

    def scatter(ax, x, y):
        x, y = np.asarray(x), np.asarray(y)
        hist, xe, ye = np.histogram2d(x, y, bins=80)
        xi = np.clip(np.digitize(x, xe) - 1, 0, 79)
        yi = np.clip(np.digitize(y, ye) - 1, 0, 79)
        density = np.log1p(hist[xi, yi])
        order = np.argsort(density)
        ax.scatter(x[order], y[order], c=density[order], cmap="magma", s=2.5, alpha=.9, linewidths=0, rasterized=True)

    def panel(ax, index):
        if index == 0 and len(foldx):
            scatter(ax, foldx.foldx_delta, foldx.contribution)
            ax.set(xlabel="Mutation Scan ΔE (kcal/mol)", ylabel="Model Residue Contribution")
            ax.text(.04, .94, rf"$\rho={summary['foldx_spearman']:.3f}$", transform=ax.transAxes, va="top")
        elif index == 1:
            scatter(ax, raw.buried_sasa, raw.contribution)
            ax.set(xlabel=r"Buried SASA ($\AA^2$)", ylabel="Model Residue Contribution")
            ax.text(.04, .94, rf"$\rho={summary['buried_sasa_pooled_spearman']:.3f}$", transform=ax.transAxes, va="top")
        elif index == 2 and summary["pooled_auc"] is not None:
            fpr, tpr, _ = roc_curve(normalized.positive.astype(bool), normalized.contribution)
            ax.plot(fpr, tpr, color="#4c78a8", lw=2, label=f"AUROC={summary['pooled_auc']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color=".65", lw=1)
            ax.set(xlabel="FPR", ylabel="TPR", xlim=(0, 1), ylim=(0, 1))
            ax.legend(frameon=False, loc="lower right")
        elif index == 3:
            ax.hist(per_auc, bins=np.linspace(0, 1, 16), color="#9680b1", edgecolor="white")
            if per_auc:
                ax.axvline(np.mean(per_auc), color=".25", ls="--", label=f"Mean AUROC={np.mean(per_auc):.3f}")
            ax.set(xlabel="Per-peptide Buried Residue Prediction AUROC", ylabel="Count", xlim=(0, 1))
            ax.legend(frameon=False, loc="upper left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)

    figures = {}
    names = ["peptide_foldx_single", "sasa_buried_scatter", "sasa_pooled_roc", "sasa_individual_auc"]
    for index, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        panel(ax, index)
        fig.tight_layout(pad=1.1)
        figures[name] = fig
        # Explicit notebook display controls where each panel appears.
        plt.close(fig)
    return figures


def run_analysis(structures, checkpoint, sasa_path, pdb_root, output, foldx_results, make_plots=True):
    structures, checkpoint, sasa_path, output = map(Path, (structures, checkpoint, sasa_path, output))
    output.mkdir(parents=True, exist_ok=True)
    all_records = read_records(structures)
    validate_sasa_cache(sasa_path, all_records, pdb_root)
    records = []
    for line_index, record in enumerate(all_records):
        if usable_record(record):
            records.append({**record, "line_index": line_index})
    attribution = ResidueAttribution(load_model(checkpoint))
    max_completeness_error = 0.0
    attributions, pair_maps, rows = {}, {}, []
    for record_index, record in enumerate(records):
        line = record["line_index"]
        pair, contribution, error = attribution(record["pseudosequence"], record["peptide"])
        max_completeness_error = max(max_completeness_error, error)
        attributions[line], pair_maps[f"line_{line}"] = contribution, pair
        matrix = np.asarray(record["distance_min_heavy_atom"], dtype=float)
        for index, (residue, value) in enumerate(zip(record["peptide"], contribution)):
            distances = matrix[:, index]
            distance = float(np.min(distances[np.isfinite(distances)])) if np.isfinite(distances).any() else np.nan
            rows.append({"record_index": record_index, "line_index": line, "pdb_id": record["pdb_id"],
                         "peptide": record["peptide"], "peptide_index": index, "peptide_position": index + 1,
                         "residue": residue, "contribution": float(value), "min_distance": distance,
                         "contact": bool(distance <= 4) if np.isfinite(distance) else None})
        if (record_index + 1) % 200 == 0:
            print(f"Exact attributions: {record_index + 1}/{len(records)}", flush=True)
    peptide_df = pd.DataFrame(rows)
    peptide_df.to_csv(output / "peptide_distance_attribution.csv", index=False)
    np.savez_compressed(output / "residue_pair_attributions.npz", **pair_maps)
    foldx_df, foldx_provenance = load_foldx_results(foldx_results, structures, records, attributions, pdb_root)
    foldx_df.to_csv(output / "peptide_foldx_single.csv", index=False)
    sasa_df, missing = sasa_table(records, attributions, sasa_path, pdb_root)
    if missing:
        raise ValueError(f"SASA is missing {len(missing)} selected complexes; rerun src/peptide_sasa.py and inspect its errors")
    if sasa_df.empty:
        raise ValueError("No validated SASA records are available")
    sasa_df.to_csv(output / "peptide_sasa_attribution.csv", index=False)
    summary = metric_summary(foldx_df, sasa_df)[0]
    figures = plot_results(foldx_df, sasa_df) if make_plots else {}
    summary.update({"retained_complexes": len(records), "input_records": len(all_records),
                    "sasa_missing_complexes": missing, "max_attribution_completeness_error": max_completeness_error})
    (output / "structural_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    outputs = {path.name: file_sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "run_manifest.json"}
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": file_sha256(Path(__file__)),
                "code": source_code_identity(Path(__file__).with_name(name) for name in (
                    "structural_analysis.py", "attribution.py", "lamina.py", "io_utils.py",
                    "peptide_sasa.py", "pmhc_mhci_common.py", "run_foldx_peptide_scan.py")),
                "inputs": {str(path.resolve()): file_sha256(path) for path in (structures, checkpoint, sasa_path)},
                "sasa_algorithm_version": SASA_ALGORITHM_VERSION, "foldx": foldx_provenance,
                "summary": summary, "outputs_sha256": outputs}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return {"records": records, "record_by_line": {r["line_index"]: r for r in records},
            "attribution_by_line": attributions, "peptide_df": peptide_df,
            "foldx_df": foldx_df, "sasa_df": sasa_df, "summary": summary,
            "figures": figures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, default=Path("artifacts/pmhc_mhci_distance_matrices.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sasa", type=Path, default=Path("artifacts/peptide_sasa.jsonl"))
    parser.add_argument("--foldx-results", type=Path, required=True, help="Local FoldX CSV with its .manifest.json")
    parser.add_argument("--pdb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/interpretability"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run_analysis(args.structures, args.checkpoint, args.sasa, args.pdb_root,
                 args.output, args.foldx_results, make_plots=False)


if __name__ == "__main__":
    main()
