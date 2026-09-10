#!/usr/bin/env python3
"""Format the BA and EL rows used to train the two paper models.

Run this after extracting ``NetMHCpan_train.tar.gz`` under
``Data/NetMHCpan``. The output is one gzip-compressed JSON object per row.
Only the binding-affinity (BA) and eluted-ligand (EL) training folds are
included because no other NetMHCpan datasets are used in the paper models.
"""

import argparse
import gzip
import io
import json
import math
from pathlib import Path

from tqdm import tqdm

if __package__:
    from .io_utils import file_sha256
else:
    from io_utils import file_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "Data" / "NetMHCpan"


def read_lookup(path):
    """Read a two-column NetMHCpan lookup table."""
    lookup = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            key, value = line.split(maxsplit=1)
            lookup[key] = value.strip()
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    train_dir = args.data_dir / "NetMHCpan_train"
    output = args.output or args.data_dir / "netmhcpan_training.jsonl.gz"

    # ``pseudoseqs`` maps each allele to the 34 groove residues seen by the
    # model. ``allelelist`` also resolves multi-allelic mass-spectrometry
    # samples to all HLA alleles that could have presented the peptide.
    allele_to_pseudosequence = read_lookup(train_dir / "pseudoseqs")
    sample_to_alleles = read_lookup(train_dir / "allelelist")

    fold_files = [
        (dataset, train_dir / f"c{fold:03d}_{dataset}")
        for dataset in ("ba", "el")
        for fold in range(5)
    ]
    missing = [str(path) for _, path in fold_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing NetMHCpan training folds: {', '.join(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = {"ba": 0, "el": 0}
    skipped = {"malformed": 0, "invalid_target": 0, "missing_pseudosequence": 0}
    temporary = output.with_name(output.name + ".tmp")
    # Omit gzip timestamps and filenames so identical folds produce identical
    # compressed data, independently of the download directory or run date.
    with temporary.open("wb") as raw, gzip.GzipFile(
        fileobj=raw, mode="wb", filename="", mtime=0
    ) as compressed, io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as destination:
        for dataset, path in fold_files:
            with path.open() as source:
                for line in tqdm(source, desc=path.name, unit="row"):
                    fields = line.split()
                    if len(fields) != 4:
                        skipped["malformed"] += 1
                        continue
                    peptide, target_text, allele_or_sample, _context = fields
                    try:
                        target = float(target_text)
                    except ValueError:
                        skipped["invalid_target"] += 1
                        continue
                    if not math.isfinite(target):
                        skipped["invalid_target"] += 1
                        continue

                    # BA rows name one allele directly. EL rows can name a
                    # multi-allelic sample, whose comma-separated allele list
                    # comes from ``allelelist``.
                    allele_text = sample_to_alleles.get(
                        allele_or_sample, allele_or_sample
                    )
                    pseudosequences = [
                        allele_to_pseudosequence[allele]
                        for allele in allele_text.split(",")
                        if allele in allele_to_pseudosequence
                    ]
                    if not pseudosequences:
                        skipped["missing_pseudosequence"] += 1
                        continue

                    record = {
                        "dataset": dataset,
                        "peptide": peptide,
                        "target": target,
                        "pseudosequences": pseudosequences,
                    }
                    destination.write(
                        json.dumps(record, separators=(",", ":")) + "\n"
                    )
                    written[dataset] += 1

    temporary.replace(output)
    manifest = {
        "rows": written,
        "skipped": skipped,
        "inputs_sha256": {
            path.name: file_sha256(path)
            for path in [train_dir / "pseudoseqs", train_dir / "allelelist"]
            + [path for _, path in fold_files]
        },
        "output": output.name,
        "output_sha256": file_sha256(output),
    }
    manifest_path = output.with_name(output.name.removesuffix(".jsonl.gz") + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"Wrote {sum(written.values()):,} rows to {output} "
        f"(BA={written['ba']:,}, EL={written['el']:,})"
    )
    print(f"Skipped rows: {skipped}; provenance: {manifest_path}")


if __name__ == "__main__":
    main()
