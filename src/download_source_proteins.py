#!/usr/bin/env python3
"""Attach current UniProt source sequences to the Pearson eluted-ligand cohort.

The input is ``pearson_el_current.csv`` from ``download_iedb_eval_data.py``.
The only output is ``pearson_el_with_source.csv.gz``. Each row contains its
UniProt accession, the full source-protein sequence, and a status indicating
whether the reported peptide occurs verbatim in that sequence.
"""

import argparse
import csv
import gzip
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data" / "NetMHCpan"
DEFAULT_INPUT = DATA_DIR / "pearson_el_current.csv"
DEFAULT_OUTPUT = DATA_DIR / "pearson_el_with_source.csv.gz"
UNIPROT = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
BATCH_SIZE = 80


def uniprot_accession(source_iri):
    """Return ``P12345`` from the IEDB value ``UNIPROT:P12345``."""
    prefix, separator, accession = source_iri.strip().partition(":")
    if separator and prefix.upper() == "UNIPROT":
        return accession.strip()
    return ""


def parse_fasta(text):
    """Parse UniProt FASTA text into ``accession -> sequence``."""
    records = {}
    accession = None
    sequence = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(">"):
            if accession:
                records[accession] = "".join(sequence).upper()
            token = line[1:].split()[0]
            fields = token.split("|")
            accession = fields[1] if len(fields) >= 3 else token
            sequence = []
        elif line:
            sequence.append(line)
    if accession:
        records[accession] = "".join(sequence).upper()
    return records


def download_sequences(accessions):
    """Download the requested accessions in small UniProt stream queries."""
    records = {}
    batches = range(0, len(accessions), BATCH_SIZE)
    for start in tqdm(batches, desc="UniProt batches", unit="batch"):
        batch = accessions[start : start + BATCH_SIZE]
        query = "(" + " OR ".join(
            f"accession:{accession}" for accession in batch
        ) + ")"
        url = UNIPROT + "?" + urllib.parse.urlencode(
            {"format": "fasta", "query": query}
        )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "lamina-evaluation/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            records.update(parse_fasta(response.read().decode("utf-8")))

    # Stream searches may omit isoforms or return the primary identifier for
    # a redirected accession. Resolve missing identifiers individually,
    # retaining the identifier reported by IEDB.
    missing = [accession for accession in accessions if accession not in records]
    for accession in tqdm(missing, desc="UniProt individual entries", unit="protein"):
        url = UNIPROT_ENTRY.format(accession=urllib.parse.quote(accession, safe=""))
        request = urllib.request.Request(
            url, headers={"User-Agent": "lamina-evaluation/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                resolved = parse_fasta(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise
        if accession in resolved:
            records[accession] = resolved[accession]
        elif len(resolved) == 1:
            records[accession] = next(iter(resolved.values()))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        input_columns = reader.fieldnames or []
        rows = list(reader)

    # IEDB can also name non-UniProt antigens. FRANK requires a full source
    # protein, so only direct UniProt accessions are sent to UniProt.
    accessions = sorted({
        accession
        for row in rows
        if (
            accession := uniprot_accession(
                row.get("parent_source_antigen_iri", "")
            )
        )
    })
    sequences = download_sequences(accessions)

    output_columns = [
        *input_columns,
        "accession",
        "status",
        "source_sequence",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    status_counts = {}
    with gzip.open(args.output, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        for row in tqdm(rows, desc=f"Writing {args.output.name}", unit="row"):
            accession = uniprot_accession(
                row.get("parent_source_antigen_iri", "")
            )
            sequence = sequences.get(accession, "")
            peptide = row.get("linear_sequence", "").strip().upper()

            # Only rows whose reported ligand is actually present in the
            # downloaded source protein enter the FRANK evaluation.
            if not accession:
                status = "missing_accession"
            elif not sequence:
                status = "missing_source_sequence"
            elif peptide in sequence:
                status = "peptide_found_in_sequence"
            else:
                status = "peptide_not_found_in_sequence"

            status_counts[status] = status_counts.get(status, 0) + 1
            writer.writerow({
                **row,
                "accession": accession,
                "status": status,
                "source_sequence": sequence,
            })

    print(f"Wrote {len(rows):,} rows to {args.output}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count:,}")


if __name__ == "__main__":
    main()
