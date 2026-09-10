#!/usr/bin/env python3
"""Download recent class-I binding assays and Pearson eluted ligands from IEDB.

Writes ``iedb_recent_ba.csv`` for 2021--2026 binding measurements and
``pearson_el_current.csv`` for Pearson et al. ligands (PMID 27841757).
Queries use the assay ID as an explicit sort key and advance through
10,000-row pages. These are current database records at download time.
"""

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "NetMHCpan"
API = "https://query-api.iedb.org/api/v1"
PAGE_SIZE = 10_000
YEARS = range(2021, 2027)

# The output keeps only columns used to construct the evaluation cohorts.
# ``assay_names`` is queried for the local IC50/dissociation-constant filter.
BA_COLUMNS = (
    "elution_id",
    "linear_sequence",
    "mhc_allele_name",
    "assay_names",
    "quantitative_measure",
)
PEARSON_COLUMNS = (
    "elution_id",
    "linear_sequence",
    "mhc_allele_name",
    "parent_source_antigen_iri",
)


def fetch_pages(endpoint, columns, filters, order_column):
    """Return all rows for one ordered PostgREST query."""
    rows = []
    offset = 0
    with tqdm(desc=f"IEDB {endpoint}", unit="row") as progress:
        while True:
            parameters = [
                ("select", ",".join(columns)),
                ("order", f"{order_column}.asc"),
                *filters,
                ("limit", PAGE_SIZE),
                ("offset", offset),
            ]
            url = f"{API}/{endpoint}?{urllib.parse.urlencode(parameters)}"
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "lamina-evaluation/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                page = json.load(response)
            rows.extend(page)
            progress.update(len(page))
            if len(page) < PAGE_SIZE:
                return rows
            offset += PAGE_SIZE


def csv_value(value):
    """Serialize IEDB array-valued fields as compact JSON inside the CSV."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in tqdm(rows, desc=f"Writing {path.name}", unit="row"):
            writer.writerow({column: csv_value(row.get(column)) for column in columns})
    print(f"Wrote {len(rows):,} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # IEDB stores reference years in an array. Querying one year at a time is
    # the simplest way to express the fixed 2021--2026 interval. The same
    # assay can match more than one year, so the assay ID removes duplicates.
    recent_by_id = {}
    print("Downloading recent binding-affinity measurements")
    for year in YEARS:
        rows = fetch_pages(
            "mhc_search",
            BA_COLUMNS,
            [
                ("mhc_class", "eq.I"),
                ("linear_sequence_length", "gte.8"),
                ("linear_sequence_length", "lte.14"),
                ("reference_dates", f"cs.{{{year}}}"),
                ("quantitative_measure", "not.is.null"),
            ],
            "elution_id",
        )
        for row in rows:
            # The API's quantitative field also contains other assay types.
            # Retain IC50/KD affinity annotations, including KD (~EC50).
            assay = str(row.get("assay_names") or "").lower()
            if "ic50" in assay or "dissociation constant" in assay:
                recent_by_id[row["elution_id"]] = row

    recent = [recent_by_id[key] for key in sorted(recent_by_id)]
    write_csv(args.output_dir / "iedb_recent_ba.csv", BA_COLUMNS, recent)

    # Pearson et al. is the source-protein recovery (FRANK) cohort. Length and
    # MHC-class filters match the model's class-I 8--14-mer input domain.
    print("Downloading Pearson et al. eluted ligands")
    pearson = fetch_pages(
        "mhc_search",
        PEARSON_COLUMNS,
        [
            ("mhc_class", "eq.I"),
            ("pubmed_id", "eq.27841757"),
            ("linear_sequence_length", "gte.8"),
            ("linear_sequence_length", "lte.14"),
        ],
        "elution_id",
    )
    write_csv(
        args.output_dir / "pearson_el_current.csv",
        PEARSON_COLUMNS,
        pearson,
    )


if __name__ == "__main__":
    main()
