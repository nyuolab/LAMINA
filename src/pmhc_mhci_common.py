#!/usr/bin/env python3
"""The small amount of mmCIF plumbing shared by the structure scripts.

The weekly PDB archive is distributed as compressed mmCIF. We only need a
few tables from each file, so this module contains a tiny parser.
"""

from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


# All defaults are inside this repository. A different local PDB mirror can still be supplied with --pdb-root on each command line.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "Data"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" # this is where the preprocessing files are written
DEFAULT_PDB_ROOT = DEFAULT_DATA_DIR / "PDB"
DEFAULT_PSEUDOSEQ_PATH = (
    DEFAULT_DATA_DIR / "NetMHCpan" / "NetMHCpan_train" / "pseudoseqs"
)

# The 34 NetMHCpan class-I pseudosequence positions, numbered in the mature MHC heavy chain. These are the only MHC residues used by the model.
MHC_I_PSEUDO_POSITIONS = (
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
    84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159,
    163, 167, 171,
)

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
STRUCTURE_SUFFIXES = (".cif", ".cif.gz", ".mmcif", ".mmcif.gz")
HLA_ALLELE_RE = re.compile(
    r"\bHLA[-_\s]?([ABC])\*?[-_\s:]?(\d{2})[-_\s:]?(\d{2})\b",
    re.IGNORECASE,
)


def normalize_missing(value: object | None) -> str:
    """Convert the mmCIF missing markers '.' and '?' to an empty string."""
    value = "" if value is None else str(value).strip()
    return "" if value in {"", ".", "?"} else value


def clean_sequence(value: object | None) -> str:
    return re.sub(r"\s+", "", normalize_missing(value)).upper()


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _tokenize_line(line: str) -> list[str]:
    """Split an ordinary mmCIF line while respecting quoted values."""
    tokens, current, quote = [], [], ""
    for index, char in enumerate(line):
        if quote:
            if char == quote and (index + 1 == len(line) or line[index + 1].isspace()):
                tokens.append("".join(current))
                current, quote = [], ""
            else:
                current.append(char)
        elif char in {"'", '"'} and not current:
            quote = char
        elif char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _iter_cif_tokens(path: Path) -> Iterator[str]:
    """Yield both normal tokens and semicolon-delimited text blocks."""
    with open_text(path) as handle:
        text_block: list[str] | None = None
        for line in handle:
            if text_block is not None:
                if line.startswith(";"):
                    yield "".join(text_block).rstrip("\n")
                    text_block = None
                else:
                    text_block.append(line)
                continue
            if line.startswith(";"):
                text_block = [line[1:]]
                continue
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                yield "#"
            else:
                yield from _tokenize_line(line)


class _TokenStream:
    def __init__(self, tokens: Iterable[str]):
        self.tokens = iter(tokens)
        self.buffer: list[str] = []

    def get(self) -> str | None:
        return self.buffer.pop() if self.buffer else next(self.tokens, None)

    def push(self, token: str | None) -> None:
        if token is not None:
            self.buffer.append(token)


def parse_mmcif_categories(
    path: str | Path,
    wanted_categories: set[str],
    stop_before_categories: set[str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Read selected mmCIF categories into lists of row dictionaries."""
    stop_before_categories = stop_before_categories or set()
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    stream = _TokenStream(_iter_cif_tokens(Path(path)))

    while (token := stream.get()) is not None:
        if token == "#":
            continue
        if token == "loop_":
            columns: list[str] = []
            while (token := stream.get()) is not None and token.startswith("_"):
                columns.append(token)
            stream.push(token)
            if not columns:
                continue
            category = columns[0].split(".", 1)[0]
            if category in stop_before_categories:
                break
            keep = category in wanted_categories
            values: list[str] = []
            while (token := stream.get()) is not None and token != "#":
                if token == "loop_" or token.startswith("_"):
                    stream.push(token)
                    break
                if keep:
                    values.append(token)
                    if len(values) == len(columns):
                        rows[category].append({
                            column.split(".", 1)[-1]: normalize_missing(value)
                            for column, value in zip(columns, values)
                        })
                        values = []
            continue
        if token.startswith("_"):
            category = token.split(".", 1)[0]
            if category in stop_before_categories:
                break
            value = stream.get()
            if category in wanted_categories:
                rows[category].append({
                    token.split(".", 1)[-1]: normalize_missing(value)
                })
    return dict(rows)


def resolve_pdb_root(root: str | Path) -> Path:
    root = Path(root).expanduser()
    return root / "mmCIF" if (root / "mmCIF").is_dir() else root


def iter_structure_paths(root: str | Path) -> Iterator[Path]:
    """Iterate over the complete local weekly mmCIF mirror."""
    for path in sorted(resolve_pdb_root(root).rglob("*")):
        if path.is_file() and path.name.lower().endswith(STRUCTURE_SUFFIXES):
            yield path


def pdb_id_from_path(path: Path) -> str:
    name = path.name.lower()
    for suffix in (".cif.gz", ".mmcif.gz", ".cif", ".mmcif"):
        if name.endswith(suffix):
            return name.removesuffix(suffix).upper()
    return path.stem.upper()


def path_for_pdb_id(root: str | Path, pdb_id: str) -> Path:
    """Apply the wwPDB two-character shard layout to a four-character ID."""
    root = resolve_pdb_root(root)
    pdb_id = pdb_id.lower()
    shard = pdb_id[1:3]
    candidates = [
        root / shard / f"{pdb_id}{suffix}"
        for suffix in (".cif.gz", ".cif", ".mmcif.gz", ".mmcif")
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def extract_hla_alleles(text: str) -> list[str]:
    alleles = [
        f"HLA-{match.group(1).upper()}*{match.group(2)}:{match.group(3)}"
        for match in HLA_ALLELE_RE.finditer(text or "")
    ]
    return list(dict.fromkeys(alleles))


def _allele_keys(allele: object) -> list[str]:
    """Match the few spelling variants found in the NetMHCpan table."""
    value = normalize_missing(allele).upper()
    keys = [value, value.replace("*", ""), value.replace("*", "").replace(":", "")]
    return list(dict.fromkeys(keys))


def load_pseudoseqs(path: str | Path = DEFAULT_PSEUDOSEQ_PATH) -> dict[str, str]:
    pseudoseqs: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2 and not line.lstrip().startswith("#"):
                for key in _allele_keys(parts[0]):
                    pseudoseqs[key] = clean_sequence(parts[1])
    return pseudoseqs


def lookup_pseudoseq(
    alleles: Iterable[str], pseudoseqs: dict[str, str]
) -> tuple[str | None, str | None]:
    for allele in alleles:
        for key in _allele_keys(allele):
            if key in pseudoseqs:
                return allele, pseudoseqs[key]
    return None, None


def parse_float(value: object | None) -> float | None:
    value = normalize_missing(value)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        match = re.match(r"[-+]?\d+(?:\.\d+)?", value)
        return float(match.group()) if match else None
