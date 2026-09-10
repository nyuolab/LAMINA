"""Map deposited class-I heavy chains to explicitly numbered mature HLA residues.

The reference supplies numbering only. Model inputs always contain the residues
in the deposited polymer, including substitutions relative to a named allele.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from Bio.Align import PairwiseAligner, substitution_matrices

from pmhc_mhci_common import AA3_TO_1, MHC_I_PSEUDO_POSITIONS, STANDARD_AA


REFERENCE_PATH = Path(__file__).with_name("hla_a_mature_P04439.json")
GROOVE_LENGTH = 182  # UniProt alpha-1 and alpha-2 domains, mature numbering.
MIN_GROOVE_COVERAGE = 0.90
MIN_GROOVE_IDENTITY = 0.70
MAX_OPTIMAL_ALIGNMENTS = 1000
ALIGNMENT_VERSION = 1


class HlaMappingError(ValueError):
    """An unsafe numbering assignment, with a machine-readable rejection code."""

    def __init__(self, code: str, **details):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {json.dumps(details, sort_keys=True)}")


@lru_cache(maxsize=1)
def load_reference() -> dict:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    sequence = reference["mature_sequence"]
    if hashlib.sha256(sequence.encode("ascii")).hexdigest() != reference["sequence_sha256"]:
        raise ValueError(f"Mature HLA reference checksum mismatch: {REFERENCE_PATH}")
    if len(sequence) != 341 or not set(sequence) <= STANDARD_AA:
        raise ValueError("Invalid pinned mature HLA reference sequence")
    return reference


def _canonical_map(alignment, reference_length: int) -> dict[int, int | None]:
    result = {position: None for position in range(1, reference_length + 1)}
    for reference_block, deposited_block in zip(*alignment.aligned):
        reference_start, reference_end = map(int, reference_block)
        deposited_start, deposited_end = map(int, deposited_block)
        if reference_end - reference_start != deposited_end - deposited_start:
            raise ValueError("Nonlinear alignment block")
        for offset in range(reference_end - reference_start):
            result[reference_start + offset + 1] = deposited_start + offset + 1
    return result


def align_hla_to_reference(sequence: str) -> dict:
    """Align the full deposited polymer to the pinned full mature HLA sequence.

    Free terminal gaps accommodate signal peptides, cloning tags and terminal
    truncations. Affine internal gaps preserve explicit numbering across indels.
    All optimal alignments must agree throughout the antigen-binding groove;
    otherwise the record is rejected. Ambiguities outside the groove are flagged
    and their positions left unmapped, never resolved by an arbitrary first hit.
    Missing any of the 34 model positions is also a rejection.
    """
    sequence = "".join(str(sequence).split()).upper()
    if not sequence or not set(sequence) <= STANDARD_AA | {"X"}:
        raise HlaMappingError("unsupported_heavy_chain_sequence")
    reference = load_reference()
    mature = reference["mature_sequence"]
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.end_gap_score = 0.0
    alignments = aligner.align(mature, sequence)
    try:
        count = len(alignments)
    except OverflowError:
        raise HlaMappingError("too_many_optimal_hla_alignments") from None
    if count > MAX_OPTIMAL_ALIGNMENTS:
        raise HlaMappingError("too_many_optimal_hla_alignments", count=count)
    if not count:
        raise HlaMappingError("no_hla_alignment")

    first = alignments[0]
    mapping = _canonical_map(first, len(mature))
    ambiguous = set()
    for alignment_index in range(1, count):
        alignment = alignments[alignment_index]
        alternative = _canonical_map(alignment, len(mature))
        ambiguous.update(position for position in mapping if mapping[position] != alternative[position])
    groove_ambiguity = sorted(position for position in ambiguous if position <= GROOVE_LENGTH)
    if groove_ambiguity:
        raise HlaMappingError("ambiguous_hla_groove_alignment", canonical_positions=groove_ambiguity, optimal_alignments=count)
    for position in ambiguous:
        mapping[position] = None

    covered = [position for position in range(1, GROOVE_LENGTH + 1) if mapping[position] is not None]
    matches = sum(mature[position - 1] == sequence[mapping[position] - 1] for position in covered)
    coverage = len(covered) / GROOVE_LENGTH
    identity = matches / len(covered) if covered else 0.0
    if coverage < MIN_GROOVE_COVERAGE or identity < MIN_GROOVE_IDENTITY:
        raise HlaMappingError("low_confidence_hla_alignment", groove_coverage=coverage, groove_identity=identity)
    missing = [position for position in MHC_I_PSEUDO_POSITIONS if mapping[position] is None]
    if missing:
        raise HlaMappingError("missing_canonical_pseudosequence_positions", canonical_positions=missing)
    pseudosequence = "".join(sequence[mapping[position] - 1] for position in MHC_I_PSEUDO_POSITIONS)
    if not set(pseudosequence) <= STANDARD_AA:
        raise HlaMappingError("unsupported_deposited_pseudosequence_residue")

    offsets = {mapping[position] - position for position in MHC_I_PSEUDO_POSITIONS}
    return {"canonical_to_deposited": mapping, "pseudosequence": pseudosequence, "metadata": {
        "version": ALIGNMENT_VERSION,
        "status": "accepted",
        "reference_accession": reference["accession"],
        "reference_sequence_version": reference["sequence_version"],
        "reference_sequence_sha256": reference["sequence_sha256"],
        "deposited_sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "algorithm": "global BLOSUM62; internal gap open -10, extend -0.5; free terminal gaps",
        "score": float(first.score),
        "optimal_alignments": count,
        "groove_coverage": coverage,
        "groove_identity": identity,
        "minimum_groove_coverage": MIN_GROOVE_COVERAGE,
        "minimum_groove_identity": MIN_GROOVE_IDENTITY,
        "mapping_numbering": "canonical_to_deposited[index] maps mature canonical index+1 to one-based deposited polymer label_seq_id; null is unmapped",
        "canonical_to_deposited": [mapping[position] for position in range(1, len(mature) + 1)],
        "unmapped_canonical_positions": [position for position, value in mapping.items() if value is None],
        "ambiguous_outside_groove_positions": sorted(ambiguous),
        "pseudosequence_position_offset": next(iter(offsets)) if len(offsets) == 1 else None,
    }}


def validate_coordinate_residue(expected, component, chain, label_seq_id, canonical_position):
    """Reject a standard atom identity that contradicts its deposited polymer."""
    actual = AA3_TO_1.get(component)
    if actual in STANDARD_AA and actual != expected:
        raise HlaMappingError(
            "conflicting_deposited_hla_identity", canonical_position=canonical_position,
            label_asym_id=chain, label_seq_id=label_seq_id,
            polymer_residue=expected, atom_component=component, atom_residue=actual,
        )
