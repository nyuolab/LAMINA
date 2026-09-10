"""
LAMINA

Each model has one learned interaction map and predicts one task: either
binding affinity (BA) or eluted-ligand presentation (EL). The two tasks use
separate checkpoints but exactly the same architecture.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn


CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_ALPHABET = CANONICAL_AA + "OU"
PAD_ID, UNK_ID = 0, 3
AA_TO_ID = {aa: index + 4 for index, aa in enumerate(AA_ALPHABET)}
AA_BYTE_TO_ID = np.full(256, UNK_ID, dtype=np.int64)
for amino_acid, token_id in AA_TO_ID.items():
    AA_BYTE_TO_ID[ord(amino_acid)] = token_id

VOCAB_SIZE = len(AA_ALPHABET) + 4
FEATURE_DIM = 256
MOTIF_DIM = 32
MOTIFS_PER_LENGTH = 8
MOTIF_LENGTHS = tuple(range(1, 9))
MOTIF_GROUPS = len(MOTIF_LENGTHS) * MOTIFS_PER_LENGTH
MAX_POSITION_EMBEDDINGS = 34


def encode_batch(
    sequences: list[str],
    width: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode nonempty ASCII sequences, padding on the right with ID zero.

    Residue IDs are 4..25; unrecognized ASCII characters use unknown ID 3.
    IDs 1 and 2 are reserved by the released checkpoints.
    """
    if not sequences:
        raise ValueError("Cannot encode an empty batch")
    width = max(map(len, sequences)) if width is None else width
    if width < 1:
        raise ValueError("Encoding width must be positive")
    token_ids = np.full((len(sequences), width), PAD_ID, dtype=np.int64)
    lengths = np.empty(len(sequences), dtype=np.int64)
    for index, sequence in enumerate(sequences):
        if not 1 <= len(sequence) <= width:
            raise ValueError(f"Sequence length {len(sequence)} is outside 1..{width}")
        try:
            encoded = sequence.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("Amino-acid sequences must contain ASCII characters") from error
        token_ids[index, :len(encoded)] = AA_BYTE_TO_ID[np.frombuffer(encoded, dtype=np.uint8)]
        lengths[index] = len(encoded)
    return torch.from_numpy(token_ids).to(device), torch.from_numpy(lengths).to(device)


class LAMINA(nn.Module):
    """Bias-free latent motif interaction model with one 64 x 64 map."""

    def __init__(self) -> None:
        super().__init__()

        # Amino-acid identity and sequence position share the same 256-dimensional
        # space. The embedding tables are shared by the HLA and peptide inputs.
        self.token_embedding = nn.Embedding(VOCAB_SIZE, FEATURE_DIM)
        self.position_embedding = nn.Embedding(MAX_POSITION_EMBEDDINGS, FEATURE_DIM)

        # For every window length 1..8, eight filters each emit a 32-dimensional
        # latent motif. HLA and peptide filters are learned independently.
        self.hla_convs = self._make_convolutions()
        self.peptide_convs = self._make_convolutions()

        # One coefficient is learned for every pair of HLA and peptide motif banks.
        self.interaction_map = nn.Parameter(torch.ones(MOTIF_GROUPS, MOTIF_GROUPS))

    @staticmethod
    def _make_convolutions() -> nn.ModuleList:
        return nn.ModuleList(
            nn.Conv1d(
                FEATURE_DIM,
                MOTIFS_PER_LENGTH * MOTIF_DIM,
                kernel_size=length,
                bias=False,
            )
            for length in MOTIF_LENGTHS
        )

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return self.token_embedding(token_ids) + self.position_embedding(positions)[None]

    def _pool_motifs(
        self,
        token_ids: torch.Tensor,
        convolutions: nn.ModuleList,
        lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return position-summed motifs, valid motif counts, and squared activity."""

        convolution_input = self._embed(token_ids).transpose(1, 2)
        batch_size = token_ids.shape[0]
        pooled = []
        valid_counts = torch.zeros(batch_size, dtype=torch.long, device=token_ids.device)
        squared_activity = torch.zeros((), dtype=torch.float32, device=token_ids.device)

        for motif_length, convolution in zip(MOTIF_LENGTHS, convolutions):
            # [batch, position, motif number, motif dimension]
            features = convolution(convolution_input).transpose(1, 2)
            features = features.reshape(
                batch_size, -1, MOTIFS_PER_LENGTH, MOTIF_DIM
            )

            if lengths is None:
                # HLA pseudosequences are always the complete 34-residue sequence.
                pooled.append(features.sum(dim=1, dtype=torch.float32))
                squared_activity += features.float().square().sum()
                valid_counts += features.shape[1] * MOTIFS_PER_LENGTH
            else:
                # Peptides are padded to a common batch width. A convolution start
                # is valid only when its complete window lies inside the true peptide.
                starts = torch.arange(features.shape[1], device=token_ids.device)[None]
                valid = starts + motif_length <= lengths[:, None]
                valid = valid[:, :, None, None]
                pooled.append((features * valid).sum(dim=1, dtype=torch.float32))
                squared_activity += (features.float().square() * valid).sum()
                valid_counts += valid[:, :, 0, 0].sum(dim=1) * MOTIFS_PER_LENGTH

        return torch.cat(pooled, dim=1), valid_counts, squared_activity

    def forward(
        self,
        hla_token_ids: torch.Tensor,
        peptide_token_ids: torch.Tensor,
        peptide_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return task logits and the paper's mean-squared motif penalty."""

        hla_motifs, hla_counts, hla_squared = self._pool_motifs(
            hla_token_ids, self.hla_convs, None
        )
        peptide_motifs, peptide_counts, peptide_squared = self._pool_motifs(
            peptide_token_ids, self.peptide_convs, peptide_lengths
        )

        # The position-independent interaction map lets the full window-by-window
        # sum factor into this small bilinear operation (Methods, Eq. 4). Pooling
        # and scoring remain float32 when the convolutions run under BF16 autocast,
        # matching the accumulation precision of the paper's fused implementation.
        with torch.autocast(device_type=hla_token_ids.device.type, enabled=False):
            logits = torch.einsum(
                "bgd,gh,bhd->b",
                hla_motifs.float(),
                self.interaction_map.float(),
                peptide_motifs.float(),
            ) / (hla_counts * peptide_counts).float()

        # Regularize the mean squared value of every valid latent-motif component.
        motif_components = (hla_counts.sum() + peptide_counts.sum()) * MOTIF_DIM
        motif_l2 = (hla_squared + peptide_squared) / motif_components
        return logits, motif_l2


def load_model(path: str | Path, device: str | torch.device = "cpu") -> LAMINA:
    """Load a BA or EL state dictionary using the fixed LAMINA architecture."""

    model = LAMINA()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()
