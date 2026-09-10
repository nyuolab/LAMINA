"""Exact residue contributions for a fixed LAMINA model."""

import torch
import torch.nn.functional as F

if __package__:
    from .lamina import (
        LAMINA, MOTIF_DIM, MOTIF_GROUPS, MOTIF_LENGTHS, MOTIFS_PER_LENGTH, encode_batch,
    )
else:
    from lamina import (
        LAMINA, MOTIF_DIM, MOTIF_GROUPS, MOTIF_LENGTHS, MOTIFS_PER_LENGTH, encode_batch,
    )


class ResidueAttribution:
    """Cache residue components for one model in evaluation mode.

    Create a new instance after changing the model's weights or device.
    """

    def __init__(self, model: LAMINA):
        if model.training:
            raise ValueError("Attribution requires a model in evaluation mode")
        self.model = model
        self.cache = {}

    def _components(self, sequence, branch):
        key = branch, sequence
        if key in self.cache:
            return self.cache[key]
        device = self.model.token_embedding.weight.device
        token_ids, _ = encode_batch([sequence], device=device)
        embeddings = self.model._embed(token_ids)[0]
        components = embeddings.new_zeros((len(sequence), MOTIF_GROUPS, MOTIF_DIM))
        convolutions = getattr(self.model, f"{branch}_convs")
        pooled, count = [], 0
        for bank, (length, convolution) in enumerate(zip(MOTIF_LENGTHS, convolutions)):
            starts = len(sequence) - length + 1
            weights = convolution.weight.reshape(
                MOTIFS_PER_LENGTH, MOTIF_DIM, embeddings.shape[1], length
            )
            group = slice(bank * MOTIFS_PER_LENGTH, (bank + 1) * MOTIFS_PER_LENGTH)
            # Distribute each window's linear terms to its source residues.
            for offset in range(length):
                components[offset:offset + starts, group] += torch.einsum(
                    "sf,mdf->smd", embeddings[offset:offset + starts], weights[..., offset]
                )
            direct = F.conv1d(embeddings.T[None], convolution.weight).sum(-1)[0]
            pooled.append(direct.reshape(MOTIFS_PER_LENGTH, MOTIF_DIM))
            count += starts * MOTIFS_PER_LENGTH
        self.cache[key] = components, torch.cat(pooled), count
        return self.cache[key]

    @torch.inference_mode()
    def __call__(self, pseudosequence, peptide):
        """Return NumPy pair contributions, peptide contributions, and sum error."""
        if len(pseudosequence) != 34 or not 8 <= len(peptide) <= 14:
            raise ValueError("Attribution requires a 34-residue HLA and an 8–14-residue peptide")
        hla, pooled_hla, hla_count = self._components(pseudosequence, "hla")
        pep, pooled_pep, pep_count = self._components(peptide, "peptide")
        weights = self.model.interaction_map.float()
        denominator = hla_count * pep_count
        pair = torch.einsum("ugd,gq,vqd->uv", hla.float(), weights, pep.float()) / denominator
        direct = torch.einsum(
            "gd,gq,qd->", pooled_hla.float(), weights, pooled_pep.float()
        ) / denominator
        torch.testing.assert_close(pair.sum(), direct, atol=2e-5, rtol=2e-5)
        error = abs(float(pair.sum() - direct))
        return pair.cpu().numpy(), pair.sum(0).cpu().numpy(), error
