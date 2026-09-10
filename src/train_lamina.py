#!/usr/bin/env python3
"""Train one paper LAMINA model: ``python src/train_lamina.py ba|el``."""

import argparse
import gzip
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

try:
    from .io_utils import file_sha256
    from .lamina import LAMINA, encode_batch
except ImportError:  # Running this file directly from the repository root.
    from io_utils import file_sha256
    from lamina import LAMINA, encode_batch


# -----------------------------------------------------------------------------
# Fixed paper training settings
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data/NetMHCpan/netmhcpan_training.jsonl.gz"
OUTPUT_DIR = ROOT / "checkpoints"
SEED = 0
STEPS = 15_000
MICROBATCH_SIZE = 8_192
GRADIENT_ACCUMULATION = 4  # nominal batch size: 8,192 x 4 = 32,768; tails are retained
ROWS_PER_CYCLE = 200_000
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0
MOTIF_L2_WEIGHT = 1e-4
LOADER_WORKERS = 2
PREFETCH_BATCHES = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate(rows: list[tuple[str, list[str], float]]):
    """Choose one candidate allele and form the fixed 34-by-14 paper inputs."""

    peptides = [row[0] for row in rows]
    hlas = [random.choice(row[1]) for row in rows]
    targets = torch.tensor([row[2] for row in rows], dtype=torch.float32)
    hla_ids, _ = encode_batch(hlas, 34)
    peptide_ids, peptide_lengths = encode_batch(peptides, 14)
    return hla_ids, peptide_ids, peptide_lengths, targets


def deterministic_rng(*parts: object) -> np.random.Generator:
    """Derive a reproducible random stream from the seed and stream labels."""

    entropy = [SEED]
    for part in parts:
        encoded = str(part).encode("utf-8")
        entropy.append(len(encoded))
        entropy.extend(encoded)
    return np.random.default_rng(np.random.SeedSequence(entropy))


class RotatingSampler(Sampler[int]):
    """Visit the whole task pool through shuffled 200,000-row loader cycles.

    The initial pool permutation is fixed. Consecutive cycles take consecutive
    slices, wrapping at the end; each selected slice is shuffled independently.
    """

    def __init__(self, row_count: int, task: str) -> None:
        self.order = np.arange(row_count, dtype=np.int32)
        deterministic_rng("source", task).shuffle(self.order)
        self.count = min(ROWS_PER_CYCLE, row_count)
        self.cycle = 0

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        start = (self.cycle * self.count) % len(self.order)
        positions = (start + np.arange(self.count)) % len(self.order)
        indexes = self.order[positions].copy()
        deterministic_rng("mixed", self.cycle).shuffle(indexes)
        return iter(indexes.tolist())


def seed_worker(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def train(task: str, data_path: Path = DATA_PATH, output_dir: Path = OUTPUT_DIR) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # The formatter has already removed non-training records. Keep only the
    # requested task: BA targets are normalized affinities in [0, 1], whereas
    # EL targets are binary presentation labels.
    rows = []
    with gzip.open(data_path, "rt", encoding="utf-8") as handle:
        for line in tqdm(handle, desc=f"Loading {task.upper()} training rows", unit="row"):
            record = json.loads(line)
            if record["dataset"] == task:
                rows.append(
                    (
                        record["peptide"],
                        record["pseudosequences"],
                        float(record["target"]),
                    )
                )

    if not rows:
        raise ValueError(f"No {task.upper()} training rows found in {data_path}")

    # The shorter final microbatch receives the same loss weight as a full one.
    sampler = RotatingSampler(len(rows), task)
    loader = DataLoader(
        rows,
        batch_size=MICROBATCH_SIZE,
        sampler=sampler,
        drop_last=False,
        num_workers=LOADER_WORKERS,
        collate_fn=collate,
        worker_init_fn=seed_worker if LOADER_WORKERS else None,
        **({"persistent_workers": True, "prefetch_factor": PREFETCH_BATCHES}
           if LOADER_WORKERS else {}),
        generator=torch.Generator().manual_seed(SEED),
    )
    batches = iter(loader)

    model = LAMINA().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 - step / STEPS
    )
    use_bf16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lamina_{task}.pt"
    metadata = {
        "task": task,
        "seed": SEED,
        "optimizer_steps": STEPS,
        "microbatch_size": MICROBATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "sampling": "fixed shuffled pool; consecutive rotating slices; shuffle each cycle",
        "rows_per_cycle": len(sampler),
        "drop_last": False,
        "rows": len(rows),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "learning_rate_schedule": "linear decay to zero",
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "motif_l2_weight": MOTIF_L2_WEIGHT,
        "loader_workers": LOADER_WORKERS,
        "prefetch_batches": PREFETCH_BATCHES,
        "device": str(DEVICE),
        "dtype": "bfloat16 convolutions; float32 accumulation" if use_bf16 else "float32",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else None,
        "data_file": data_path.name,
        "data_sha256": file_sha256(data_path),
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("lamina.py", "train_lamina.py", "prepare_netmhcpan_data.py", "io_utils.py")
        },
    }

    print(f"Task: {task.upper()}; rows: {len(rows):,}; device: {DEVICE}")
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    example_presentations = 0
    progress = tqdm(range(1, STEPS + 1), desc=f"Training {task.upper()}")
    for _ in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=DEVICE)

        for _ in range(GRADIENT_ACCUMULATION):
            try:
                batch = next(batches)
            except StopIteration:
                sampler.cycle += 1
                batches = iter(loader)
                batch = next(batches)

            hla_ids, peptide_ids, peptide_lengths, targets = (
                value.to(DEVICE, non_blocking=True) for value in batch
            )
            example_presentations += targets.shape[0]
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits, motif_l2 = model(hla_ids, peptide_ids, peptide_lengths)
                if task == "ba":
                    data_loss = F.mse_loss(torch.sigmoid(logits), targets)
                else:
                    data_loss = F.binary_cross_entropy_with_logits(logits, targets)
                loss = data_loss + MOTIF_L2_WEIGHT * motif_l2

            (loss / GRADIENT_ACCUMULATION).backward()
            step_loss += loss.detach() / GRADIENT_ACCUMULATION

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()
        progress.set_postfix(loss=f"{step_loss.item():.4f}")

    torch.save(model.state_dict(), output_path)
    metadata["example_presentations"] = example_presentations
    metadata["checkpoint_sha256"] = file_sha256(output_path)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("ba", "el"))
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    train(args.task, args.data_path, args.output_dir)
