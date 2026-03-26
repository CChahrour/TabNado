"""Create a minimal mock zarr dataset for integration tests."""

from pathlib import Path

import numpy as np
import zarr
from quantnado.dataset.store_bam import _compute_sample_hash

DEFAULT_OUT = Path(__file__).parent / "data" / "dataset" / "coverage.zarr"

TARGET_STR = "TEST"

SAMPLE_NAMES = [
    f"ChIP-CELL_{TARGET_STR}",
    f"CAT-CELL_{TARGET_STR}",
    f"CM-CELL_{TARGET_STR}",
    "ChIP-CELL_H3K4me3",
    "ChIP-CELL_H3K27ac",
    "ChIP-CELL_CTCF",
]
CHROMOSOMES = ["chr1", "chr2", "chr3", "chr8", "chr9"]
CHROMSIZES = {
    "chr1": 20_000,
    "chr2": 20_000,
    "chr3": 20_000,
    "chr8": 20_000,
    "chr9": 20_000,
}
CHUNK_LEN = 65_536
N = len(SAMPLE_NAMES)


def create_test_dataset(out: Path | None = None) -> Path:
    out_path = out or DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    rng = np.random.default_rng(42)
    root = zarr.open_group(str(out_path), mode="w", zarr_format=3)

    target_indices = [i for i, s in enumerate(SAMPLE_NAMES) if TARGET_STR in s]
    feature_indices = [i for i, s in enumerate(SAMPLE_NAMES) if TARGET_STR not in s]

    # Chromosome arrays are stored as (sample x position).
    for chrom, size in CHROMSIZES.items():
        arr = root.create_array(
            name=chrom,
            shape=(N, size),
            chunks=(1, CHUNK_LEN),
            dtype=np.uint16,
            fill_value=0,
        )
        # Build a smooth latent signal by generating coarse values every 500bp
        # then linearly interpolating, so structure spans window/tile scales.
        coarse_size = size // 500 + 2
        coarse = rng.uniform(2, 12, size=coarse_size)
        coarse_x = np.linspace(0, size - 1, coarse_size)
        fine_x = np.arange(size)
        latent = np.interp(fine_x, coarse_x, coarse)

        data = np.zeros((N, size), dtype=np.uint16)
        for i in target_indices:
            noise = rng.normal(0, 0.5, size=size)
            data[i] = np.clip(latent + noise, 0, 65535).astype(np.uint16)
        for i in feature_indices:
            noise = rng.normal(0, 1.5, size=size)
            data[i] = np.clip(latent + noise, 0, 65535).astype(np.uint16)
        arr[:] = data

    meta = root.create_group("metadata")

    completed = meta.create_array("completed", shape=(N,), dtype=bool, fill_value=False)
    completed[:] = [True] * N

    total_reads = meta.create_array(
        "total_reads", shape=(N,), dtype=np.int64, fill_value=0
    )
    total_reads[:] = rng.integers(500_000, 2_000_000, size=N)

    mean_read_length = meta.create_array(
        "mean_read_length", shape=(N,), dtype=np.float32, fill_value=np.nan
    )
    mean_read_length[:] = np.full(N, 100.0, dtype=np.float32)

    sparsity = meta.create_array(
        "sparsity", shape=(N,), dtype=np.float32, fill_value=np.nan
    )
    sparsity[:] = rng.uniform(20, 80, size=N).astype(np.float32)

    sample_hashes = meta.create_array(
        "sample_hashes", shape=(N, 16), dtype=np.uint8, fill_value=0
    )
    sample_hashes[:] = rng.integers(0, 255, size=(N, 16), dtype=np.uint8)

    root.attrs.update(
        {
            "chromosomes": CHROMOSOMES,
            "chromsizes": CHROMSIZES,
            "n_samples": N,
            "chunk_len": CHUNK_LEN,
            "construction_compression": "default",
            "structure": "per-chromosome (sample x position)",
            "bin_size": 1,
            "sample_names": SAMPLE_NAMES,
            "sample_names_hash": _compute_sample_hash(SAMPLE_NAMES),
            "stranded": {s: "" for s in SAMPLE_NAMES},
        }
    )

    print(f"Created {out_path}")
    print(f"  samples ({N}): {SAMPLE_NAMES}")
    print(f"  chromosomes: {CHROMOSOMES}")
    return out_path


if __name__ == "__main__":
    create_test_dataset()
