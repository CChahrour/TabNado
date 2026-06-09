"""Create a minimal mock zarr dataset for integration tests."""

from pathlib import Path

import numpy as np
import zarr

DEFAULT_OUT = Path(__file__).parent / "data" / "dataset"

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


def _parse_sample_name(name: str) -> tuple[str, str]:
    """Return (assay, ip) from a name like 'ChIP-CELL_H3K4me3'."""
    assay = name.split("-")[0]
    ip = name.rsplit("_", 1)[-1] if "_" in name else ""
    return assay, ip


def _make_sample_zarr(
    out_path: Path,
    sample_name: str,
    assay: str,
    ip: str,
    signal: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> None:
    root = zarr.open_group(str(out_path), mode="w", zarr_format=3)

    for chrom, data in signal.items():
        chrom_group = root.require_group(chrom)
        arr = chrom_group.create_array(
            name="coverage",
            shape=(1, len(data)),
            chunks=(1, CHUNK_LEN),
            dtype=np.uint16,
            fill_value=0,
        )
        arr[0, :] = data

    meta = root.require_group("metadata")
    completed = meta.create_array("completed", shape=(1,), dtype=bool, fill_value=False)
    completed[0] = True
    total_reads = meta.create_array(
        "total_reads", shape=(1,), dtype=np.int64, fill_value=0
    )
    total_reads[0] = int(rng.integers(500_000, 2_000_000))
    mean_read_length = meta.create_array(
        "mean_read_length", shape=(1,), dtype=np.float32, fill_value=np.nan
    )
    mean_read_length[0] = np.float32(100.0)
    sparsity = meta.create_array(
        "sparsity", shape=(1,), dtype=np.float32, fill_value=np.nan
    )
    sparsity[0] = np.float32(rng.uniform(20, 80))

    root.attrs.update(
        {
            "assay": assay,
            "sample": sample_name,
            "ip": ip,
            "stranded": "",
            "chromsizes": CHROMSIZES,
            "chunk_len": CHUNK_LEN,
            "chromosomes": CHROMOSOMES,
            "bin_size": 1,
            "construction_compression": "default",
        }
    )


def create_test_dataset(out: Path | None = None) -> Path:
    out_dir = out or DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if all expected zarr stores already exist.
    expected = [out_dir / f"{name}.zarr" for name in SAMPLE_NAMES]
    if all(p.exists() for p in expected):
        return out_dir

    rng = np.random.default_rng(42)

    target_set = {n for n in SAMPLE_NAMES if TARGET_STR in n}

    for sample_name in SAMPLE_NAMES:
        assay, ip = _parse_sample_name(sample_name)
        is_target = sample_name in target_set
        signal: dict[str, np.ndarray] = {}
        for chrom, size in CHROMSIZES.items():
            coarse_size = size // 500 + 2
            coarse = rng.uniform(2, 12, size=coarse_size)
            coarse_x = np.linspace(0, size - 1, coarse_size)
            latent = np.interp(np.arange(size), coarse_x, coarse)
            noise_std = 0.5 if is_target else 1.5
            noise = rng.normal(0, noise_std, size=size)
            signal[chrom] = np.clip(latent + noise, 0, 65535).astype(np.uint16)

        zarr_path = out_dir / f"{sample_name}.zarr"
        _make_sample_zarr(zarr_path, sample_name, assay, ip, signal, rng)

    print(f"Created {len(SAMPLE_NAMES)} per-sample zarr stores in {out_dir}")
    print(f"  samples: {SAMPLE_NAMES}")
    print(f"  chromosomes: {CHROMOSOMES}")
    return out_dir


if __name__ == "__main__":
    create_test_dataset()
