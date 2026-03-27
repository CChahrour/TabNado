import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyranges1 as pr
import quantnado as qn
import seaborn as sns
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from tabnado.params import PipelineParams
from tabnado.utils import figure_style


def sliding_window(df, window_size: int, step_size: int, tile_size: int):
    steps = window_size // step_size
    row_idx = np.repeat(np.arange(len(df)), steps)
    bin_idx = np.tile(np.arange(steps), len(df))
    window_starts = df["Start"].values[row_idx] - window_size // 2
    starts = window_starts + bin_idx * step_size
    return pr.PyRanges(
        {
            "Chromosome": df["Chromosome"].values[row_idx],
            "Start": starts,
            "End": starts + tile_size,
            "Strand": df["Strand"].values[row_idx],
            "Name": df["gene_name"].values[row_idx],
            "Score": starts - df["Start"].values[row_idx],
        }
    )


def get_tss_windows(
    gtf_file: str,
    windows_bed: Path,
    window_size: int = 3000,
    step_size: int = 100,
    tile_size: int = 100,
):
    """Load TSS windows from BED if cached, otherwise generate from GTF and save."""
    if windows_bed.exists():
        logger.info(f"Loading TSS windows from {windows_bed}")
        tss_windows = pr.read_bed(windows_bed)
        if tss_windows is None:
            raise RuntimeError(f"Failed to read BED file: {windows_bed}")
        return tss_windows

    logger.info(f"Generating TSS windows and saving to {windows_bed}")
    gencode = pr.read_gtf(gtf_file)
    gencode = gencode[gencode.Feature == "gene"]
    gencode = gencode[gencode.gene_type == "protein_coding"]
    cols = ["Chromosome", "Feature", "Start", "End", "Strand", "gene_type", "gene_name"]
    gencode_tss = gencode[cols]
    gencode_tss["Start"] = np.where(
        gencode_tss.Strand == "+", gencode_tss.Start, gencode_tss.End
    )
    gencode_tss["End"] = gencode_tss["Start"] + 1
    gencode_tss = gencode_tss.drop(columns=["Feature"])
    n_before = len(gencode_tss)
    gencode_tss = pr.PyRanges(
        gencode_tss.drop_duplicates(subset=["Chromosome", "Start", "gene_name"])
    )
    n_after = len(gencode_tss)
    if n_before != n_after:
        logger.warning(
            f"{n_after} unique TSS loci from GTF (dropped {n_before - n_after} duplicate gene/position entries)"
        )
    else:
        logger.info(f"{n_after} protein-coding TSS loci")

    tss_windows = sliding_window(gencode_tss, window_size, step_size, tile_size)
    Path(windows_bed).parent.mkdir(parents=True, exist_ok=True)
    tss_windows.to_bed(str(windows_bed))  # type: ignore[operator]
    n_genes = n_after
    n_windows = len(tss_windows) // n_genes if n_genes else 0
    logger.info(f"{len(tss_windows)} TSS windows ({n_genes} loci × {n_windows} tiles)")
    return tss_windows


def build_signal_df(
    ds,
    samples: list[str],
    tss_windows,
    signal_path: str,
    rpkm_path: str | None = None,
    chunk_size_rows: int | None = None,
) -> pd.DataFrame:
    """Extract binned signal and scale, then save to parquet.

    Pipeline:
        0. RPKM normalisation (lazy, using ds.coverage.library_sizes)
        1. Compute array into memory (cached to rpkm_path if provided)
        2. UQ normalisation — divide per column by 75th percentile of non-zero values
        3. Clip at 99.5th percentile per column
        4. log1p (numpy)
        5. MinMax [0, 1] per column (sklearn MinMaxScaler)
    """
    if rpkm_path and os.path.exists(rpkm_path):
        logger.info(f"Loading cached RPKM array from {rpkm_path}")
        rpkm_df = pd.read_parquet(rpkm_path)
        arr = rpkm_df.values.astype("float32")
        window_names = rpkm_df.index.values
        sample_names = rpkm_df.columns.values
    else:
        logger.info("Extracting signal from dataset")
        binned_signal = ds.reduce(ranges_df=tss_windows, samples=samples)
        if ds.coverage is None:
            raise RuntimeError(
                "Dataset has no coverage data — cannot compute library sizes"
            )
        library_sizes = ds.coverage.library_sizes

        logger.info("Normalising to RPKM")
        rpkm_signal = qn.normalise(
            data=binned_signal["mean"], method="rpkm", library_sizes=library_sizes
        )

        n_rows, n_samples = rpkm_signal.data.shape
        if chunk_size_rows is None or int(chunk_size_rows) <= 0:
            effective_chunk_rows = int(n_rows)
        else:
            effective_chunk_rows = min(int(chunk_size_rows), int(n_rows))
        logger.info(
            f"Using chunk size {effective_chunk_rows} rows × {n_samples} samples"
        )

        arr = (
            rpkm_signal.data.astype("float32")
            .rechunk((effective_chunk_rows, n_samples))
            .compute()
        )
        window_names = rpkm_signal.coords["name"].values
        sample_names = rpkm_signal.coords["sample"].values

        if rpkm_path:
            logger.info(f"Caching RPKM array to {rpkm_path}")
            pd.DataFrame(arr, index=window_names, columns=sample_names).to_parquet(
                rpkm_path
            )

    # UQ normalisation on non-zeros per column
    non_zeros = np.where(arr > 0, arr, np.nan)
    upper_quartile = np.nanpercentile(non_zeros, 75, axis=0)
    upper_quartile = np.where(upper_quartile == 0, 1.0, upper_quartile)
    arr = arr / upper_quartile

    # clip at 99.5th percentile per column
    arr = np.clip(arr, None, np.percentile(arr, 99.5, axis=0))

    # log1p
    arr = np.log1p(arr)

    # MinMax [0, 1] per column
    arr = MinMaxScaler().fit_transform(arr).astype(np.float32)

    signal_df = pd.DataFrame(arr, index=window_names, columns=sample_names)
    signal_df = signal_df.fillna(0)
    tss_coords = tss_windows.Start.values - tss_windows.Score.values  # type: ignore[union-attr]
    signal_df.index = pd.MultiIndex.from_arrays(
        [
            tss_windows.Chromosome.values,  # type: ignore[union-attr]
            tss_coords,
            tss_windows.Name.values,  # type: ignore[union-attr]
            tss_windows.Score.values,  # type: ignore[union-attr]
        ],
        names=["contig", "tss_coord", "region", "tss_offset"],
    )
    n_windows, n_samples = signal_df.shape
    logger.info(
        f"Saving scaled signal to {signal_path} — {n_windows} windows × {n_samples} samples"
    )
    signal_df.to_parquet(signal_path)
    return signal_df


def reshape_signal_to_region_features(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape signal_df from (contig, tss_coord, region, tss_offset) rows x (samples) columns
    to (contig, tss_coord, region) rows x (sample_tss_offset) columns.
    """
    df = signal_df.reset_index()
    df_indexed = df.set_index(["contig", "tss_coord", "region", "tss_offset"])
    stacked = df_indexed.stack()
    stacked.index.names = ["contig", "tss_coord", "region", "tss_offset", "sample"]
    result = stacked.unstack(["sample", "tss_offset"])
    result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]
    result.columns.name = None
    return result


def get_samples(
    ds,
    target: str,
    exclude_ips: list[str] | None,
    prefixes: list[str] | None = None,
    min_target: int = 1,
    min_features: int = 10,
):
    """
    Filter dataset metadata to samples usable for modelling.
    Returns (samples, target_cols, feature_cols).
    """
    prefixes = prefixes or ["CAT", "ChIP", "CM"]
    assay_prefixes_norm = [str(x).strip().upper() for x in prefixes if str(x).strip()]
    exclude_ips = exclude_ips or []

    metadata = ds.get_metadata()
    metadata["assay"] = metadata.index.str.split("-").str[0].str.upper()
    factors = metadata[metadata.assay.isin(assay_prefixes_norm)].copy()

    if factors.empty:
        raise ValueError(
            "No samples matched assay prefixes {}. Check params.prefixes and sample naming.".format(
                prefixes
            )
        )
    factors["cell_type"] = (
        factors.index.str.split("_")
        .str[0]
        .str.split("-")
        .str[1:]
        .str.join("-")
        .str.replace(r"-\d$", "", regex=True)
    )
    factors["ip"] = factors.index.str.split("_").str[1]
    factors = factors[~factors.ip.isin(exclude_ips)]

    sample_counts = factors[["assay", "cell_type", "ip"]].value_counts()
    sample_counts = sample_counts.reset_index(name="count").sort_values(
        ["cell_type", "count", "ip"], ascending=False
    )
    model_samples = (
        sample_counts.assign(
            role=lambda df: df["ip"].map(
                lambda x: "target" if x == target else "features"
            )
        )
        .groupby(["cell_type", "role"])["count"]
        .sum()
        .unstack(fill_value=0)
        .sort_values(["features", "target"], ascending=False)
    )
    model_samples["n_feature_ips"] = (
        sample_counts[sample_counts["ip"] != target]
        .groupby("cell_type")["ip"]
        .nunique()
    )
    factors_model = factors[
        factors.cell_type.isin(
            model_samples[
                (model_samples.target >= min_target)
                & (model_samples.n_feature_ips >= min_features)
            ].index
        )
    ]
    samples = factors_model.index.tolist()
    target_cols = [s for s in samples if target in s]
    feature_cols = [s for s in samples if target not in s]
    return samples, target_cols, feature_cols


def validate_features(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Warn and fix any out-of-[0,1] or NaN values in feature columns."""
    splits = [("train", train_data), ("eval", eval_data), ("test", test_data)]
    result = []
    for split_name, df in splits:
        lo = df[feature_cols].min().min()
        hi = df[feature_cols].max().max()
        nan_count = df[feature_cols].isnull().sum().sum()
        if nan_count > 0:
            logger.warning(
                f"{split_name}: features contain NaN (count={nan_count}) — filling with 0"
            )
            df = df.copy()
            df[feature_cols] = df[feature_cols].fillna(0)
        if lo < 0.0 or hi > 1.0:
            logger.warning(
                f"{split_name}: features out of [0,1] (min={lo:.4f}, max={hi:.4f}) — clipping"
            )
            df = df.copy()
            df[feature_cols] = df[feature_cols].clip(lower=0.0, upper=1.0)
        else:
            logger.info(
                f"{split_name}: features in range [0,1] ✓ (min={lo:.4f}, max={hi:.4f})"
            )
        result.append(df)
    return result[0], result[1], result[2]


def stratified_sample(df, target_cols: list[str], frac: float, seed: int = 42):
    """Stratified subsample by binning mean target signal into deciles."""
    if df.empty:
        return df
    if not target_cols:
        return df
    if frac >= 1:
        return df.sample(frac=1.0, random_state=seed)

    mean_signal = df[target_cols].mean(axis=1)
    bins = pd.qcut(mean_signal, q=min(10, len(df)), labels=False, duplicates="drop")

    sampled = df.groupby(bins, group_keys=False).apply(
        lambda g: g.sample(n=max(1, int(np.ceil(len(g) * frac))), random_state=seed)
    )

    if sampled.empty:
        return df.sample(n=1, random_state=seed)
    return sampled


def load_or_build_datasets(
    ds,
    samples: list[str],
    gtf_file: str,
    windows_bed: Path,
    signal_path: str,
    dataset_train_path: str,
    dataset_eval_path: str,
    dataset_test_path: str,
    eval_chr: str = "chr8",
    test_chr: str = "chr9",
    window_size: int = 3000,
    step_size: int = 100,
    tile_size: int = 100,
    fig_dir: str | None = None,
    target_cols: list[str] | None = None,
    chunk_size_rows: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_data, eval_data, test_data), building from scratch if not cached."""
    if (
        os.path.exists(dataset_train_path)
        and os.path.exists(dataset_eval_path)
        and os.path.exists(dataset_test_path)
    ):
        logger.info("Loading cached train/eval/test datasets")
        train_data = pd.read_parquet(dataset_train_path)
        eval_data = pd.read_parquet(dataset_eval_path)
        test_data = pd.read_parquet(dataset_test_path)
        if any(df.isnull().values.any() for df in [train_data, eval_data, test_data]):
            logger.warning("NaNs found in parquets — filling with 0 and re-saving")
            train_data = train_data.fillna(0)
            eval_data = eval_data.fillna(0)
            test_data = test_data.fillna(0)
            train_data.to_parquet(dataset_train_path)
            eval_data.to_parquet(dataset_eval_path)
            test_data.to_parquet(dataset_test_path)
        return train_data, eval_data, test_data

    if os.path.exists(signal_path):
        logger.info(f"Loading cached scaled signal from {signal_path}")
        signal_df = pd.read_parquet(signal_path)
    else:
        tss_windows = get_tss_windows(
            gtf_file, windows_bed, window_size, step_size, tile_size
        )
        rpkm_path = str(Path(signal_path).parent / "data_rpkm.parquet")
        signal_df = build_signal_df(
            ds,
            samples,
            tss_windows,
            signal_path,
            rpkm_path=rpkm_path,
            chunk_size_rows=chunk_size_rows,
        )

    logger.info(f"Splitting by chromosome: eval={eval_chr}, test={test_chr}")
    train_data = signal_df[
        ~signal_df.index.get_level_values("contig").isin([eval_chr, test_chr])
    ]
    eval_data = signal_df[signal_df.index.get_level_values("contig") == eval_chr]
    test_data = signal_df[signal_df.index.get_level_values("contig") == test_chr]
    logger.info("Reshaping to region x (sample_tss_offset) format for features")
    train_data = reshape_signal_to_region_features(train_data)
    eval_data = reshape_signal_to_region_features(eval_data)
    test_data = reshape_signal_to_region_features(test_data)
    logger.info(
        f"Train: {len(train_data)} regions, Eval: {len(eval_data)}, Test: {len(test_data)} — {train_data.shape[1]} feature columns"
    )

    def _aggregate_target_tiles(
        df: pd.DataFrame, target_col: str, offset_cols: list[str]
    ) -> pd.DataFrame:
        df[target_col] = df[offset_cols].mean(axis=1)
        return df.drop(columns=offset_cols)

    extracted_targets = []
    if target_cols:
        for target_col in target_cols:
            offset_cols = [
                col for col in train_data.columns if col.startswith(target_col + "_")
            ]
            if offset_cols:
                train_data = _aggregate_target_tiles(
                    train_data, target_col, offset_cols
                )
                eval_data = _aggregate_target_tiles(eval_data, target_col, offset_cols)
                test_data = _aggregate_target_tiles(test_data, target_col, offset_cols)
                extracted_targets.append(f"{target_col} ({len(offset_cols)} tiles)")

    if extracted_targets:
        logger.info(f"Targets averaged: {', '.join(extracted_targets)}")
    logger.info(
        f"Final shape — Train: {train_data.shape}, Eval: {eval_data.shape}, Test: {test_data.shape}"
    )

    train_data.to_parquet(dataset_train_path)
    eval_data.to_parquet(dataset_eval_path)
    test_data.to_parquet(dataset_test_path)
    logger.info(f"Saved train/eval/test parquets to {Path(dataset_train_path).parent}")

    if fig_dir and target_cols:
        target_features = [col for col in train_data.columns if col in target_cols]

        if target_features:
            target_data_melted = train_data[target_features].melt(
                var_name="feature", value_name="value"
            )
            target_data_melted["assay"] = (
                target_data_melted["feature"].str.split("-").str[0]
            )
            figure_style()
            plt.figure(figsize=(12, 6))
            sns.violinplot(
                x="feature",
                y="value",
                data=target_data_melted,
                hue="assay",
                palette="tab20",
            )
            plt.xticks(rotation=45, ha="right")
            plt.title("Distribution of targets in training set")
            plt.xlabel("Target")
            plt.ylabel("MinMax-scaled log1p RPKM")
            plt.legend(title="Assay", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{fig_dir}/target_distributions.png")
            plt.close()

    return train_data, eval_data, test_data


def load_data(
    TARGET: str = "MLLN",
    DATASET: str = "data/dataset",
    GTF_FILE: Path = "data/gencode.vM25.annotation.gtf.gz",
    WINDOWS_BED: Path = "data/tss_windows.bed",
    EVAL_CHR: str = "chr8",
    TEST_CHR: str = "chr9",
    FIG_DIR: str = "figures",
    RES_DIR: str = "results",
    MIN_TARGET: int = 1,
    MIN_FEATURES: int = 10,
    EXCLUDE_IPS: list[str] | None = None,
    ASSAY_PREFIXES: list[str] | None = None,
    WINDOW_SIZE: int = 3000,
    STEP_SIZE: int = 100,
    TILE_SIZE: int = 100,
    CHUNK_SIZE_ROWS: int | None = None,
    **_,
):
    DATA_DIR = f"{RES_DIR}/dataset"
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    ds = qn.open_dataset(DATASET)
    samples, target_cols, feature_cols = get_samples(
        ds,
        TARGET,
        exclude_ips=EXCLUDE_IPS,
        prefixes=ASSAY_PREFIXES,
        min_target=MIN_TARGET,
        min_features=MIN_FEATURES,
    )
    train_data, eval_data, test_data = load_or_build_datasets(
        ds=ds,
        samples=samples,
        gtf_file=GTF_FILE,
        windows_bed=WINDOWS_BED,
        signal_path=f"{DATA_DIR}/data_scaled.parquet",
        dataset_train_path=f"{DATA_DIR}/dataset_train.parquet",
        dataset_eval_path=f"{DATA_DIR}/dataset_eval.parquet",
        dataset_test_path=f"{DATA_DIR}/dataset_test.parquet",
        eval_chr=EVAL_CHR,
        test_chr=TEST_CHR,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        tile_size=TILE_SIZE,
        fig_dir=FIG_DIR,
        target_cols=target_cols,
        chunk_size_rows=CHUNK_SIZE_ROWS,
    )

    actual_target_cols = [col for col in target_cols if col in train_data.columns]

    feature_col_patterns = [col + "_" for col in feature_cols]
    actual_feature_cols = [
        col
        for col in train_data.columns
        if any(col.startswith(pat) for pat in feature_col_patterns)
    ]
    train_data, eval_data, test_data = validate_features(
        train_data, eval_data, test_data, actual_feature_cols
    )
    n_tiles = len(actual_feature_cols) // max(len(feature_cols), 1)
    logger.info(
        f"Data loaded — {len(actual_target_cols)} targets, {len(feature_cols)} feature samples × {n_tiles} tiles = {len(actual_feature_cols)} feature columns"
    )
    logger.info(
        f"Train: {len(train_data):,} regions  Eval: {len(eval_data):,}  Test: {len(test_data):,}"
    )
    return (
        ds,
        samples,
        actual_target_cols,
        actual_feature_cols,
        train_data,
        eval_data,
        test_data,
    )


def main():
    from tabnado.utils import parse_params_arg, setup_logger

    params = PipelineParams.from_yaml(parse_params_arg())
    setup_logger(params["RES_DIR"], params["PROJECT"])
    logger.info("========== MAKE DATASET ==========")
    _, _, target_cols, feature_cols, train_data, eval_data, test_data = load_data(
        **vars(params)
    )
    logger.info(
        f"train={train_data.shape}  eval={eval_data.shape}  test={test_data.shape}"
    )
    logger.info(f"features={len(feature_cols)}  targets={target_cols}")


if __name__ == "__main__":
    main()
