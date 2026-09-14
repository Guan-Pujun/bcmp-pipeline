"""Evaluate four label schemes on the same embeddings and compare rankings."""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scib_metrics.benchmark import Benchmarker


EMBEDDINGS = [
    "X_Uncorrected",
    "X_Seurat_cca",
    "X_Seurat_rpca",
    "X_Harmony",
    "X_fastMNN",
    "X_CSS",
    "X_ComBat",
    "X_LIGER",
    "X_Scanorama",
    "X_scVI",
    "X_scANVI",
]
METRICS = ["BRAS", "iLISI", "KBET", "Graph connectivity", "PCR comparison"]


def rank_methods(scores: pd.Series) -> pd.Series:
    ordered = scores.dropna().sort_index().sort_values(ascending=False, kind="stable")
    ranks = pd.Series(np.nan, index=scores.index)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered.iloc[start] - ordered.iloc[end] <= 0.002:
            end += 1
        ranks.loc[ordered.index[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def main(input_path: str, output_dir: str) -> None:
    output = Path(output_dir)
    adata = ad.read_h5ad(input_path)
    bcmp_labels = pd.read_csv(
        output / "bcmp_labels.tsv", sep="\t", dtype=str, keep_default_na=False,
    ).set_index("cell_id")
    if not adata.obs_names.is_unique or not bcmp_labels.index.equals(adata.obs_names):
        raise ValueError("BCMP labels and embeddings must have identical cell IDs and order.")
    if adata.obs[["batch", "cell_type"]].isna().any().any():
        raise ValueError("batch and cell_type must have no missing values.")

    curated = adata.obs["cell_type"].astype(str).to_numpy()
    labels = {
        "BCMP": bcmp_labels["bcmp_domain"].to_numpy(),
        "curated": curated,
        "constant": np.repeat("Full", adata.n_obs),
    }
    rng = np.random.default_rng()
    for repeat in range(1, 11):
        labels[f"shuffled_{repeat:02d}"] = rng.permutation(curated)

    raw_tables = {}
    scores = pd.DataFrame(index=pd.Index(EMBEDDINGS, name="embedding"))
    for scheme, values in labels.items():
        adata.obs["evaluation_label"] = pd.Categorical(values)
        benchmark = Benchmarker(
            adata,
            batch_key="batch",
            label_key="evaluation_label",
            embedding_obsm_keys=EMBEDDINGS,
            bio_conservation_metrics=None,
            n_jobs=10,
        )
        benchmark.benchmark()
        raw = benchmark.get_results().loc[EMBEDDINGS, METRICS].astype(float)
        raw_tables[scheme] = raw

        span = raw.max() - raw.min()
        scaled = (raw - raw.min()) / span.mask(span == 0, 1)
        scores[scheme] = scaled.mean(axis=1)

    rankings = scores.apply(rank_methods)
    correlations = {}
    for scheme in labels:
        paired_scores = scores[[scheme, "curated"]].dropna()
        paired_ranks = paired_scores.apply(rank_methods)
        correlations[scheme] = paired_ranks.iloc[:, 0].corr(paired_ranks.iloc[:, 1])
    correlations = pd.Series(correlations, name="spearman_rho")
    correlations.loc["shuffled_mean"] = correlations.loc[
        [f"shuffled_{repeat:02d}" for repeat in range(1, 11)]
    ].mean()

    pd.concat(raw_tables, names=["scheme", "embedding"]).to_csv(
        output / "raw_metrics.tsv", sep="\t", mode="x",
    )
    pd.concat(
        {scheme: pd.DataFrame({"score": scores[scheme], "rank": rankings[scheme]})
         for scheme in labels},
        names=["scheme", "embedding"],
    ).to_csv(output / "rankings.tsv", sep="\t", mode="x")
    correlations.rename_axis("scheme").to_csv(
        output / "correlations.tsv", sep="\t", mode="x",
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
