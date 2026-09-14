"""Generate BCMP labels from prepared counts."""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
from bcmp import bcmp


def main(input_path: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True)
    adata = ad.read_h5ad(input_path)

    result = bcmp(adata)
    result.labels.to_csv(output / "bcmp_labels.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
