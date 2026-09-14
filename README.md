# BCMP manuscript pipeline

Scripts to reproduce the paper's comparison of integration-method rankings
using BCMP, curated, constant and shuffled labels.



## Example


First, generate BCMP labels:

```bash
python 01_bcmp.py "<path to counts.h5ad>" "<output directory>"
```

Then, evaluate the integrated embeddings and compare rankings:

```bash
python 02_rankings.py "<path to integrated.h5ad>" "<output directory>"
```
