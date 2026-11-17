"""Pre-specified test and realized Cohen's d on each power-design dataset.

Test: OLS of employed_m12 on treatment and site indicators, HC2 standard errors, two-sided t-test.
Cohen's d: treatment-control difference in mean employed_m12 over its standard deviation.

Inputs:  data/raw/pathways_power/power-NN.csv
Outputs: data/derived/04_power/datasets.csv
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "pathways_power"
OUT = ROOT / "data" / "derived" / "04_power"


def prespecified_p(y, treatment, site):
    fe = np.column_stack([(site == s).astype(float) for s in np.unique(site)[1:]])
    X = np.column_stack([np.ones(len(y)), treatment.astype(float), fe])
    XtX_inv = np.linalg.inv(X.T @ X)
    e = y - X @ (XtX_inv @ (X.T @ y))
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    V = XtX_inv @ ((X * (e ** 2 / (1 - h))[:, None]).T @ X) @ XtX_inv  # HC2
    t = (XtX_inv @ (X.T @ y))[1] / np.sqrt(V[1, 1])
    return 2 * stats.t.sf(abs(t), len(y) - X.shape[1])


OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "datasets.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["slot", "n", "prespecified_p", "cohens_d"])
    for path in sorted(RAW.glob("power-*.csv"), key=lambda p: int(p.stem.split("-")[1])):
        d = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
        y, T = d["employed_m12"], d["treatment"]
        effect = y[T == 1].mean() - y[T == 0].mean()
        w.writerow([int(path.stem.split("-")[1]), len(y), f"{prespecified_p(y, T, d['site']):.17g}",
                    f"{effect / y.std():.17g}"])
