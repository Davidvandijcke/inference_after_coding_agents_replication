"""Effect sizes for the power panel: analytic power of the codebook's fixed test on
employed_m12, then Monte Carlo at n = 2,000 and at m = 200. No model calls.

Fixed test: OLS of the outcome on treatment and site indicators, HC2 errors, two-sided
t-test at 5% and 1%. Population quantities come from one very large draw of the
generator, with potential outcomes paired on the seed.

Usage: python calibrate_effect.py OUT.json [--reps 20000] [--workers 8]
"""
import argparse
import json
import multiprocessing as mp

import numpy as np
from scipy import optimize, stats

import generate

BIG_N = 4_000_000
BIG_SEED = 20261030
AFFECTED = ["employed_m12", "hours_m12", "earnings_m12", "training_m12"]
GROUPS = {"all": lambda d: np.ones(len(d["site"]), bool), "women": lambda d: d["female"] == 1,
          "parents": lambda d: d["parent"] == 1, "under30": lambda d: d["age"] < 30,
          "unemployed_base": lambda d: d["employed_baseline"] == 0}


def hc2_t(y, X):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    V = XtX_inv @ ((X * (e ** 2 / (1 - h))[:, None]).T @ X) @ XtX_inv
    return b[1] / np.sqrt(V[1, 1]), len(y) - X.shape[1]


def fixed_p(d, outcome, mask=None):
    """Two-sided p-value of the codebook's fixed test."""
    mask = np.ones(len(d["site"]), bool) if mask is None else mask
    site = d["site"][mask]
    fe = np.column_stack([(site == s).astype(float) for s in np.unique(site)[1:]])
    X = np.column_stack([np.ones(mask.sum()), d["treatment"][mask].astype(float), fe])
    t, df = hc2_t(d[outcome][mask].astype(float), X)
    return 2 * stats.t.sf(abs(t), df)


def within_variance(y, T, site):
    """Mean within-site, within-arm variance, which sets the OLS variance with site indicators."""
    return sum(((site == s) & (T == t)).mean() * y[(site == s) & (T == t)].var()
               for s in (1, 2, 3, 4) for t in (0, 1))


def analytic_power(ncp, n, alpha):
    crit = stats.t.ppf(1 - alpha / 2, n - 5)
    return float(stats.nct.sf(crit, n - 5, ncp) + stats.nct.cdf(-crit, n - 5, ncp))


def population(tau):
    d0 = generate.generate(BIG_SEED, BIG_N, 0.0)
    d1 = generate.generate(BIG_SEED, BIG_N, tau)
    treated = d0["treatment"] == 1
    out = {}
    for k in AFFECTED:
        delta = float((d1[k][treated] - d0[k][treated]).mean())
        s2 = float(within_variance(d1[k].astype(float), d1["treatment"], d1["site"]))
        out[k] = {"ate": delta, "mean_control": float(d1[k][~treated].mean()), "sd": float(d1[k].std()),
                  "within_var": s2}
    # Effect on employment by baseline subgroup, on the probability scale.
    out["employed_m12_ate_by_group"] = {
        g: float((d1["employed_m12"][treated & f(d0)] - d0["employed_m12"][treated & f(d0)]).mean())
        for g, f in GROUPS.items()}
    return out


def se_ncp(pop, n):
    return pop["ate"] / np.sqrt(4 * pop["within_var"] / n)


def solve_tau(target):
    """Latent shift at which the analytic power of the fixed test at n = 2,000 equals target."""
    def gap(tau):
        return analytic_power(se_ncp(population(tau)["employed_m12"], 2000), 2000, 0.05) - target
    return optimize.brentq(gap, 0.1, 0.8, xtol=1e-3)


def one_rep(args):
    tau, seed = args
    d = generate.generate(seed, 2000, tau)
    rng = np.random.default_rng(seed + 1)
    keep = rng.choice(2000, 200, replace=False)
    s = {k: v[keep] for k, v in d.items()}
    row = {f"{k}|{g}": fixed_p(d, k, f(d)) for k in AFFECTED for g, f in GROUPS.items()}
    row["distress_m12|all"] = fixed_p(d, "distress_m12")
    row["employed_m6|all"] = fixed_p(d, "employed_m6")
    row["sub|employed_m12"] = fixed_p(s, "employed_m12")
    row["sub|hours_m12"] = fixed_p(s, "hours_m12")
    return row


def monte_carlo(tau, reps, workers, first_seed):
    with mp.Pool(workers) as pool:
        rows = pool.map(one_rep, [(tau, first_seed + i) for i in range(reps)], chunksize=200)
    out = {}
    for k in rows[0]:
        p = np.array([r[k] for r in rows])
        a5, a1 = float((p <= 0.05).mean()), float((p <= 0.01).mean())
        out[k] = {"reject_05": a5, "reject_01": a1, "mc_se_05": float(np.sqrt(a5 * (1 - a5) / reps)),
                  "median_p": float(np.median(p))}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    report = {"big_n": BIG_N, "reps": a.reps, "exact_solutions": {}}
    for label, target in (("80", 0.80), ("99", 0.99)):
        report["exact_solutions"][label] = solve_tau(target)
    report["arms"] = {}
    for arm, tau in (("null", 0.0), ("moderate", generate.TAU["moderate"]), ("strong", generate.TAU["strong"])):
        pop = population(tau)
        rec = {"tau": tau, "population": pop}
        ncp = {n: float(se_ncp(pop["employed_m12"], n)) for n in (2000, 200)}
        rec["analytic"] = {str(n): {"ncp": ncp[n], "power_05": analytic_power(ncp[n], n, 0.05),
                                    "power_01": analytic_power(ncp[n], n, 0.01)} for n in ncp}
        rec["monte_carlo"] = monte_carlo(tau, a.reps, a.workers, 50_000_000 + int(tau * 100) * 1_000_000)
        report["arms"][arm] = rec
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({arm: {"tau": r["tau"], "ate": r["population"]["employed_m12"]["ate"],
                            "analytic": r["analytic"],
                            "mc_all": r["monte_carlo"]["employed_m12|all"],
                            "mc_sub": r["monte_carlo"]["sub|employed_m12"]}
                      for arm, r in report["arms"].items()}, indent=1))
