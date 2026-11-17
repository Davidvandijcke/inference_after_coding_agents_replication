"""Synthetic randomized evaluation of an employment-support program, power panel.

Treatment is assigned within site, independently of everything else, and enters
one outcome equation: the latent index of employment at 12 months, shifted by tau.
Hours, earnings and training at 12 months depend on that employment status, so
they move with it. Every other effect, at either follow-up, is exactly zero.
With tau = 0 the output is byte for byte that of the frozen main-run generator.

Usage: python generate.py --seed SEED --n N --out data.csv
"""
import argparse
import csv

import numpy as np

SITES = np.array([1, 2, 3, 4])
SITE_SHARE = np.array([0.30, 0.25, 0.25, 0.20])
SITE_LABOUR = np.array([0.25, 0.0, -0.10, -0.30])  # local labour-market strength

# Power panel: latent shift by arm. Even slots moderate, odd slots strong, so the arm is
# balanced over run order. Originals have seed 9,000,000 + slot; fresh references have
# seed 8,000,000,000 + slot * 1,000,000 + j (make_references.fresh_seed).
TAU = {"moderate": 0.28, "strong": 0.43}
PANEL_SEED = 9_000_000
PANEL_SLOTS = 200
FRESH_NAMESPACE = 8_000_000_000


def tau_for_seed(seed):
    if PANEL_SEED <= seed < PANEL_SEED + PANEL_SLOTS:
        slot = seed - PANEL_SEED
    elif seed >= FRESH_NAMESPACE:
        slot = (seed - FRESH_NAMESPACE) // 1_000_000
    else:
        return 0.0
    return TAU["strong"] if slot % 2 else TAU["moderate"]


def generate(seed, n, tau=None):
    tau = tau_for_seed(seed) if tau is None else tau
    rng = np.random.default_rng(seed)
    site = rng.choice(SITES, size=n, p=SITE_SHARE)

    age = np.clip(np.round(rng.normal(37, 11, n)), 18, 64).astype(int)
    female = (rng.random(n) < 0.56).astype(int)
    parent = (rng.random(n) < 1 / (1 + np.exp(-(-0.6 + 0.6 * female + 0.03 * (age - 35))))).astype(int)
    educ = rng.choice([1, 2, 3, 4], size=n, p=[0.16, 0.40, 0.30, 0.14])
    hh_size = np.clip(1 + rng.poisson(0.8, n) + parent * (1 + rng.poisson(0.7, n)), 1, 8)

    # Latent economic capacity and wellbeing; treatment does not enter them.
    labour = SITE_LABOUR[site - 1]
    econ = 0.35 * (educ - 2.4) - 0.015 * np.abs(age - 40) + labour + rng.normal(0, 1, n)
    well = 0.4 * econ + rng.normal(0, 1, n)

    employed_base = (econ + rng.normal(0, 0.9, n) > -0.1).astype(int)
    earnings_base = np.where(employed_base == 1,
                             np.round(np.exp(7.7 + 0.25 * econ + rng.normal(0, 0.45, n)), -1), 0)
    months_unemployed = np.where(employed_base == 1, 0,
                                 np.clip(np.round(rng.exponential(8, n) + 1), 1, 60)).astype(int)

    # Treatment: exactly half of each site, independent of everything above.
    T = np.zeros(n, dtype=int)
    for s in SITES:
        idx = np.flatnonzero(site == s)
        T[rng.choice(idx, size=len(idx) // 2, replace=False)] = 1

    out = {}
    econ_t = {6: econ + rng.normal(0, 0.5, n)}
    econ_t[12] = 0.7 * econ_t[6] + 0.3 * econ + rng.normal(0, 0.5, n)
    well_t = {6: well + rng.normal(0, 0.6, n)}
    well_t[12] = 0.7 * well_t[6] + 0.3 * well + rng.normal(0, 0.6, n)
    for m in (6, 12):
        e, w = econ_t[m], well_t[m]
        shift = tau * T if m == 12 else 0.0  # the one treatment effect; draws no random numbers
        emp = (e + 0.8 * employed_base - 0.4 + shift + rng.normal(0, 0.8, n) > 0).astype(int)
        hours = np.where(emp == 1, np.clip(np.round(rng.normal(31 + 3 * e, 8, n)), 4, 60), 0).astype(int)
        earn = np.where(emp == 1, np.round(np.exp(7.6 + 0.3 * e + 0.25 * np.log1p(earnings_base) / 8
                                                  + rng.normal(0, 0.5, n)), -1), 0)
        fin = np.clip(np.round(10 + 3 * e + 1.5 * w + rng.normal(0, 3, n)), 0, 20).astype(int)
        food = (-(0.7 * e + 0.3 * w) + 0.25 * (hh_size - 2.5) + rng.normal(0, 1, n) > 0.9).astype(int)
        debt = (-(0.6 * e) + rng.normal(0, 1, n) > 0.8).astype(int)
        distress = np.clip(np.round(8 - 2.5 * w - 0.8 * e + rng.normal(0, 3.5, n)), 0, 24).astype(int)
        health = np.clip(np.round(3.2 + 0.45 * w + 0.15 * e - 0.02 * (age - 37) + rng.normal(0, 0.8, n)),
                         1, 5).astype(int)
        training = (-0.6 * emp + 0.2 * (educ - 2.4) - 0.01 * (age - 37) + rng.normal(0, 1, n) > 0.9).astype(int)
        out.update({f"employed_m{m}": emp, f"hours_m{m}": hours, f"earnings_m{m}": earn.astype(int),
                    f"fin_security_m{m}": fin, f"food_insecure_m{m}": food, f"debt_collections_m{m}": debt,
                    f"distress_m{m}": distress, f"health_m{m}": health, f"training_m{m}": training})

    cols = {
        "participant_id": rng.permutation(np.arange(100001, 100001 + n)),
        "site": site, "treatment": T, "age": age, "female": female, "parent": parent,
        "education": educ, "household_size": hh_size, "employed_baseline": employed_base,
        "earnings_baseline": earnings_base.astype(int), "months_unemployed_baseline": months_unemployed,
        **out}
    order = rng.permutation(n)
    return {k: np.asarray(v)[order] for k, v in cols.items()}


def write_csv(cols, path, rows=None):
    names = list(cols)
    idx = range(len(cols[names[0]])) if rows is None else rows
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in idx:
            w.writerow([int(cols[k][i]) for k in names])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    write_csv(generate(a.seed, a.n), a.out)
