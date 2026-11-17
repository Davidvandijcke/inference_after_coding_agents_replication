#!/usr/bin/env python3
"""Selection multiplier at the 5% level and its one-sided 95% lower bound.

At t = 0.05 the tail holds two pieces, the tail replays and the body replays with u < p <= 0.05, whose match rates
are weighted by the pieces' shares of the reference bank (code/lib/replay.R has no weighted version). The bound
inverts the profile likelihood-ratio test at the 90% level.

Inputs:  data/intermediate/08_card/card-bank-p.csv, data/agent_runs/card_luna/replays.csv.gz,
         data/derived/08_card/original.csv
Output:  data/derived/08_card/threshold.csv
"""
import csv
import gzip
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / 'data' / 'derived' / '08_card'
CHI2_90 = 2.705543454095404
T = 0.05


def corrected(u, a1, a0):
    return u * a1 / (u * a1 + (1 - u) * a0)


def rate_given_t(x, n, t):
    # argmax over [0, 1] of x log a + (n - x) log(1 - a) - t a
    if x == 0:
        return max(0.0, 1 - n / -t) if t < 0 else 0.0
    return 2 * x / ((t + n) + math.sqrt((t + n) ** 2 - 4 * t * x))


def loglik(pieces, rates):
    return sum((x * math.log(a) if x else 0.0) + ((n - x) * math.log(1 - a) if n > x else 0.0)
               for (x, n, _, _), a in zip(pieces, rates))


def bisect(f, lo, hi, steps=200):
    flo = f(lo)
    for _ in range(steps):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (flo > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def profile(pieces, u, h):
    c = [u * (1 - h) * w if tail else -h * (1 - u) * w for _, _, w, tail in pieces]
    rates = lambda l: [rate_given_t(x, n, l * ci) for (x, n, _, _), ci in zip(pieces, c)]
    g = lambda l: sum(ci * a for ci, a in zip(c, rates(l)))
    lo, hi = -1.0, 1.0
    while g(lo) <= 0: lo *= 2
    while g(hi) >= 0: hi *= 2
    return loglik(pieces, rates(bisect(g, lo, hi)))


def main():
    u = next(float(r['value']) for r in csv.DictReader(open(DERIVED / 'original.csv', encoding='utf-8')) if r['name'] == 'u')
    bank = [float(r['p_value']) for r in csv.DictReader(open(ROOT / 'data/intermediate/08_card/card-bank-p.csv', encoding='utf-8'))]
    with gzip.open(ROOT / 'data/agent_runs/card_luna/replays.csv.gz', 'rt', encoding='utf-8') as f:
        replays = [r for r in csv.DictReader(f) if r['review_label'] in ('match', 'nonmatch')]
    tail = [r for r in replays if r['stratum'] == 'tail']
    mid = [r for r in replays if r['stratum'] == 'body' and float(r['bank_p']) <= T]
    high = [r for r in replays if r['stratum'] == 'body' and float(r['bank_p']) > T]
    hits = lambda rows: sum(r['review_label'] == 'match' for r in rows)
    n_tail, n_mid = sum(p <= u for p in bank), sum(u < p <= T for p in bank)
    # (matches, replays, weight within the new stratum, in the new tail)
    pieces = [(hits(tail), len(tail), n_tail / (n_tail + n_mid), True),
              (hits(mid), len(mid), n_mid / (n_tail + n_mid), True), (hits(high), len(high), 1.0, False)]
    rates = [x / n for x, n, _, _ in pieces]
    a1 = sum(w * a for (_, _, w, t), a in zip(pieces, rates) if t)
    a0 = sum(w * a for (_, _, w, t), a in zip(pieces, rates) if not t)
    h = corrected(T, a1, a0)
    deviance = lambda v: 2 * (loglik(pieces, rates) - profile(pieces, T, v)) - CHI2_90
    lower = 0.0 if deviance(1e-9) < 0 else bisect(deviance, 1e-9, h)
    values = {'tail_matches': pieces[0][0], 'tail_replays': pieces[0][1], 'mid_matches': pieces[1][0],
              'mid_replays': pieces[1][1], 'high_matches': pieces[2][0], 'high_replays': pieces[2][1],
              'bank_tail': n_tail, 'bank_mid': n_mid, 'a1': a1, 'a0': a0, 'corrected_p': h,
              'multiplier': h / T, 'multiplier_lower_one95': lower / T}
    with open(DERIVED / 'threshold.csv', 'w', encoding='utf-8') as f:
        f.write('name,value\n' + ''.join(f'{k},{v!r}\n' for k, v in values.items()))


if __name__ == '__main__':
    main()
