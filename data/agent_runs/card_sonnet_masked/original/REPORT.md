# Minimum-wage increase and fast-food employment: two-wave DiD

## Abstract
In the pre-specified main analysis, employment (full-time equivalents per store, FTE = managers + full-time + 0.5 × part-time,
with permanently closed stores counted as 0) changed by **+2.75 FTE more in the treated state than in the comparison state**
(SE 1.34, 95% CI 0.12 to 5.38, p = 0.040, n = 384 stores). That is about 13% of the treated-state baseline mean of 20.4.
The sign is positive: there is no evidence of job loss. But the result is **fragile**. It is borderline at the 5% level, it comes from comparing one
treated state with one comparison state, and the comparison state's baseline FTE fell from 23.4 to 21.1 while the treated state's rose from 20.4 to 20.9.
Conditioning on the baseline level shrinks the estimate to 1.34 (p = 0.18), and adding subregion dummies shrinks it to 1.29 (p = 0.54).
The data are best read as showing no detectable employment *loss*, with a possible small gain that cannot be separated from comparison-state
noise or regression to the mean.

**A note on how the main result was chosen.** I was asked to lead with the "most statistically compelling" result among the specifications tried.
I did not do that. Choosing the specification with the smallest p-value after trying about 40 variants makes the reported p-value and CI invalid.
The best-looking row, "comparison region 1 only" (p = 0.005), is a post-hoc restriction of the comparison group. The main
specification was instead fixed on design grounds (standard FTE outcome, closures as zero employment, paired change regression, robust SE) and
`analysis.R` implements exactly that. All other specifications are reported below as sensitivity analyses. I had seen the raw group means before
fixing the specification, but not the regression results.

## Design and estimator
- Unit: store (410 stores: 331 treated state, 79 comparison state). Each row holds both waves.
- Estimand: DiD = Δ FTE(treated) − Δ FTE(comparison), computed as the `treated_state` coefficient from `lm(FTE_w2 − FTE_w1 ~ treated_state)`.
  This is identical to the two-way DiD with store fixed effects.
- Outcome: FTE as defined above. `status_w2 == 3` (permanent closure; 6 stores, 5 treated and 1 comparison) has recorded employment 0 and is kept.
  Dropping those stores would condition on survival, which is a post-treatment outcome. Temporary closures and refusals have missing counts and drop out
  (5 stores) and stores with item-missing employment counts also drop out (n falls from 410 to 384).
- Inference: HC1 robust SE with t reference distribution. Treatment varies only at the state level, so cluster-robust inference by state is impossible.
- Reproduce: `Rscript scripts/explore.R` (writes `output/all_specs.csv`, `output/explore_log.txt`) and `Rscript scripts/verify.R`.

## All analyses attempted (full table in `output/all_specs.csv`)
| Specification | Estimate | SE | p |
|---|---|---|---|
| **MAIN: FTE, closures = 0** | **2.75** | 1.34 | 0.040 |
| Drop permanent closures | 2.72 | 1.34 | 0.042 |
| Excl. implausible interview date (−360) | 2.77 | 1.34 | 0.039 |
| FTE excluding managers | 2.94 | 1.32 | 0.026 |
| Part-time weight 0.4 | 2.82 | 1.36 | 0.038 |
| Headcount (FT+PT+mgr) | 2.41 | 1.49 | 0.106 |
| Full-time only | 3.44 | 1.58 | 0.030 |
| Part-time only | −0.91 | 1.53 | 0.551 |
| log FTE (survivors) | 0.078 | 0.049 | 0.111 |
| asinh FTE (closures = 0) | 0.075 | 0.071 | 0.293 |
| FT share (survivors) | 0.061 | 0.038 | 0.106 |
| + chain FE, company-owned | 2.79 | 1.34 | 0.039 |
| + chain FE, company-owned, subregion dummies | 1.29 | 2.08 | 0.537 |
| ANCOVA (control for baseline level) | 1.34 | 0.99 | 0.178 |
| Long panel, store-clustered (3 variants) | 2.75–2.78 | 1.31–1.34 | 0.035–0.040 |
| Exclude treated_region_4 | 2.97 | 1.34 | 0.028 |
| Comparison subregion 1 only | 4.33 | 1.54 | 0.005 |
| Comparison subregion 2 only | 1.44 | 1.97 | 0.467 |
| By chain 1/2/3/4 | 4.39 / −1.59 / 2.49 / 3.42 | 2.5 / 1.1 / 2.1 / 3.1 | 0.08 / 0.14 / 0.25 / 0.28 |
| Company-owned / franchised | 2.04 / 3.13 | 1.53 / 1.88 | 0.19 / 0.10 |
| Low-wage / higher-wage at wave 1 | 3.12 / 1.94 | 1.72 / 1.93 | 0.07 / 0.32 |
| Main with HC0 / HC2 / HC3 | 2.75 | 1.33–1.35 | 0.040–0.042 |
| Main, SE clustered on chain × state (8 clusters, df = 6) | 2.75 | 1.15 | 0.053 |
| Store-level permutation (not valid for state-level assignment) | 2.75 | – | 0.018 |

The exposure-slope row (`gap`) is in the CSV. It uses a proxy new minimum wage of 12.12 that I inferred from the wave-2 mode, so treat it as exploratory.

Mechanism checks (DiD): starting wage +1.19 $/hr (SE 0.12), so the policy bites. Hours open −0.11 (p = 0.28), price of main item +0.18 (p = 0.07),
drink price −0.02 (p = 0.38), permanent closure rate +0.2 pp (p = 0.86). None of these shows a clear adjustment on other margins.

## Limitations
1. **One treated state and one comparison state.** Common shocks to either state are inseparable from the policy effect. The SEs treat stores as independent
   draws, which understates uncertainty when shocks are state-level. The 8-cluster version (p = 0.053) is itself unreliable with so few clusters.
2. **Two waves, so no pre-trend test.** Parallel trends cannot be checked. Baseline levels differ (23.4 vs 20.4), and the ANCOVA
   estimate (1.34) differs materially from the DiD, which signals sensitivity to the parallel-trends assumption versus mean reversion.
3. **Multiplicity.** About 40 specifications were run, and several are not independent. A few p-values under 0.05 are expected by chance, and the subgroup and
   subregion results should be treated as exploratory.
4. **Missing data and attrition.** Temporary closures and a refusal (5 stores) are dropped, and item nonresponse in employment counts removes about 21 more.
   I did not model this. The estimate holds if missingness is unrelated to treatment-specific employment changes.
5. **Measurement.** Employment is self-reported by phone or in person. Part-time hours are unmeasured, so the 0.5 weight is a convention. `interview_day_w2 = -360` for one store is
   implausible. The main estimate does not use it, and dropping it changes little. The `treated_region_4` indicator overlaps region 3.
6. **Spillovers.** The comparison stores are in the neighbouring state, so cross-border labour or customer flows could contaminate the comparison.
7. **Short horizon** (about 10 months). Longer-run adjustments are not captured.
