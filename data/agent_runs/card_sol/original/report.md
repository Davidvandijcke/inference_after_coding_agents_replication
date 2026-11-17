# New Jersey's 1992 minimum wage and fast-food employment

## Abstract

The most statistically compelling finding among 26 recorded two-wave difference-in-differences analyses was an estimated increase of **2.97 worker full-time equivalents (FTEs) per New Jersey store relative to Pennsylvania**, excluding managers and adjusting for baseline chain and ownership (HC3 SE 1.34; ordinary 95% CI [0.34, 5.60]; two-sided p = 0.0269; 391 paired stores). This result was selected after comparing specifications: its Holm-adjusted p-value is **0.700**, and its 95% Bonferroni simultaneous interval is [-1.21, 7.14]. Thus the search does not provide evidence at the 5% familywise level. The conventional manager-inclusive FTE estimate was +2.75 (p = 0.042); treating temporary closures as zero reduced that estimate to +2.51 (p = 0.065). Headcount and proportional-growth analyses were less conclusive. The estimates favor relative employment growth, but one pre-period and only two states prevent a strong causal conclusion or reliable inference accounting for state-wide shocks.

## Design, target and data handling

New Jersey raised its minimum wage from $4.25 to $5.05 on April 1, 1992; Pennsylvania's minimum stayed unchanged. The packet follows stores in four fast-food chains in NJ and eastern PA in February–March and November–December 1992. Only the supplied data.csv and its DESIGN.md/CODEBOOK.md informed the empirical work; no outside observations were used.

There are 410 unique store IDs (331 NJ, 79 PA). Sheet numbers alone are not unique. Six stores permanently closed (five NJ, one PA), four NJ stores temporarily closed, and one NJ store refused follow-up. Raw fractional employee counts were retained. Missing components were not silently replaced by zero. Permanent closures were assigned zero post employment; the temporary-closure sensitivity also assigns zero to renovation, construction and fire closures. The latter is an assumption about employment at a closed site, not a directly observed count. Refusals and unknown follow-up status remain missing. Survivor-only results condition on status 1 and can be biased by post-treatment selection.

The main endpoint is worker FTE = empft + 0.5 × emppt, excluding managers. Full-time and part-time counts exclude managers in the codebook. Manager-inclusive FTE adds nmgrs; headcount adds full-time, part-time and managers at weight one. The 0.5 weight is an hours proxy, not measured labor hours. Symmetric FTE growth is 2 × (post − pre)/(post + pre), set to zero if both periods equal zero; a permanent closure has growth −2. It is a unitless bounded change, not a log effect or an ordinary percentage change.

Each regression uses equal store weights and complete pairs for its endpoint and covariates. The main analysis retains 391 stores (315 NJ, 76 PA), excluding 19 (16 NJ, 3 PA). Manager-inclusive FTE retains 384 because it also needs manager counts. Therefore cross-endpoint differences reflect both measurement and sample changes. The target is employment at the sampled baseline stores, including observed permanent exits, rather than net employment across the entire industry; new store entry is unobserved.

## Main estimator and inference

For store i, estimate ΔY_i = α + τ NJ_i + γ_chain(i) + δ company_owned_i + ε_i. The coefficient τ is the adjusted NJ–PA difference in within-store changes. Chain and ownership are baseline covariates; their coefficients allow systematic differences in employment trends. The unadjusted version omits them. With two waves this is equivalent to store and wave fixed effects with corresponding chain-by-post and ownership-by-post interactions. No post-treatment covariates, baseline outcome adjustment, outcome-based trimming or geographic indicators collinear with state were used.

The fixed main specification is T08: worker FTE, chain/ownership adjusted, permanent closures zero. Its estimate is 2.967886, HC3 SE 1.336222, and ordinary p-value 0.02692520, using a t reference with 385 residual degrees of freedom. The ordinary interval is [0.340680, 5.595092]. All tests are two-sided. HC3 addresses heteroskedasticity and leverage under independent-store sampling; differencing handles the two observations on a store.

A causal interpretation requires parallel counterfactual trends (conditional on chain and ownership for adjusted models), no consequential spillovers or anticipation, and missingness that does not distort the differential changes. Assignment occurred at the state level. There is only one treated and one control state, so neither store-robust standard errors nor hundreds of stores provide replication of policy assignment. Dependable state-clustered inference is unavailable, and treating stores as independent can overstate precision. The reported intervals and multiplicity corrections are conditional on this working sampling model.

## Recorded search and all alternatives

After inspecting schema and missingness, but before estimating effects, artifacts/analysis_plan.md fixed six endpoints × two adjustment choices × two closure rules (24 tests), plus two manager-inclusive FTE survivor-only tests. The minimum raw two-sided p-value among all 26 selected the main specification, with original grid order breaking ties. No additional specifications were tried after observing the results. This was an exploratory plan written during this task, not a preregistered study.

Ordinary p-values and intervals below do not account for selection. Holm correction covers the full recorded family of 26 under valid marginal tests; dependence between specifications does not invalidate that correction. The Bonferroni simultaneous interval for the selected result also covers this finite search under the working model, at the cost of conservatism. Neither correction addresses unrecorded researcher choices, confounding, correlated state shocks or nonrandom missingness. The selected estimate can also be exaggerated by selection.

| ID | Outcome | Adjusted | Closure rule | NJ / PA n | Estimate | HC3 SE | Ordinary 95% CI | Raw p | Holm p |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| T01 | FTE including managers | No | Permanent closures = 0 | 309 / 75 | 2.750 | 1.351 | [0.095, 5.405] | 0.0424 | 0.7941 |
| T02 | Worker FTE excluding managers | No | Permanent closures = 0 | 315 / 76 | 2.943 | 1.331 | [0.326, 5.559] | 0.0276 | 0.7001 |
| T03 | Headcount including managers | No | Permanent closures = 0 | 309 / 75 | 2.412 | 1.501 | [-0.540, 5.364] | 0.1090 | 1.0000 |
| T04 | Full-time workers | No | Permanent closures = 0 | 316 / 76 | 3.443 | 1.598 | [0.301, 6.585] | 0.0318 | 0.7640 |
| T05 | Part-time workers | No | Permanent closures = 0 | 320 / 76 | -0.911 | 1.541 | [-3.942, 2.119] | 0.5547 | 1.0000 |
| T06 | Symmetric FTE growth | No | Permanent closures = 0 | 309 / 75 | 0.069 | 0.056 | [-0.041, 0.179] | 0.2168 | 1.0000 |
| T07 | FTE including managers | Yes | Permanent closures = 0 | 309 / 75 | 2.785 | 1.358 | [0.115, 5.454] | 0.0410 | 0.7941 |
| T08 **main** | Worker FTE excluding managers | Yes | Permanent closures = 0 | 315 / 76 | 2.968 | 1.336 | [0.341, 5.595] | 0.0269 | 0.7001 |
| T09 | Headcount including managers | Yes | Permanent closures = 0 | 309 / 75 | 2.487 | 1.514 | [-0.489, 5.463] | 0.1012 | 1.0000 |
| T10 | Full-time workers | Yes | Permanent closures = 0 | 316 / 76 | 3.429 | 1.603 | [0.277, 6.581] | 0.0331 | 0.7640 |
| T11 | Part-time workers | Yes | Permanent closures = 0 | 320 / 76 | -0.824 | 1.552 | [-3.875, 2.226] | 0.5955 | 1.0000 |
| T12 | Symmetric FTE growth | Yes | Permanent closures = 0 | 309 / 75 | 0.070 | 0.056 | [-0.039, 0.179] | 0.2097 | 1.0000 |
| T13 | FTE including managers | No | All closures = 0 | 313 / 75 | 2.509 | 1.354 | [-0.153, 5.172] | 0.0646 | 0.8997 |
| T14 | Worker FTE excluding managers | No | All closures = 0 | 319 / 76 | 2.750 | 1.333 | [0.130, 5.371] | 0.0397 | 0.7941 |
| T15 | Headcount including managers | No | All closures = 0 | 313 / 75 | 2.048 | 1.510 | [-0.920, 5.017] | 0.1757 | 1.0000 |
| T16 | Full-time workers | No | All closures = 0 | 320 / 76 | 3.372 | 1.597 | [0.231, 6.512] | 0.0354 | 0.7790 |
| T17 | Part-time workers | No | All closures = 0 | 324 / 76 | -1.150 | 1.544 | [-4.186, 1.886] | 0.4569 | 1.0000 |
| T18 | Symmetric FTE growth | No | All closures = 0 | 313 / 75 | 0.044 | 0.057 | [-0.069, 0.156] | 0.4460 | 1.0000 |
| T19 | FTE including managers | Yes | All closures = 0 | 313 / 75 | 2.524 | 1.360 | [-0.150, 5.199] | 0.0643 | 0.8997 |
| T20 | Worker FTE excluding managers | Yes | All closures = 0 | 319 / 76 | 2.759 | 1.337 | [0.130, 5.388] | 0.0397 | 0.7941 |
| T21 | Headcount including managers | Yes | All closures = 0 | 313 / 75 | 2.087 | 1.522 | [-0.905, 5.079] | 0.1710 | 1.0000 |
| T22 | Full-time workers | Yes | All closures = 0 | 320 / 76 | 3.357 | 1.602 | [0.209, 6.506] | 0.0367 | 0.7790 |
| T23 | Part-time workers | Yes | All closures = 0 | 324 / 76 | -1.095 | 1.556 | [-4.154, 1.965] | 0.4822 | 1.0000 |
| T24 | Symmetric FTE growth | Yes | All closures = 0 | 313 / 75 | 0.042 | 0.057 | [-0.070, 0.154] | 0.4580 | 1.0000 |
| T25 | FTE including managers | No | Respondents only | 304 / 74 | 2.722 | 1.348 | [0.071, 5.373] | 0.0442 | 0.7941 |
| T26 | FTE including managers | Yes | Respondents only | 304 / 74 | 2.780 | 1.360 | [0.106, 5.455] | 0.0417 | 0.7941 |

Positive estimates mean larger employment changes in NJ than PA. Employment levels are FTEs or workers per store; symmetric growth is unitless. Adjusted means chain and baseline company ownership. Ordinary intervals use HC3 t statistics with each model's residual degrees of freedom. The machine-readable table also contains sample sizes, residual df and simultaneous intervals for every test.

For the benchmark manager-inclusive FTE outcome, paired state means reproduce the unadjusted DID exactly:

| State | Paired stores | Before | After | Change |
|---|---:|---:|---:|---:|
| PA | 75 | 23.380 | 21.097 | -2.283 |
| NJ | 309 | 20.431 | 20.897 | 0.467 |

The benchmark DID is +2.750 FTEs. This reflects the difference between NJ's change and PA's change, and does not mean NJ itself grew by 2.750. Full-time worker estimates are positive; part-time estimates are negative and imprecise. The results therefore depend on the employment measure and do not establish a clear increase in the total number of people employed. The headcount and growth intervals include zero under every recorded version.

![All recorded DID estimates with ordinary 95% intervals](artifacts/specification_plot.png)

The highlighted point is the selected model. Intervals in the figure are ordinary, not simultaneous. A standalone PDF is in artifacts/specification_plot.pdf.

## Limitations and interpretation

Only one pre-treatment wave is available, so parallel trends and anticipation cannot be checked. Time-varying state-specific demand, seasonality, regional conditions and other policy changes could explain differential employment changes. Chain/ownership adjustment cannot rule these out. The two-state design is the principal inferential limitation, even before specification selection.

Missing employee counts and temporary closures may be informative. Complete-case analyses do not identify effects for excluded stores without additional assumptions. Counting temporary closures as zero changes the target and the estimates; conditioning on respondents omits exits. Self-reported headcounts, inconsistent measured worker types, and an assumed part-time FTE weight add measurement uncertainty. The four-chain regional frame supplies neither a representative estimate for all NJ businesses nor information on entry, hours actually worked or long-run effects.

The strongest selected result is compatible with positive relative worker FTE growth, but its ordinary significance does not survive correction for the recorded search. Across the alternatives there is no precisely estimated employment reduction; the intervals still allow reductions as well as increases, especially after simultaneous correction. Failure to reject a zero effect is not evidence of no effect.

## Reuse and verification

analysis.R contains definitions only and uses base R. Source it, then call analysis(your_data_frame). It returns exactly a named list with estimate, se and p_value. The procedure stays fixed at T08 for new datasets: it does not choose another endpoint or model using new results. The returned p_value is the ordinary two-sided HC3 t value; the search-adjusted p-value for this packet is reported above and in the results table. No file is read by analysis(data).

```r
source("analysis.R")
d <- read.csv("data.csv")
analysis(d)
```

Reproduce the complete work from the packet directory with Rscript run_analysis.R, Rscript verify_analysis.R, and Rscript make_report.R. Exploration and plotting use base/recommended R; independent verification additionally uses the already installed sandwich package. The report generation requires Cairo PNG support. run_analysis.R verifies that packet file hashes remain unchanged.

Verification matched all 26 covariance matrices to sandwich::vcovHC(type = 'HC3'), reproduced the benchmark by paired means and the main point estimate by a two-wave fixed-effects regression, and checked row reordering, input scaling, subsets, closure coding and side-effect-free sourcing in an empty directory. Fixed-specification leave-one-store-out estimates ranged from 2.418 to 3.311; no store was removed or new test selected. This is an influence diagnostic, not a remedy for state-level dependence. Full checks, session/package versions and source hashes are saved in artifacts/verification.txt and artifacts/inspection.txt.

The complete audit includes analysis_plan.md, all_analyses.csv, exploration.rds (all fitted models), paired_fte_means.csv, influence.csv, inspection.txt and verification.txt in artifacts/. The only estimated specifications are the 26 table rows; verification calculations are separately identified above. Supplied packet files were left unchanged.
