# New Jersey minimum-wage increase: fast-food employment DID

## Abstract

Among the defensible specifications explored, the standard full-time-equivalent (FTE) employment measure gives the strongest evidence of a positive relative change: New Jersey stores had an estimated **2.97 additional FTE employees per store relative to eastern Pennsylvania** after the 1992 minimum-wage increase (chain-adjusted two-wave difference-in-differences estimate 2.966, HC2 SE 1.320, 95% CI 0.371 to 5.560, two-sided *p* = 0.025; 391 complete store pairs). The result is stable to chain adjustment and to restricting the sample to stores recorded as interviewed at wave two. It is smaller and not conventionally significant after adjusting for baseline employment, and alternative headcount/log outcomes are also not significant. This is a positive relative change in the observed two-state panel, but causal interpretation depends on untestable parallel trends and store-level standard errors cannot account for state-level shocks with only one treated and one comparison state.

## Data and estimand

The analysis uses only the supplied `data.csv` and packet documentation. There are 410 stores: 331 in New Jersey (state = 1) and 79 in Pennsylvania (state = 0). Wave one preceded New Jersey's April 1992 increase; wave two was collected in November and December 1992. The analysis treats each store as one panel unit and uses the change from wave one to wave two.

The main outcome is the conventional FTE approximation `empft + 0.5 * emppt`, excluding managers as the codebook specifies. The main regression is

`(FTE_wave2 - FTE_wave1) = intercept + NJ + chain fixed effects + error`.

The coefficient on NJ is the two-wave DID: the mean change in New Jersey relative to Pennsylvania, adjusted for differences in change across the four chains. This change-score regression is the two-wave panel DID equivalent of a store fixed-effects regression with an NJ-by-post interaction and chain-specific period shifts. Each store receives equal weight. HC2 heteroskedasticity-robust standard errors and a two-sided t reference distribution with residual degrees of freedom are used.

Employment is complete at both waves for 391 stores (76 Pennsylvania, 315 New Jersey). Six permanently closed stores have zero reported wave-two employment and are retained. The other 385 complete pairs have status 1 (answered wave two). Among all stores, 399 have status 1. Paired employment is missing for 3/79 Pennsylvania stores and 16/331 New Jersey stores. A few records have one employee category missing, so a total cannot be formed without imputation; those stores are omitted from analyses requiring that outcome.

## Main result and alternatives

Estimates below are in employees per store except the log outcome. Confidence intervals use the HC2 standard error and residual-df t critical value. Sample sizes vary where one component of employment is missing.

| Analysis | Complete stores | DID estimate | HC2 SE | 95% CI | p-value |
|---|---:|---:|---:|---:|---:|
| **FTE, chain fixed effects (main)** | 391 | **2.966** | **1.320** | **[0.371, 5.560]** | **0.0252** |
| FTE, unadjusted | 391 | 2.943 | 1.323 | [0.342, 5.543] | 0.0267 |
| FTE, chain fixed effects, baseline FTE added | 391 | 1.522 | 0.907 | [-0.261, 3.305] | 0.0942 |
| FTE, status 1 respondents only, chain fixed effects | 385 | 2.937 | 1.325 | [0.332, 5.541] | 0.0272 |
| Total headcount (FT + PT), unadjusted | 391 | 2.440 | 1.467 | [-0.443, 5.324] | 0.0970 |
| Total headcount, chain fixed effects | 391 | 2.530 | 1.464 | [-0.349, 5.409] | 0.0848 |
| Total headcount, chain fixed effects, baseline headcount added | 391 | 1.400 | 1.093 | [-0.748, 3.548] | 0.2009 |
| Full-time employees, unadjusted | 392 | 3.443 | 1.588 | [0.320, 6.566] | 0.0308 |
| Part-time employees, unadjusted | 396 | -0.911 | 1.532 | [-3.924, 2.101] | 0.5524 |
| Change in log(1 + FTE), unadjusted | 391 | 0.097 | 0.067 | [-0.035, 0.228] | 0.1493 |

In the complete-pair sample, mean FTE changed from 20.01 to 17.52 in Pennsylvania (change -2.49) and from 17.05 to 17.50 in New Jersey (change 0.45). The unadjusted DID is therefore 2.94 FTE. Chain adjustment changes it very little. Adding baseline FTE reduces the DID estimate to 1.52 and the interval includes zero. The split full-time/part-time results suggest that any increase is concentrated in full-time counts, while part-time counts are imprecisely estimated.

The six status-3 permanent closures all have zero wave-two employee counts in the packet and are included as such. Restricting to status-1 respondents removes those six stores and leaves the estimate essentially unchanged (2.94 FTE, *p* = 0.027). Thus closure inclusion does not account for the positive main estimate. A model adding the supplied NJ regional indicators alongside state is not identified: those region indicators are state-specific and collinear with the NJ treatment indicator, so it cannot separate their coefficients.

## Interpretation and limitations

The main estimate is a conditional difference in changes, not a definitive causal effect. Only one state is treated and one state supplies the comparison. HC2 inference treats stores as independent observations and accommodates store-level heteroskedasticity; it cannot estimate uncertainty from state-level shocks or policy assignment. Clustering by state is not informative with two state clusters. The reported p-value and interval are therefore conditional store-level inference, and may overstate precision for a causal claim.

There is only one pre-increase wave, so the parallel-trends assumption cannot be assessed. Other New Jersey-specific changes between survey waves, cross-border spillovers, or different shocks could contribute to the observed gap. The baseline-employment-adjusted model is less compelling statistically and yields a smaller estimate. Multiple outcomes and specifications were examined; the p-values shown are nominal and are not adjusted for this selection. The analysis also uses complete pairs, with 19 of 410 stores omitted from the main outcome because at least one employment component is missing at one wave; similar completeness rates across states do not establish that this missingness is harmless. Finally, employment counts are survey reports, the FTE definition assumes each part-time worker equals one-half a full-time worker, and the sample covers four chains in selected areas rather than the full restaurant sector.

## Reproducibility files

- `analysis.R` defines `analysis(data)` and the HC2 covariance helper; sourcing it does not read data or run an analysis.
- `explore.R` separately inspects the supplied file, records the alternative specifications above, and verifies that `analysis(data)` reproduces the main DID.
