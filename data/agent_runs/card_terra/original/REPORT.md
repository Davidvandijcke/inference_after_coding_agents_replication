# Minimum wage and fast-food employment

## Abstract

Among the specifications explored, the strongest nominal result was a **2.94 full-time-equivalent (FTE) employee increase per store** in New Jersey relative to Pennsylvania, using non-manager employment changes (95% confidence interval 0.32 to 5.57; two-sided Welch p = 0.028; 391 stores). Part-time employees count as half an FTE. This result was selected after examining eight related specifications, so its nominal p-value and interval do not account for selection and should not be read as confirmatory evidence. A simple Bonferroni calculation over eight tests gives 0.227 for the selected Welch p-value; the tests are correlated and include different estimands, so this is an illustrative multiplicity adjustment, not a preplanned family-wise analysis. Results using other plausible definitions and closure assumptions range from p = 0.039 to 0.107. The survey is consistent with a relative employment increase but does not establish that the wage increase caused it.

## Main analysis

Each store is a single observation, with change calculated as wave 2 minus wave 1. Non-manager FTE equals full-time workers plus one half of part-time workers. The treatment contrast is the mean change in New Jersey minus the mean change in Pennsylvania. A permanently closed store is assigned zero wave-2 employment; interviews and permanent closures are eligible, while refusals and temporary closures are excluded. We require both employment components at baseline and, for interviews, follow-up. Welch's standard error and t distribution allow different store-level change variances in the two states.

| State | Stores | Before FTE | After FTE | Mean change | SD of change |
|---|---:|---:|---:|---:|---:|
| New Jersey | 315 | 17.047 | 17.499 | 0.452 | 8.231 |
| Pennsylvania | 76 | 20.013 | 17.523 | -2.490 | 10.800 |

Difference-in-differences = 2.943 FTE; standard error = 1.323; Welch degrees of freedom = 97.04; 95% CI [0.317, 5.568]; p = 0.0284. Six permanently closed stores are included. Compared with all 331 New Jersey and 79 Pennsylvania stores, 16 and 3 respectively are omitted. Selection of non-manager employment was based on its lowest nominal p-value among the analyses, not on a prespecified primary outcome. It is substantively interpretable because the survey separates managers from other employees; it nevertheless omits a component of restaurant employment.

## Complete specification record

The following eight analyses were run on the supplied data during exploration. All use the difference in mean store changes or the coefficient on state in a regression of store change. Standard errors are heteroskedasticity-robust HC2 with a residual-degrees-of-freedom t reference. The final main test uses the same non-manager contrast but Welch's degrees of freedom, yielding the slightly different p-value and confidence interval above. The unadjusted HC2 standard error equals the Welch standard error for the corresponding groups. FTE includes managers unless otherwise specified; closed stores with recorded zero employment are included in the standard definition. Each row uses its own complete cases.

| Specification | N (NJ, PA) | Difference | SE | 95% CI | p |
|---|---:|---:|---:|---:|---:|
| Standard FTE: full-time + 0.5 part-time + managers | 384 (309, 75) | 2.750 | 1.342 | [0.111, 5.389] | 0.0412 |
| Standard FTE, adjusted for chain and company ownership | 384 (309, 75) | 2.785 | 1.344 | [0.141, 5.428] | 0.0390 |
| Standard FTE, interviews only | 378 (304, 74) | 2.722 | 1.340 | [0.087, 5.357] | 0.0429 |
| Standard FTE, all permanent and temporary closures set to zero | 388 (313, 75) | 2.509 | 1.346 | [-0.137, 5.156] | 0.0631 |
| Non-manager FTE | 391 (315, 76) | 2.943 | 1.323 | [0.342, 5.543] | 0.0267 |
| Headcount: full-time + part-time + managers | 384 (309, 75) | 2.412 | 1.493 | [-0.523, 5.346] | 0.1069 |
| FTE with 0.25 weight on part-time | 384 (309, 75) | 2.919 | 1.418 | [0.131, 5.708] | 0.0402 |
| FTE with 0.75 weight on part-time | 384 (309, 75) | 2.581 | 1.369 | [-0.111, 5.272] | 0.0601 |

No other model or outcome was examined. The different part-time weights are sensitivity analyses, not measured hours; the temporary-closure zero assumption describes employment at the survey date and may not reflect permanent job loss. Adjusting for chain and ownership changes the state contrast conditional on those baseline characteristics, not the identifying assumption. The full executable exploration and its original output are in `artifacts/explore.R` and `artifacts/exploration_results.txt`; verification is in `artifacts/verify.R` and `artifacts/verification_results.txt`.

## Interpretation and limitations

A causal interpretation requires that, absent the New Jersey increase, average employment in these New Jersey stores would have changed like that in eastern Pennsylvania. Two waves cannot test this parallel-trends assumption or rule out other contemporaneous state differences, changes in store composition or survey measurement. Stores may share local shocks; the reported uncertainty treats stores as independent within states and does not account for state-level or geographic clustering. There are only two states, so reliable state-clustered inference is not available from this comparison alone. Complete-case analysis can be biased if missingness or closure status relates to employment changes differently across states. The FTE conversion assumes part-time hours equal half of full-time hours; hours were not observed here. Findings apply to the surveyed chains and geography, not necessarily all employers. The outcome-specific and specification-specific selection makes nominal statistical significance especially fragile.

## Reuse

Source `analysis.R` and call `analysis(data)`, where `data` is a data frame with the supplied column schema. It returns the named list `estimate`, `se`, and `p_value`; it does not load the original file. If either state has fewer than two eligible stores, it stops with an informative error. From the workspace root, run `Rscript artifacts/explore.R` and `Rscript artifacts/verify.R` to reproduce the recorded checks.
