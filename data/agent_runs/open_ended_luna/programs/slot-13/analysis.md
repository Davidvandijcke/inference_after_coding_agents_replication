# Pathways to Work randomized evaluation: exploratory finding

## Finding to carry into the next study

Among the 1,005 participants who were not employed at baseline, assignment to
Pathways to Work was associated with a **6.2 percentage point lower rate of
debt in collections at 12 months**. The site-adjusted estimate is **−0.0622**
(HC2 standard error **0.0309**; 95% CI **−0.1228 to −0.0015**; two-sided
`t`-test **p = 0.0447**). This is an intention-to-treat difference in the
binary outcome, in percentage-point units.

The unadjusted-by-baseline-covariates estimate follows the codebook convention:
OLS of the outcome on treatment and site indicators, with HC2 standard errors.
The subgroup was defined using baseline employment, measured before
randomization. In this subgroup the observed debt-collection rates were 43.9%
in control and 37.6% in treatment. These raw rates are descriptive; the
reported estimate adjusts for site.

The pattern differed by baseline employment status. In participants employed
at baseline, the site-adjusted estimate was **+4.1 percentage points** (SE
2.5 points; p = 0.098). The treatment-by-baseline-employment interaction was
**+10.3 percentage points** (SE 4.0 points; 95% CI **+2.5 to +18.1 points**;
p = 0.0094), where a positive interaction means the treatment effect was less
favorable among those initially employed. Adding baseline covariates left the
interaction similar (**+9.8 points**, p = 0.013), while the estimate within
the initially unemployed subgroup remained about −6.0 points but its interval
included zero (p = 0.052).

This result is a candidate for a follow-up study because avoiding debt
collections is a consequential outcome, the initially unemployed group had a
high control rate, and the heterogeneity test suggests baseline employment may
be relevant to who benefits. It is not a confirmatory finding: the within-group
interval is just below zero under the specified site-only model and just
includes zero after baseline adjustment.

## What else was examined

For the full sample, the site-adjusted treatment estimates were small and
imprecise across all nine outcomes at both follow-ups. No overall outcome had a
nominal p-value below 0.34. Selected full-sample results include:

| Outcome | Estimate | HC2 SE | p-value |
|---|---:|---:|---:|
| Debt collections, 12 months | −0.0154 | 0.0204 | 0.451 |
| Employment, 6 months | +0.0129 | 0.0221 | 0.561 |
| Financial security, 12 months | +0.1696 | 0.2317 | 0.464 |
| Earnings, 12 months | −$67.53 | $110.52 | 0.541 |

I also examined treatment effects within baseline employment, sex, parenthood,
education, and site groups, plus treatment-effect heterogeneity across those
groups. Other nominal signals included higher six-month employment among
participants recorded as male (+7.4 points; p = 0.029; interaction p = 0.017),
greater 12-month financial security among those initially unemployed (+0.58
index points; p = 0.049), and lower collections at Site 3 (−9.4 points;
p = 0.024). These are competing exploratory signals, with no evidence that
the employment signal persisted to 12 months and no site-wide heterogeneity
signal for collections.

The subgroup scan covered 252 subgroup-outcome estimates and 90
treatment-effect heterogeneity tests. The selected interaction was the smallest
nominal p-value among the 90 heterogeneity tests, but it does not survive a
Holm correction across those tests (adjusted p approximately 0.846). The
selected within-subgroup p-value also does not survive correction across the
252 subgroup-outcome estimates. The overall outcome tests were likewise all
non-significant. Thus the finding is best used to motivate a preregistered,
adequately powered follow-up that specifies baseline employment and debt
collections in advance, rather than as evidence that the program has already
been shown to reduce collections.

## Reproduction

Run the selected subgroup analysis on any CSV with the same columns:

```sh
python score.py --data data.csv --output score.json
```

The JSON contains the treatment estimate, HC2 standard error, and two-sided
`t`-test p-value for debt collections at 12 months among participants who were
unemployed at baseline. `score.py` runs only this analysis; it does not repeat
the exploratory search.
