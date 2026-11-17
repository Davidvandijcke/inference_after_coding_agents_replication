# Pathways to Work: exploratory randomized evaluation analysis

## Lead finding

At 12 months, assignment to Pathways to Work reduced the probability of being
employed in the survey week by an estimated **5.5 percentage points**. The
site-adjusted OLS estimate was **-0.0550** (HC2 SE **0.0221**; 95% t interval
**[-0.0984, -0.0116]**; two-sided *p* = **0.0130**). The unadjusted observed
employment rates were 43.4% among those assigned to the program and 49.0% among
controls. These are intention-to-treat results for the 2,000 participants in
the supplied trial.

I would lead with this finding because employment is a direct, consequential
outcome for this employment-support program, and the direction at 12 months is
also present in average weekly hours (site-adjusted effect **-1.77 hours**, SE
0.78, *p* = 0.023). There was no employment effect at month 6 (-1.6 points, SE
2.2 points, *p* = 0.470), so the pattern suggests a later difference worth
following up, not a demonstrated explanation for it. The data do not show a
statistically clear earnings difference at month 12 (-$156, SE $107, *p* =
0.144).

## Estimation

For each outcome and follow-up, I fit OLS with the outcome as the response,
treatment assignment and site indicators as predictors. I used HC2
heteroskedasticity-robust standard errors and two-sided t-tests, as specified
in `CODEBOOK.md`. The reported interval uses the residual degrees of freedom.
No observations were missing. `score.py` implements only the selected
12-month employment analysis; it does not repeat the outcome or subgroup
search.

## Why this remains exploratory

The codebook does not identify a primary outcome. I examined 9 outcomes at two
follow-up points (18 outcome-timepoint tests), then looked at selected
baseline-defined groups. The conventional *p* = 0.013 is unadjusted. As a
sensitivity check, a Holm adjustment across just the 18 overall outcome-time
tests gives adjusted *p* = 0.233 for employment at month 12. Accounting for the
additional subgroup exploration would make selection uncertainty greater.
Thus this estimate is a signal to design a focused next study, not a
confirmatory finding.

The result was similar after adding baseline age, gender, parent status,
education, household size, baseline employment, baseline earnings and months
unemployed to the site-adjusted model: **-4.8 percentage points** (HC2 SE 1.8
points, *p* = 0.008). Estimates were also similar across baseline employment
status (-4.5 points among those not employed at baseline; -6.5 points among
those employed; treatment-by-baseline-employment interaction *p* = 0.582),
gender (-4.9 points among men; -5.6 among women; interaction *p* = 0.880), and
parent status (-4.7 points among nonparents; -6.8 among parents; interaction
*p* = 0.633). Site-specific treatment differences showed no detectable
heterogeneity (joint interaction *p* = 0.600). These tests are imprecise
checks; they do not establish that effects are identical across groups or
sites.

## Alternatives considered

The program increased training enrollment by 3.6 percentage points at month 6
(SE 1.5 points, *p* = 0.017), with no detectable difference at month 12
(-1.3 points, SE 1.5 points, *p* = 0.407). Debt in collections was higher by
4.5 percentage points at month 12 (SE 2.0 points, *p* = 0.025), but showed no
clear difference at month 6 (+3.5 points, *p* = 0.086). These are plausible
alternative leads, although each is also one of the many exploratory
comparisons. The lower month-12 employment estimate, its aligned hours result,
and the similar direction across the examined baseline groups make employment
the most direct signal to motivate a next study.

The next study should pre-specify employment and follow-up timing, collect
enough repeated measures to locate when any difference emerges, and measure
work transitions and program engagement to investigate possible pathways.
This trial alone cannot establish why the month-12 employment difference
appeared or whether it will replicate.
