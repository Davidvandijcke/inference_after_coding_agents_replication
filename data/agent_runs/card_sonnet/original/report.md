# Effect of NJ's 1992 minimum-wage increase on fast-food employment

## Abstract
Using a two-wave store survey (331 NJ, 79 PA stores), I estimate a difference-in-differences in
full-time-equivalent (FTE) employment, with Pennsylvania as the control. The pre-specified main analysis
(within-store change in FTE regressed on an NJ indicator; permanent closures counted as zero employment;
HC2 robust SE) gives **+2.75 FTE per store (SE 1.34, p = 0.041, n = 384)**. This is a positive point
estimate: no evidence that employment fell. It is not strong evidence of an increase either. The p-value is
close to 0.05, and it came from one of 31 employment specifications I ran. After Holm correction across
that family, no specification is significant (smallest adjusted p = 0.19). The estimate is also not robust
to adding controls for wave-1 employment level or geography (A4, A5: p = 0.53, 0.18). The defensible
reading is that the data are inconsistent with a sizeable employment loss, not that employment rose.

## How the main analysis was chosen
I chose it from the design before comparing p-values: FTE is the standard employment measure for this
survey. Keeping permanent closures at zero avoids survivorship bias, and the first-difference regression is
the exact two-wave DiD. Some alternatives had smaller p-values (C4: p = 0.006, restricting the control group
to the Philadelphia suburbs; B1: p = 0.007, wage-gap dose-response). I did not lead with them. Picking the
smallest p-value out of about 30 looks would overstate the evidence. C4 also discards 43 of the 79 PA stores (54%),
a choice I would only have made after seeing the data.

## Main result
Mean FTE: PA 23.38 -> 21.1; NJ 20.43 -> 20.9. DiD = +2.75 (SE 1.34), 95% CI approx. [0.1, 5.4]
(t, df = 382). `analysis(data)` in `analysis.R` reproduces this.

## All specifications attempted (employment-effect rows; P1 and P2 are diagnostics)
| ID | Specification | Estimate | SE | p | N |
|---|---|---|---|---|---|
| M | MAIN: dFTE ~ NJ (closures=0) | 2.75 | 1.342 | 0.041 | 384 |
| A1 | Continuing stores only (status2==1; drops closures) | 2.722 | 1.34 | 0.043 | 378 |
| A2 | Drop permanent closures only | 2.722 | 1.34 | 0.043 | 378 |
| A3 | Controls: chain + co_owned | 2.785 | 1.344 | 0.039 | 384 |
| A4 | Controls: chain + co_owned + shore + pa1 | 1.27 | 2.012 | 0.528 | 384 |
| A5 | ANCOVA-type: control for wave-1 FTE | 1.339 | 0.996 | 0.18 | 384 |
| A6 | FTE excluding managers (FT+0.5PT) | 2.943 | 1.323 | 0.027 | 391 |
| A7 | Full-time employees only | 3.443 | 1.588 | 0.031 | 392 |
| A8 | Part-time employees only | -0.911 | 1.532 | 0.552 | 396 |
| A9 | Managers only | -0.088 | 0.165 | 0.596 | 398 |
| A10 | dlog FTE (continuing stores with FTE>0) | 0.078 | 0.049 | 0.112 | 378 |
| A11 | d asinh(FTE) (closures included) | 0.075 | 0.071 | 0.295 | 384 |
| A12 | Headcount (FT+PT+mgr) | 2.412 | 1.492 | 0.107 | 384 |
| A13 | Winsorised dFTE (1/99%) | 2.481 | 1.26 | 0.05 | 384 |
| B1 | Wage-gap dose-response (PA gap=0), coef on gap | 16.359 | 5.988 | 0.007 | 368 |
| B2 | NJ only: wave-1 wage=4.25 vs higher | 1.46 | 1.003 | 0.147 | 293 |
| B3 | NJ low-wage (4.25) stores vs all PA | 3.351 | 1.529 | 0.03 | 166 |
| B4 | NJ high-wage (>=5.00) stores vs all PA (placebo) | -0.011 | 1.636 | 0.995 | 139 |
| C1 | Exclude NJ shore | 2.974 | 1.348 | 0.028 | 352 |
| C2 | NJ north+central vs PA | 2.652 | 1.397 | 0.059 | 295 |
| C3 | NJ south vs PA | 2.999 | 1.452 | 0.041 | 147 |
| C4 | PA1 only vs NJ | 4.334 | 1.555 | 0.006 | 343 |
| D1 | Chain 1 only | 4.39 | 2.549 | 0.087 | 159 |
| D2 | Chain 2 only | -1.594 | 1.074 | 0.142 | 79 |
| D3 | Chain 3 only | 2.488 | 2.168 | 0.254 | 94 |
| D4 | Chain 4 only | 3.417 | 3.171 | 0.286 | 52 |
| E1 | Company-owned only | 2.037 | 1.547 | 0.19 | 134 |
| E2 | Franchised only | 3.132 | 1.893 | 0.099 | 250 |
| F1 | Wave-2 phone interviews only | 2.582 | 1.398 | 0.066 | 353 |
| G1 | Drop |dFTE|>25 | 1.938 | 1.187 | 0.103 | 377 |
| H1 | Bounds: temporarily closed stores coded 0 FTE | 2.509 | 1.346 | 0.063 | 388 |
| P1 | Wave-1 FTE level difference NJ-PA (balance, not an effect) | -2.892 | 1.444 | 0.046 | 398 |
| P2 | First stage: change in starting wage NJ-PA ($) | 0.497 | 0.051 | 0 | 370 |

Of the 31 employment specifications, 11 have p < 0.05 and 17 have p < 0.10. Under the null of no effect, we
would expect about 1.5 of 31 below 0.05 even if they were independent. They are strongly correlated, so
this count is not a calibrated test. Holm-adjusted p-values are all >= 0.19. I did not run any
specification beyond this list. The only code changes were fixing a bug in my covariate matrix (A3-A5
initially failed) and dropping an NJ-subregion control set that was collinear (no results seen).

## Interpretation notes
- Sign pattern: the NJ-PA gap comes from FT employment (+3.4) with PT (-0.9) and managers (~0) flat. The
  log specifications are positive but imprecise (A10 p = 0.11, A11 p = 0.30).
- Dose-response (B1) and the placebo (B4: NJ stores already paying >= $5.00 vs PA, estimate ~0) are
  consistent with a wage-driven pattern, but B2 (within NJ only) is weaker (p = 0.15).
- First stage (P2): NJ starting wages rose about $0.50 more than PA's, so the treatment is real.

## Limitations
- Two waves only: parallel pre-trends cannot be tested. Wave-1 FTE differs by 2.9 between NJ and PA
  (p = 0.046), so the groups were not balanced at baseline. Mean reversion (ANCOVA, A5) shrinks the
  estimate to 1.3.
- 22 NJ and 4 PA stores lack usable employment in one wave and are excluded. 21 of them (17 NJ, 4 PA)
  answered wave 2 but have missing employment items, 4 NJ stores were temporarily closed (status 2/4/5) and
  1 NJ store refused. Coding temporarily closed stores as zero (H1) gives 2.5 (p = 0.06), a bound for
  closure-related attrition only. Item nonresponse is not addressed and could bias results if it
  differs by state.
- Survey employment counts are self-reported by phone/in person and include outliers. Winsorising and
  trimming move the estimate to 1.9-2.5.
- Single labor market/season (Feb-Mar vs Nov-Dec 1992) is a seasonal comparison. It is
  differenced out only if seasonality is common to both states.
- Chain-specific estimates are imprecise and one (KFC) is negative; heterogeneity is not resolved.
- The unit is the store; I did not cluster by area or chain. HC2 SEs ignore possible spatial correlation,
  which would widen the CIs.
- The wage-gap and low-wage-group analyses condition on a post-hoc wave-1 variable, so they are
  descriptive.
