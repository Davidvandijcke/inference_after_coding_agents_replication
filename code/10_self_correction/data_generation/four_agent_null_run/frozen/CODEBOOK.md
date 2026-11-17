# Pathways to Work evaluation: codebook

Pathways to Work is an employment-support program for adults seeking work or
better jobs. It offers career coaching, help with job applications, and a small
monthly stipend while participants are enrolled. The program was evaluated in a
randomized trial at four sites. Within each site, half of the enrolled
participants were randomly assigned to the program and half to a control group
that could use existing public services. Outcomes were measured by survey and
administrative records 6 and 12 months after random assignment.

Each row of `data.csv` is one participant. All participants completed both
follow-up surveys, and every variable is observed for every participant.

## Design and baseline variables

| Variable | Description |
|---|---|
| `participant_id` | Participant identifier |
| `site` | Evaluation site (1–4); randomization was conducted within site |
| `treatment` | 1 if randomly assigned to the program, 0 if assigned to control |
| `age` | Age in years at random assignment |
| `female` | 1 if female |
| `parent` | 1 if the participant lives with at least one child under 18 |
| `education` | 1 = less than high school, 2 = high school diploma or GED, 3 = some college, 4 = bachelor's degree or higher |
| `household_size` | Number of people in the household |
| `employed_baseline` | 1 if employed at random assignment |
| `earnings_baseline` | Earnings in the month before random assignment, in dollars |
| `months_unemployed_baseline` | Months since last job at random assignment (0 if employed) |

## Outcomes (each measured at 6 and 12 months; suffix `_m6` or `_m12`)

| Variable | Description |
|---|---|
| `employed` | 1 if employed in the survey week |
| `hours` | Usual weekly hours worked (0 if not employed) |
| `earnings` | Earnings in the previous month, in dollars (0 if not employed) |
| `fin_security` | Financial security index, 0–20; higher means more secure |
| `food_insecure` | 1 if the household reported food insecurity in the past 30 days |
| `debt_collections` | 1 if the participant had a debt in collections in the past 6 months |
| `distress` | Psychological distress score (Kessler-6), 0–24; higher means more distress |
| `health` | Self-rated health, 1 = poor to 5 = excellent |
| `training` | 1 if enrolled in education or job training in the survey week |

## Analysis convention

The evaluation estimates program effects by ordinary least squares regression of
an outcome on the `treatment` indicator and site indicators (because
randomization was conducted within site), optionally adding baseline covariates.
Standard errors are heteroskedasticity-robust (HC2), and tests are two-sided
t-tests of a zero coefficient on `treatment`. Analyses of participant subgroups
apply the same convention within the subgroup.
