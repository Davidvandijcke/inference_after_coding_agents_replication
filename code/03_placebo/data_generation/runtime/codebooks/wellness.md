# Workplace Health Program evaluation: codebook

The Workplace Health Program is a health-promotion program for the employees of a
large employer. It offers health screenings, an online health risk assessment, and
wellness activities such as exercise, nutrition and stress-management classes, with
small cash rewards for taking part. The program was evaluated in a randomized trial.
Within each of 12 randomization blocks defined by sex, age group and race, about
two-thirds of the participating employees were randomly assigned to be eligible for
the program and the rest to a control group that was not. Outcomes were measured by
online surveys at baseline, before random assignment, and one and two years later.

Each row of `data.csv` is one employee. Every employee completed the baseline survey.
Not every employee completed the follow-up surveys, and a blank cell means that the
value was not recorded.

## Design and baseline variables

| Variable | Description |
|---|---|
| `participant_id` | Employee identifier |
| `block` | Randomization block (1–12); randomization was conducted within block |
| `treatment` | 1 if randomly assigned to the program, 0 if assigned to control |
| `male` | 1 if male |
| `age_group` | 1 = under 37, 2 = 37–49, 3 = 50 or over |
| `white` | 1 if white |

## Outcomes (suffix `_base` for baseline, `_y1` and `_y2` for one and two years later)

All are 1 for yes and 0 for no, except `presenteeism_score`.

| Variable | Description |
|---|---|
| `ever_screened` | Has ever had a screening for cholesterol or blood sugar, or a recent cancer screening |
| `physically_active` | More physically active than most people of the same age |
| `trying_more_active` | Advised to increase physical activity in the last 12 months, or currently trying to |
| `current_smoker` | Smokes cigarettes every day or some days |
| `other_tobacco` | Uses another tobacco product or e-cigarettes |
| `former_smoker` | Has smoked at least 100 cigarettes and no longer smokes |
| `drinker` | Drank alcohol in the last 7 days |
| `heavy_drinker` | Usually 4 or more drinks a day if female, 5 or more if male, on drinking days |
| `chronic_condition` | Has been told by a health professional of at least one chronic condition |
| `health_very_good` | Overall health in the past 4 weeks was excellent or very good |
| `health_not_poor` | Overall health in the past 4 weeks was not poor |
| `physical_problems` | Physical health limited activities or daily work, or at least moderate bodily pain, in the past 4 weeks |
| `lots_of_energy` | Had quite a lot or an extraordinary amount of energy in the past 4 weeks |
| `bad_emotional_health` | Bothered at least moderately by emotional problems in the past 4 weeks |
| `overweight` | Describes own weight as overweight or very overweight |
| `high_bp_chol_glucose` | Describes own blood pressure, cholesterol or glucose as high or very high |
| `sedentary_job` | Job involves less than one hour a day of standing or walking |
| `takes_medication` | Currently takes at least one prescription or over-the-counter medication |
| `physician_or_er_visit` | Went to a doctor's office, clinic or emergency room in the last 6 months |
| `hospital_stay` | Stayed overnight in a hospital in the last 6 months |
| `any_sick_days` | Missed at least one day of work for health reasons in the last 12 months |
| `works_50_plus_hours` | Usually works 50 or more hours a week |
| `very_satisfied_job` | Very satisfied with current job |
| `satisfied_job` | Very or somewhat satisfied with current job |
| `mgmt_values_safety` | Management places very high or some priority on worker health and safety |

These six were not asked at baseline, so they have only `_y1` and `_y2` versions:

| Variable | Description |
|---|---|
| `happier_at_work` | Feels happier at work than last year |
| `presenteeism_score` | Six-item work-limitation score from health problems, 6–30; higher means fewer limitations |
| `very_productive` | Feels very productive at work |
| `promoted` | Given a promotion or more responsibility in the last 12 months |
| `job_search_very_likely` | Very likely to make a genuine effort to find a new employer next year |
| `job_search_likely` | Very or somewhat likely to make a genuine effort to find a new employer next year |

## Analysis convention

The evaluation estimates program effects by ordinary least squares regression of
an outcome on the `treatment` indicator and block indicators (because
randomization was conducted within block), optionally adding baseline covariates.
Standard errors are heteroskedasticity-robust (HC2), and tests are two-sided
t-tests of a zero coefficient on `treatment`. Analyses of participant subgroups
apply the same convention within the subgroup.
