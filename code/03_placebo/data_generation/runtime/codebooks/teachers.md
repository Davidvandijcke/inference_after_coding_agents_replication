# Family and Work Planning evaluation: codebook

Family and Work Planning is a short online information program for employed mothers.
Its centerpiece is a video that explains how employment choices made today affect
income, pension and financial security over the long run. The program was evaluated
in a randomized trial among mothers employed in a public school system, nearly all of
them classroom teachers, whose youngest child was 16 or younger. Within each of five
randomization blocks defined by age group, one-third of the participants were randomly
assigned to watch the video during an online survey and the rest to a control group
that watched a short video on another topic. Outcomes were measured in that survey
right after the video (suffix `_post`) and in a follow-up survey about two months
later (suffix `_fu`).

Each row of `data.csv` is one participant. Every participant answered the baseline part
of the survey. Not every participant finished the survey or answered the follow-up, and
a blank cell means that the value was not recorded.

## Design and baseline variables

| Variable | Description |
|---|---|
| `participant_id` | Participant identifier |
| `block` | Randomization block (1–5, the same as `age_group`); randomization was conducted within block |
| `treatment` | 1 if randomly assigned to the program, 0 if assigned to control |
| `age_group` | 1 = 25–30, 2 = 31–35, 3 = 36–40, 4 = 41–45, 5 = 46–50 |
| `number_children` | Number of children |
| `age_youngest_child` | Age of the youngest child in years |
| `partner_status` | 1 = married, 2 = with a partner, 3 = single |
| `works_full_time` | 1 if currently employed full time |
| `employment_pct_base` | Current employment level, in percent of full time |
| `classroom_teacher` | 1 if a classroom teacher |
| `teaching_diploma` | 1 if holding a teaching diploma |
| `plan_employment_1y_base` | Most likely employment level next school year, in percent of full time |
| `ease_increase_family` | Ease of raising own employment level given family life, 1 = very difficult to 5 = very easy |
| `ease_increase_employer` | Ease of raising own employment level at the employer, 1 = very difficult to 5 = very easy |
| `pension_estimate` | Estimated monthly employer pension of a hypothetical young part-time teacher, in local currency |
| `childcare_cost_view` | Agrees that childcare cost is the most important factor in deciding how much to work after maternity, 1 = completely disagree to 5 = completely agree |
| `child_suffers_view` | Agrees that a child suffers if the mother works, same scale |
| `family_suffers_view` | Agrees that family life suffers if the mother works full time, same scale |
| `father_able_view` | Agrees that fathers can care for children as well as mothers, same scale |
| `external_care_view` | Quality of external childcare compared with parental care, 1 = much worse to 5 = much better |

## Outcomes measured right after the video (suffix `_post`)

| Variable | Description |
|---|---|
| `plan_employment_1y_post` | Desired employment level next school year, in percent of full time |
| `plan_employment_10y_post` | Desired employment level in ten years, in percent of full time |
| `if_trusted_care_post` | Desired employment level if a trusted person could always care for the children |
| `if_no_norms_post` | Desired employment level if friends and family approved of mothers working full time |
| `if_higher_pay_post` | Desired employment level if each additional working day paid 20 percent more |
| `if_partner_flexible_post` | Desired employment level if the partner's employer offered full flexibility |
| `if_partner_engaged_post` | Desired employment level if the partner wanted to spend more time with the children |
| `chose_consultation_post` | 1 if chose a financial consultation voucher rather than a shopping voucher |
| `rank_childcare_post` | Rank of total childcare cost among four factors in the long-run finances of a hypothetical mother who raises her employment, 1 = greatest impact to 4 = least |
| `rank_salary_post` | Rank of salary, same question |
| `rank_pension_post` | Rank of pension, same question |
| `rank_career_post` | Rank of career progress, same question |
| `correct_ranking_post` | 1 if childcare was ranked after salary and pension |
| `felt_angry_post`, `felt_anxious_post`, `felt_hopeful_post`, `felt_discouraged_post`, `felt_happy_post`, `felt_motivated_post` | 1 if the respondent reported feeling this way |
| `wants_pension_overview_post`, `wants_couple_finance_post`, `wants_security_info_post`, `wants_pension_course_post`, `wants_online_tool_post`, `wants_premium_info_post` | 1 if the respondent asked for this information resource: own pension overview, finances in relationships, financial security for women, a course on pension gaps, an online planning tool, information on health insurance premiums |

## Outcomes measured in the follow-up survey (suffix `_fu`)

| Variable | Description |
|---|---|
| `plan_employment_1y_fu`, `plan_employment_10y_fu` | Desired employment level next school year and in ten years |
| `predicted_employment_3y_fu`, `predicted_employment_5y_fu`, `predicted_employment_10y_fu` | Realistic employment level in 3, 5 and 10 years, stated for workforce planning |
| `partner_employment_1y_fu` | Partner's desired employment level next school year |
| `satisfied_family_time_fu`, `satisfied_relationship_fu`, `feels_understood_fu`, `satisfied_chores_fu`, `job_purpose_fu` | 1 if satisfied with this part of the current situation |
| `implemented_level_fu` | Managed to implement the desired employment level: 1 = yes, 2 = no, wanted more, 3 = no, wanted less, 4 = not yet decided, 5 = does not apply |
| `decision_timing_fu` | When the employment level was decided: 1 = not yet, 2 = after the fall holidays, 3 = before the fall holidays, 4 = before the school year |
| `discussed_partner_fu`, `discussed_colleagues_fu`, `discussed_friends_fu`, `discussed_anyone_fu` | 1 if discussed the content of the video with these people |
| `took_action_fu` | 1 if took or plans to take measures related to the video |
| `stressed_fu`, `things_going_well_fu`, `cannot_cope_fu`, `in_control_fu` | How often the respondent felt this way, 1 = seldom, 2 = sometimes, 3 = often |
| `wants_more_children_fu` | 0 = no, 1 = yes, 2 = not yet decided, 3 = declined to answer |
| `rank_childcare_fu`, `rank_salary_fu`, `rank_pension_fu`, `rank_career_fu`, `correct_ranking_fu` | The ranking question above, asked again |

## Analysis convention

The evaluation estimates program effects by ordinary least squares regression of
an outcome on the `treatment` indicator and block indicators (because
randomization was conducted within block), optionally adding baseline covariates.
Standard errors are heteroskedasticity-robust (HC2), and tests are two-sided
t-tests of a zero coefficient on `treatment`. Analyses of participant subgroups
apply the same convention within the subgroup.
