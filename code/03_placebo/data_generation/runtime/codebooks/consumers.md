# Everyday Consumption Choices evaluation: codebook

Everyday Consumption Choices is a short online information module about how markets
for everyday goods work. It was evaluated in a randomized survey experiment among
adults in the United States recruited through an online panel. Each respondent was
randomly assigned to one of eight consumption topics, and within each topic, which is
the randomization block, half of the respondents were randomly assigned to read the
module before answering the survey questions and half to a control group that answered
them directly. Each respondent was also randomly assigned to one of four social topics
for the valuation questions. Outcomes were measured in the same survey session.

Each row of `data.csv` is one respondent. A blank cell means that the value was not
recorded or does not apply.

## Design and baseline variables

| Variable | Description |
|---|---|
| `participant_id` | Respondent identifier |
| `block` | Randomization block, the assigned consumption topic: 1 = fuel, 2 = meat, 3 = flights, 4 = household energy, 5 = green electricity, 6 = efficient housing, 7 = second-hand clothing, 8 = fair-trade coffee; randomization was conducted within block |
| `treatment` | 1 if randomly assigned to the module, 0 if assigned to control |
| `wtp_topic` | Assigned social topic for the valuation questions: `co2`, `waste`, `animal_welfare` or `low_wages` |
| `gender` | 1 = male, 2 = female, 3 = other or not stated |
| `age` | Age in years |
| `region` | US census region: `northeast`, `midwest`, `south` or `west` |
| `education` | 1 = 12th grade or less, 2 = high school, 3 = some college, 4 = associate degree, 5 = bachelor's degree, 6 = post-graduate degree |
| `income` | Household income, midpoint of the reported bracket, in dollars |
| `politics` | 1 = Republican, 2 = Democratic, 3 = Independent |
| `race_white`, `race_black`, `race_hispanic`, `race_asian`, `race_indigenous` | 1 if the respondent identifies as this group |
| `uses_topic_good` | 1 if the respondent consumes the good in the assigned consumption topic |
| `uses_wtp_good` | 1 if the respondent consumes the good linked to the assigned social topic |

## Outcomes

| Variable | Description |
|---|---|
| `effect_category` | If the respondent permanently cut own consumption of the good, total global consumption would: 1 = fall by more than the cut, 2 = fall by the amount of the cut, 3 = fall by less than the cut, 4 = not change, 5 = actually rise |
| `effect_below_one_to_one` | 1 if `effect_category` is 3 or 4 |
| `effect_per_unit` | Estimated fall in global consumption per unit of own reduction |
| `reason_type` | Coded from the respondent's open-ended explanation: `direct` (own cut lowers the total directly), `market_offset` (markets offset the cut or it is too small to matter), `spillover` (others follow the respondent) or `other` |
| `reason_market_offsets` | 1 if `reason_type` is `market_offset` |
| `wtp_effective` | Dollars the respondent would pay to reduce own contribution to the social topic if the reduction lowered the global total one-for-one |
| `wtp_ineffective` | Dollars the respondent would pay for the same reduction if it left the global total unchanged |
| `wtp_effective_capped`, `wtp_ineffective_capped` | The same, capped at the 95th percentile within `wtp_topic` |
| `wtp_ratio` | `wtp_ineffective` divided by `wtp_effective`, limited to 0–1; blank if `wtp_effective` is 0 |
| `wtp_zero_if_ineffective` | 1 if `wtp_ratio` is 0 |
| `wtp_less_if_ineffective` | 1 if `wtp_ratio` is below 1 |
| `wtp_equal_either_way` | 1 if `wtp_ratio` is 1 |
| `wtp_zero_either_way` | 1 if `wtp_effective` is 0 |

## Analysis convention

The evaluation estimates program effects by ordinary least squares regression of
an outcome on the `treatment` indicator and block indicators (because
randomization was conducted within block), optionally adding baseline covariates.
Standard errors are heteroskedasticity-robust (HC2), and tests are two-sided
t-tests of a zero coefficient on `treatment`. Analyses of participant subgroups
apply the same convention within the subgroup.
