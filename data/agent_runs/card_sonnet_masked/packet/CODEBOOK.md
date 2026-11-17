# Store survey variables

Each row contains both survey waves for one store.
Missing source entries are represented by NA. Raw employee counts and closure status codes are retained without recoding.
store_id combines treated_state and survey_sheet with a space; survey_sheet alone is not unique across states.

| Variable | Meaning |
|---|---|
| survey_sheet | Original survey sheet number; not unique across states. |
| chain | Restaurant chain, coded 1 to 4. |
| company_owned | Company-owned store indicator (1 yes, 0 no). |
| treated_state | State: 1 treated state; 0 comparison state. |
| treated_region_1 | Treated-state subregion 1 indicator. |
| treated_region_2 | Treated-state subregion 2 indicator. |
| treated_region_3 | Treated-state subregion 3 indicator. |
| comparison_region_1 | Comparison-state subregion 1 indicator. |
| comparison_region_2 | Comparison-state subregion 2 indicator. |
| treated_region_4 | Treated-state area 4 indicator. |
| callbacks_w1 | Wave 1 call-backs; zero means contact on first call. |
| fulltime_w1 | Wave 1 number of full-time employees, excluding managers and assistant managers. |
| parttime_w1 | Wave 1 number of part-time employees, excluding managers and assistant managers. |
| managers_w1 | Wave 1 number of managers and assistant managers. |
| starting_wage_w1 | Wave 1 starting wage, dollars per hour. |
| months_to_raise_w1 | Wave 1 months to the usual first raise. |
| first_raise_w1 | Wave 1 usual first raise, dollars per hour. |
| hiring_bonus_w1 | Wave 1 cash bounty for new workers indicator. |
| pct_affected_w1 | Wave 1 percent of employees affected by the new minimum wage. |
| meals_w1 | Wave 1 employee meal benefit code. |
| opening_hour_w1 | Wave 1 opening hour. |
| hours_open_w1 | Wave 1 hours open per day. |
| price_drink_w1 | Wave 1 price of a standard drink including tax, dollars. |
| price_side_w1 | Wave 1 price of a standard side item including tax, dollars. |
| price_main_w1 | Wave 1 price of a standard main item including tax, dollars. |
| registers_w1 | Wave 1 number of cash registers. |
| registers_open_w1 | Wave 1 number of registers open at midday. |
| interview_type_w2 | Wave 2 interview type: 1 phone; 2 in person. |
| status_w2 | Wave 2 interview/store status; codes below. |
| interview_day_w2 | Wave 2 interview date, in days since the first wave 2 interview. |
| callbacks_w2 | Wave 2 call-backs; zero means contact on first call. |
| fulltime_w2 | Wave 2 number of full-time employees, excluding managers and assistant managers. |
| parttime_w2 | Wave 2 number of part-time employees, excluding managers and assistant managers. |
| managers_w2 | Wave 2 number of managers and assistant managers. |
| starting_wage_w2 | Wave 2 starting wage, dollars per hour. |
| months_to_raise_w2 | Wave 2 months to the usual first raise. |
| first_raise_w2 | Wave 2 usual first raise, dollars per hour. |
| hiring_program_w2 | Wave 2 special program for new workers indicator. |
| meals_w2 | Wave 2 employee meal benefit code. |
| opening_hour_w2 | Wave 2 opening hour. |
| hours_open_w2 | Wave 2 hours open per day. |
| price_drink_w2 | Wave 2 price of a standard drink including tax, dollars. |
| price_side_w2 | Wave 2 price of a standard side item including tax, dollars. |
| price_main_w2 | Wave 2 price of a standard main item including tax, dollars. |
| registers_w2 | Wave 2 number of cash registers. |
| registers_open_w2 | Wave 2 number of registers open at midday. |

Meal codes (meals_w1, meals_w2): 0 none; 1 free meals; 2 reduced-price meals; 3 both.
Status codes (status_w2): 0 refused wave 2; 1 answered wave 2; 2 temporarily closed; 3 permanently closed; 4 temporarily closed; 5 temporarily closed.
