# Store survey variables

Each row contains both survey waves for one store.
Missing source entries are represented by NA. Raw employee counts and closure status codes are retained without recoding.
store_id and sheet are unique anonymous store identifiers.

| Variable | Meaning |
|---|---|
| sheet | Unique anonymous numeric store identifier. |
| chain | Restaurant chain: 1 Burger King; 2 KFC; 3 Roy Rogers; 4 Wendy's. |
| co_owned | Company-owned store indicator (1 yes, 0 no). |
| state | State: 1 New Jersey; 0 Pennsylvania. |
| southj | Southern New Jersey indicator. |
| centralj | Central New Jersey indicator. |
| northj | Northern New Jersey indicator. |
| pa1 | Northeastern Philadelphia suburbs indicator. |
| pa2 | Easton and nearby Pennsylvania area indicator. |
| shore | New Jersey shore location indicator. |
| ncalls | Wave 1 call-backs; zero means contact on first call. |
| empft | Wave 1 number of full-time employees, excluding managers and assistant managers. |
| emppt | Wave 1 number of part-time employees, excluding managers and assistant managers. |
| nmgrs | Wave 1 number of managers and assistant managers. |
| wage_st | Wave 1 starting wage, dollars per hour. |
| inctime | Wave 1 months to the usual first raise. |
| firstinc | Wave 1 usual first raise, dollars per hour. |
| bonus | Wave 1 cash bounty for new workers indicator. |
| pctaff | Wave 1 percent of employees affected by the new minimum wage. |
| meals | Wave 1 employee meal benefit code. |
| open | Wave 1 opening hour. |
| hrsopen | Wave 1 hours open per day. |
| psoda | Wave 1 medium soda price including tax, dollars. |
| pfry | Wave 1 small fries price including tax, dollars. |
| pentree | Wave 1 entree price including tax, dollars. |
| nregs | Wave 1 number of cash registers. |
| nregs11 | Wave 1 number of registers open at 11 am. |
| type2 | Wave 2 interview type: 1 phone; 2 in person. |
| status2 | Wave 2 interview/store status; codes below. |
| date2 | Wave 2 interview date in numeric MMDDYY format. |
| ncalls2 | Wave 2 call-backs; zero means contact on first call. |
| empft2 | Wave 2 number of full-time employees, excluding managers and assistant managers. |
| emppt2 | Wave 2 number of part-time employees, excluding managers and assistant managers. |
| nmgrs2 | Wave 2 number of managers and assistant managers. |
| wage_st2 | Wave 2 starting wage, dollars per hour. |
| inctime2 | Wave 2 months to the usual first raise. |
| firstin2 | Wave 2 usual first raise, dollars per hour. |
| special2 | Wave 2 special program for new workers indicator. |
| meals2 | Wave 2 employee meal benefit code. |
| open2r | Wave 2 opening hour. |
| hrsopen2 | Wave 2 hours open per day. |
| psoda2 | Wave 2 medium soda price including tax, dollars. |
| pfry2 | Wave 2 small fries price including tax, dollars. |
| pentree2 | Wave 2 entree price including tax, dollars. |
| nregs2 | Wave 2 number of cash registers. |
| nregs112 | Wave 2 number of registers open at 11 am. |

Meal codes (meals, meals2): 0 none; 1 free meals; 2 reduced-price meals; 3 both.
Status codes (status2): 0 refused wave 2; 1 answered wave 2; 2 closed for renovations; 3 permanently closed; 4 closed for highway construction; 5 closed due to a mall fire.
