# Immigration and support for the welfare state

## Lead finding — path `p_d40f62`

In the pooled ISSP data, higher country immigrant stock is associated with lower support for government responsibility to provide for the unemployed. A 10 percentage-point higher immigrant stock corresponds to **4.59 points less support** on a 0–100 rescaling of the survey answer (SE 1.75; *p* = .012; nominal 95% CI −8.13 to −1.06). Positive values on this scale mean more support. This is a cross-national association, not a causal effect.

**Question:** “On the whole, do you think it should or should not be the government's responsibility to provide a decent standard of living for the unemployed?”  
**Null:** the OLS coefficient on `migstock` is zero in `unemployed ~ migstock + age + female + employed + C(year)`.

![Core item-specific OLS estimates and country-clustered confidence intervals](lead_chart.png)

The selected complete-case sample has **124,450 respondents, 92 country-waves and 38 countries**, spanning 1985, 1990, 1996, 2006 and 2016. The model is unweighted respondent-level OLS with age, sex, employment and wave indicators; uncertainty is clustered by country (38 clusters, 37 reference degrees of freedom). Immigrant stock is measured in percentage points of population. The chart retains separate item outcomes; it does not construct a composite or a common effect across different exposure measures.

## Alternatives and caveats

The six core outcomes all had negative point estimates, but unemployment support was clearest; the jobs item was borderline (*p* = .052), and the other four intervals included zero. The unemployment estimate stayed similar with World Bank stock (−4.96 points), UN stock (−4.39) and OECD stock (−6.63), each per +10 percentage points. The distinct annual net-migration measure was less precise (−4.01 points per +1 percentage point per year; *p* = .304). Adding GDP, unemployment and Gini controls or education reduced precision (*p* = .060 and .059); a 2016-only estimate was also imprecise (*p* = .136). With country fixed effects, the estimate was −0.30 points (*p* = .901), so the pooled association does not persist when time-invariant country differences are absorbed.

Treat the lead as suggestive: it is driven by pooled between-country variation, may reflect omitted national differences, and does not establish immigration changing attitudes. The lead *p*-value is nominal after examining six outcomes (Holm-adjusted *p* across those six is about .074); the reported interval is nominal too. The file has no survey weights, pools some group samples without population reweighting, and country-wave coverage and indicator availability differ. Stock series are interpolated and the combined measure switches source by year.
