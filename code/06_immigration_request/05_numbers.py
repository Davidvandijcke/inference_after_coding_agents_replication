#!/usr/bin/env python3
"""In-text numbers of the realistic request and its m-value.

Inputs:  data/derived/06_immigration_request/correction.csv, mvalue.json,
         data/intermediate/06_immigration_request/original_full_data.json, issp_country_waves.csv, bank_p.csv,
         data/agent_runs/issp_request_luna/original/ANALYSIS_PATHS.jsonl, inputs/CODEBOOK.md
Output:  results/numbers/06_immigration_request.json
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from paper_numbers import write_numbers  # noqa: E402

DER = ROOT / "data/derived/06_immigration_request"
INT = ROOT / "data/intermediate/06_immigration_request"
RUNS = ROOT / "data/agent_runs/issp_request_luna"


def main():
    h = {k: float(v) for k, v in next(csv.DictReader(open(DER / "correction.csv", encoding="utf-8"))).items()}
    m = json.loads((DER / "mvalue.json").read_text(encoding="utf-8"))
    o = json.loads((INT / "original_full_data.json").read_text(encoding="utf-8"))
    cw = list(csv.DictReader(open(INT / "issp_country_waves.csv", encoding="utf-8")))
    bank = list(csv.DictReader(open(INT / "bank_p.csv", encoding="utf-8")))
    respondents, countries = sum(int(r["respondents"]) for r in cw), len({r["iso_country"] for r in cw})
    sizes = {len(r["countries"].split()) for r in bank}
    k = sizes.pop() if len(sizes) == 1 else None

    log_text = (RUNS / "original/ANALYSIS_PATHS.jsonl").read_text(encoding="utf-8")
    log = [json.loads(line) for line in log_text.splitlines() if line.strip()]
    plans = {e.get("path_id"): e for e in log if e.get("event_type") == "plan"}
    estimated = [e for e in log if e.get("event_type") == "result" and e.get("status") == "estimated"]
    formulas = [re.search(r"(\w+) ~ (.+?)\.?$", plans[e["path_id"]]["model"]).groups() for e in estimated[:6]]
    six_items = len({lhs for lhs, _ in formulas}) if len({rhs for _, rhs in formulas}) == 1 else 0
    first_six = [e["estimate_per_1pp_stock"] for e in estimated[:6]]

    # Raw ISSP code of a response = 4 - 3 x (its value in the agents' 0-1 coding), the inverse of the build's recode.
    coding = dict(re.findall(r"^\| (Definitely should(?: not)? be) \| (\d) \|", (RUNS / "inputs/CODEBOOK.md").read_text(
        encoding="utf-8"), flags=re.M))
    raw_should_not = 4 - 3 * int(coding["Definitely should not be"])

    u, lam = h["u"], h["multiplier"]
    briefs, leads = m["sessions_by_brief"], m["leads_original_by_brief"]
    per_brief = [briefs[b] for b in ("pro_team", "anti_team", "nonpartisan")]
    caption = "154,921 respondents in 46 countries and 107 country-waves"
    m_quote = r"Among the 811 analyses considered across all the runs, approximately 12\%"
    breadth = "breadth check, not a result-driven decision to stop or change the model"
    write_numbers("06_immigration_request", [
        ("issp_intro_lambda", "yet it reselects its analysis 2.42 times as often when the analysis is significant",
         round(lam, 6)),
        ("issp_respondents", "which has 154,921 respondents in 46 countries", respondents),
        ("issp_countries", "which has 154,921 respondents in 46 countries", countries),
        ("issp_subsample_countries", "subsamples of 16 out of 46 countries ($m/n=0.35$)", k),
        ("issp_m_over_n", "subsamples of 16 out of 46 countries ($m/n=0.35$)", round(k / countries, 6)),
        ("issp_replays_per_stratum", "replay the agent on $500$ subsamples in each stratum",
         f"{int(h['b1'])}/{int(h['b0'])}"),
        ("issp_outcome_coding", "(1 = definitely should not be [its responsibility], 4 = definitely should be, "
         "rescaled to a $[0,100]$ scale)", f"raw ISSP code of 'definitely should not be': {raw_should_not}"),
        ("issp_original_estimate", "a 10 percentage-point higher immigrant stock was associated with 4.6 points lower "
         "support", round(o["raw_coefficient_per_1pp_stock"] * 1000, 6)),
        ("issp_original_p", "with $p=0.0123$ based on standard errors clustered at the country level", round(u, 6)),
        ("issp_original_outcomes_tried", "it estimated the same regression specification for each of the six "
         "welfare-related outcomes in the data", six_items),
        ("issp_quote_clearest", "because ``it has the clearest negative estimate''",
         "clearest negative estimate" in log_text),
        ("issp_original_six_negative", "among the six specifications it tried, all estimates were negative",
         sum(x < 0 for x in first_six)),
        ("issp_original_analyses_logged", "among the six specifications it tried ... and it did not search further",
         len(estimated)),
        ("issp_quote_breadth", f"``{breadth}''", breadth in log_text),
        ("issp_lambda", r"$\widehat \lambda = 2.42$", round(lam, 6)),
        ("issp_lambda_lower_one95", r"with the 95\% one-sided confidence band excluding $1$",
         round(h["p_lower_one95"] / u, 6)),
        ("issp_original_p_again", "the agent's original $p$-value of $0.0123$", round(u, 6)),
        ("issp_corrected_p", "gets corrected upward to $p=0.0299$", round(h["corrected_p"], 6)),
        ("issp_corrected_p_upper_one95", "the one-sided confidence band for the corrected $p$-value still excludes "
         "$0.05$", round(h["p_upper_one95"], 6)),
        ("issp_tail_matches", "Matching replays: 79 of 500 in the tail and 32 of 500 in the body",
         f"{int(h['x1'])}/{int(h['b1'])}"),
        ("issp_body_matches", "Matching replays: 79 of 500 in the tail and 32 of 500 in the body",
         f"{int(h['x0'])}/{int(h['b0'])}"),
        ("issp_leads_pro", "13/20 reruns of the original agent on the full dataset lead with the same original "
         "analysis", f"{leads['pro_team']}/{briefs['pro_team']}"),
        ("issp_leads_anti", "11/20 runs with an analogous anti-immigration prompt arrive at the same analysis",
         f"{leads['anti_team']}/{briefs['anti_team']}"),
        ("issp_reference_runs", "all 20 runs of the pro- and anti-immigration agents, as well as 20 neutral agents",
         per_brief),
        ("issp_mvalue_analyses", m_quote, m["analyses"]),
        ("issp_mvalue_share", m_quote, round(m["share_at_or_below_t0"], 6)),
        ("issp_original_z", "less than or equal to the original's $z=-2.63$", round(m["t0"], 6)),
        ("issp_mvalue", "their $m$-value is $0.23$", round(m["m"], 6)),
        ("issp_table_respondents", caption, respondents),
        ("issp_table_countries", caption, countries),
        ("issp_table_country_waves", caption, len(cw)),
        ("issp_reference_sessions", "uses 60 full-data GPT-6-Luna sessions", m["sessions"]),
        ("issp_reference_per_prompt", "20 with the original request, 20 with the team described as pushing for lower "
         "immigration", per_brief),
        ("issp_reference_analyses", "Their logs hold 811 estimated analyses", m["analyses"]),
        ("issp_reference_clustered", "nearly all with standard errors clustered by country",
         {"analyses": m["analyses_by_plan_inference"], "plan_events": m["planned_inference"]}),
        ("issp_mvalue_sessions_weighted", "weighting sessions equally gives 0.25",
         round(m["m_sessions_weighted_equally"], 6)),
    ])


if __name__ == "__main__":
    main()
