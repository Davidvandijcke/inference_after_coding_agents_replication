"""Table 1 Panel C and the placebo design's in-text numbers.

Per study and pooled: rejection rates of the agent's p-value and of the replay test, tail and body re-selection
rates, mean interval width over originals with a tail match, and the median multiplier with the medians of the
interval ends divided by u over originals matched in both strata.

Inputs:  data/derived/03_placebo/ (per_original.csv, replay_intervals.csv, identity_check.json, base/experiments.json)
Outputs: paper/tabs/simulation_panel_c.tex, results/numbers/03_placebo.json
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from paper_numbers import write_numbers  # noqa: E402
DERIVED = ROOT / "data" / "derived" / "03_placebo"
STUDIES = {"wellness": "Wellness", "teachers": "Teachers", "consumers": "Consumers"}

rows = list(csv.DictReader(open(DERIVED / "per_original.csv", encoding="utf-8")))
iv = {r["original_id"]: r for r in csv.DictReader(open(DERIVED / "replay_intervals.csv", encoding="utf-8"))}
experiments = json.loads((DERIVED / "base" / "experiments.json").read_text(encoding="utf-8"))
for r in rows:
    r.update({k: float(iv[r["original_id"]][k] or "nan") for k in ("multiplier", "lower95", "upper95")})
    r.update({k: int(r[k]) for k in ("x1", "b1", "x0", "b0", "reject_agent_05", "reject_replay_05")})
    r["u"] = float(r["u"])


def stats(rs):
    tail = [r for r in rs if r["x1"] > 0]
    both = [r for r in tail if r["x0"] > 0]
    return dict(n=len(rs), agent=sum(r["reject_agent_05"] for r in rs) / len(rs),
                replay=sum(r["reject_replay_05"] for r in rs) / len(rs),
                tail=sum(r["x1"] for r in rs) / sum(r["b1"] for r in rs),
                body=sum(r["x0"] for r in rs) / sum(r["b0"] for r in rs),
                width=sum(r["upper95"] - r["lower95"] for r in tail) / len(tail),
                lam=statistics.median(r["multiplier"] for r in both),
                lo=statistics.median(r["lower95"] / r["u"] for r in both),
                hi=statistics.median(r["upper95"] / r["u"] for r in both))


def line(label, s):
    return (f"{label} & {100 * s['agent']:.1f}\\% & {100 * s['replay']:.1f}\\% & {100 * s['tail']:.1f}\\% & "
            f"{100 * s['body']:.2f}\\% & {s['width']:.2f} & {s['lam']:.0f} & [{s['lo']:.1f}, {s['hi']:.0f}] \\\\")


by = {e: stats([r for r in rows if r["study"] == e]) for e in STUDIES}
pooled = stats(rows)
m_over_n = [int(0.10 * e["n"] + 0.5) / e["n"] for e in experiments.values()]  # m = round(0.10 n)
budget = f"{max(r['b1'] for r in rows)} + {max(r['b0'] for r in rows)}"
table = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccccc@{}}",
         rf"\multicolumn{{8}}{{@{{}}l}}{{\rlap{{\textit{{Panel C. Open-ended agents, real data with a placebo treatment: {pooled['n']} datasets, $m/n = {round(statistics.median(m_over_n), 2):.2f}$, {budget} replays}}}}}} \\",
         r"\addlinespace[2pt]",
         r" & \multicolumn{2}{c}{Reject at 5\%} & \multicolumn{2}{c}{Re-selected} & & \multicolumn{2}{c}{Multiplier $\widehat\lambda$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){7-8}",
         r"Study & Agent's $p$ & Replay test & Tail & Body & Width & Median & 95\% interval \\",
         r"\midrule"]
table += [line(label, by[e]) for e, label in STUDIES.items()]
table += [r"\midrule", line("All", pooled), r"\end{tabular*}"]
out = ROOT / "paper" / "tabs" / "simulation_panel_c.tex"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(table) + "\n", encoding="utf-8")

identity = json.loads((DERIVED / "identity_check.json").read_text(encoding="utf-8"))
replays = sum(r["b1"] + r["b0"] for r in rows)
agent_k, replay_k = sum(r["reject_agent_05"] for r in rows), sum(r["reject_replay_05"] for r in rows)
replay_by = [by[e]["replay"] for e in STUDIES]
lam_by = [by[e]["lam"] for e in STUDIES]
write_numbers("03_placebo", [
    ("placebo_min_participants", "have at least $2{,}000$ participants", min(e["n"] for e in experiments.values())),
    ("placebo_sessions_checked", "checked the logs of all 24,542 original and replay sessions", len(rows) + replays),
    ("placebo_sessions_checked_recorded", "checked the logs of all 24,542 original and replay sessions",
     sum(identity["records"].values())),
    ("placebo_identity_hits", "for any mention of the study authors, title, or title paraphrases, but did not find any",
     identity["records_with_hits"]),
    ("placebo_datasets_per_study", "For each of the 3 studies, we draw 34 fresh datasets", [by[e]["n"] for e in STUDIES]),
    ("placebo_originals", "each of the 102 full datasets", len(rows)),
    ("placebo_tail_replays", "draw 40 tail and 200 body subsample datasets", min(r["b1"] for r in rows)),
    ("placebo_body_replays", "draw 40 tail and 200 body subsample datasets", min(r["b0"] for r in rows)),
    ("placebo_m_over_n", "relative subsample size of $m/n = 0.1$", m_over_n),
    ("placebo_replay_rate_pooled", r"and 5.9\% for the real data design pooled across the three papers", replay_k / len(rows)),
    ("placebo_fig_datasets", "the real data design (102 datasets from three studies with a placebo treatment, $m/n=0.10$, 40 tail and 200 body replays)",
     len(rows)),
    ("placebo_caption_datasets_each", "34 datasets each (24,440 replays)", [by[e]["n"] for e in STUDIES]),
    ("placebo_caption_replays", "34 datasets each (24,440 replays)", replays),
    ("placebo_agent_rate_pooled", r"report significant results on nearly 100\% of simulations on all three real papers' data",
     agent_k / len(rows)),
    ("placebo_replay_rate_pooled_panel_c", r"near-nominal false rejection rate of $5.9\%$ on the $102$ simulations",
     replay_k / len(rows)),
    ("placebo_n_pooled", r"near-nominal false rejection rate of $5.9\%$ on the $102$ simulations", len(rows)),
    ("placebo_replay_rate_range", r"slightly more variable, between $0.0\%$ and $8.8\%$", [min(replay_by), max(replay_by)]),
    ("placebo_per_study_n", "smaller number of simulations (34) within each paper", [by[e]["n"] for e in STUDIES]),
    ("placebo_median_multiplier_range", r"median multipliers $\widehat \lambda$ of $10$--$22$", [min(lam_by), max(lam_by)]),
])
print(f"Panel C written; replay test rejects {replay_k} of {len(rows)}; agent {agent_k} of {len(rows)}")
