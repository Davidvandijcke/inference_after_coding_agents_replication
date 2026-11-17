"""Table 1 Panel D and the power design's in-text numbers.

Per effect size, the share of usable datasets on which the pre-specified test, the agent's own p-value, its
program on new data and the replay test reject at 5%.

Inputs:  data/derived/04_power/, data/agent_runs/power_luna_login/originals.csv and sessions.jsonl.gz,
         data/intermediate/04_power/effect_calibration.json
Outputs: paper/tabs/simulation_panel_d.tex, results/numbers/04_power.json
"""
import csv
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from paper_numbers import write_numbers  # noqa: E402
DER = ROOT / "data" / "derived" / "04_power"
RUNS = ROOT / "data" / "agent_runs" / "power_luna_login"
INTER = ROOT / "data" / "intermediate" / "04_power"
ALPHA = 0.05
SUBSAMPLE_ROWS = 200  # m
ARMS = ("moderate", "strong")
TRUE_EFFECT = ("primary", "structural")  # employment, hours, earnings or training at 12 months
# A null of equal effects for two groups, or a named treatment-by-group coefficient ("treatment x male")
INTERACTION = re.compile(r"effect[^.;]*\bis equal for\b|treatment\s*(?:\u00d7|\bx\b)\s*\w|treatment:\w", re.I)


def rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


originals = rows(RUNS / "originals.csv")
data = {int(r["slot"]): r for r in rows(DER / "datasets.csv")}
bounds = {int(r["slot"]): r for r in rows(DER / "bounds.csv")}
bench = {int(r["slot"]): r for r in rows(DER / "program_benchmark.csv")}
counts = {int(r["slot"]): r for r in rows(DER / "replay_counts.csv")}
(TAIL,), (BODY,) = ({int(r[k]) for r in counts.values()} for k in ("b1_planned", "b0_planned"))
(n_rows,) = {int(r["n"]) for r in data.values()}
with open(INTER / "effect_calibration.json", encoding="utf-8") as f:
    calib = json.load(f)["arms"]
with gzip.open(RUNS / "sessions.jsonl.gz", "rt", encoding="utf-8") as f:
    null_text = {r["slot"]: (r["commitment"] or {}).get("null", "") for r in map(json.loads, f)
                 if r["role"] == "original"}
interaction = {slot for slot, text in null_text.items() if INTERACTION.search(text)}

arm = {}
for a in ARMS:
    use = [r for r in originals if r["arm"] == a and int(r["slot"]) in counts]
    slots = [int(r["slot"]) for r in use]
    replay = {s: float(bounds[s]["upper_one95"]) < ALPHA for s in slots}
    # leads whose null is false: an affected outcome and not an interaction test
    true = [int(r["slot"]) for r in use if r["lead_family"] in TRUE_EFFECT and int(r["slot"]) not in interaction]
    n = len(slots)
    every = [int(r["slot"]) for r in originals if r["arm"] == a]
    population = calib[a]["population"]
    arm[a] = {
        "attempted": len(every), "n": n,
        "effect": population["employed_m12"]["ate"],
        "tau": calib[a]["tau"],
        "power_target": calib[a]["analytic"]["2000"]["power_05"],
        "cohens_d": sum(float(data[s]["cohens_d"]) for s in slots) / n,
        "prespecified": sum(float(data[s]["prespecified_p"]) <= ALPHA for s in slots) / n,
        "agent": sum(float(counts[s]["u"]) <= ALPHA for s in slots) / n,
        "new_data": sum(float(bench[s]["effect_reject_05"]) for s in slots) / n,
        "replay": sum(replay.values()) / n,
        "null_leads": n - len(true),
        "replay_given_true": sum(replay[s] for s in true) / len(true),
        "effect_draws": sorted({int(bench[s]["effect_scored"]) for s in slots}),
        "tail": [int(counts[s]["b1"]) for s in slots],
        "body": [int(counts[s]["b0"]) for s in slots],
    }

null_rates = [float(bench[s]["null_reject_05"]) for s in bench]
spill = [k for k in ("hours_m12", "earnings_m12", "training_m12")
         if calib["moderate"]["population"][k]["ate"] != 0 and calib["strong"]["population"][k]["ate"] != 0]
m, s = arm["moderate"], arm["strong"]

tex = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccc@{}}',
       r'\multicolumn{6}{@{}l}{\rlap{\textit{Panel D (Power): Open-ended agents, true effect on employment at 12 months, '
       rf'$m/n = {SUBSAMPLE_ROWS / n_rows:.2f}$, {TAIL} + {BODY} replays}}}}}} \\', r'\addlinespace[2pt]',
       r' & & \multicolumn{4}{c}{Reject at 5\%} \\', r'\cmidrule(l){3-6}',
       r"Effect (pp) & Datasets & Pre-specified test & Agent's $p$ & Agent's analysis, new data & Replay test \\",
       r'\midrule']
for a in (m, s):
    tex.append(' & '.join([f"+{100 * a['effect']:.1f}", str(a['n'])]
                          + [f"{100 * a[k]:.1f}\\%" for k in ("prespecified", "agent", "new_data", "replay")]) + r' \\')
tex += [r'\bottomrule', r'\end{tabular*}']
(ROOT / "paper" / "tabs").mkdir(parents=True, exist_ok=True)
(ROOT / "paper" / "tabs" / "simulation_panel_d.tex").write_text('\n'.join(tex) + '\n', encoding="utf-8")

Q_DESIGN = "into 200 new simulated randomized control trials from the open-ended agent design above, 100 for each effect size"
Q_SIZES = "by 6.2 and 9.6 percentage points in two sets of simulations"
Q_TARGET = "chosen to target 80\\% and 99\\% power for a pre-specified test"
Q_D = "Cohen's d (effect over outcome standard deviation) of 0.127 and 0.190"
Q_ERROR = ("on 100 datasets per effect size in Panel D; in each set, the original agent's code produced an error on one "
           "dataset")
Q_NULL = "the same programs reject 4.1 to 5.9\\% of new datasets"
Q_RUN = "We run 40+200 replays on each of 99 datasets"
Q_REJECT = "reject 78.8\\% and 67.9\\% of the time under the smaller effect"
Q_TRUE = "rejection frequencies rise to 56.3 and 88.8\\%"
Q_TAU = "with $\\tau=0.28$ on 100 datasets and $\\tau=0.43$ on the other 100"
write_numbers("04_power", [
    ("power_datasets_per_effect", Q_DESIGN, [m["attempted"], s["attempted"]]),
    ("power_effect_moderate_pp", Q_SIZES, 100 * m["effect"]),
    ("power_effect_strong_pp", Q_SIZES, 100 * s["effect"]),
    ("power_target_moderate", Q_TARGET, m["power_target"]),
    ("power_target_strong", Q_TARGET, s["power_target"]),
    ("power_cohens_d_moderate", Q_D, m["cohens_d"]),
    ("power_cohens_d_strong", Q_D, s["cohens_d"]),
    ("power_spillover_outcomes", "The employment effect spills over into three other outcomes at 12 months", len(spill)),
    ("power_errors_moderate", Q_ERROR, m["attempted"] - m["n"]),
    ("power_errors_strong", Q_ERROR, s["attempted"] - s["n"]),
    ("power_new_datasets", "share of 4,000 new simulated datasets, drawn with the same effect",
     min(m["effect_draws"] + s["effect_draws"])),
    ("power_null_reject_min", Q_NULL, min(null_rates)),
    ("power_null_reject_max", Q_NULL, max(null_rates)),
    ("power_tail_replays", Q_RUN, min(m["tail"] + s["tail"])),
    ("power_body_replays", Q_RUN, min(m["body"] + s["body"])),
    ("power_usable_datasets", Q_RUN, [m["n"], s["n"]]),
    ("power_excluded", "excluding the two datasets where the original agents' code produced an error",
     m["attempted"] + s["attempted"] - m["n"] - s["n"]),
    ("power_replay_moderate", "the replay test rejects 49.5\\% of the time for the smaller effect", m["replay"]),
    ("power_replay_strong", "and 87.9\\% of the time for the larger one", s["replay"]),
    ("power_prespecified_moderate", Q_REJECT, m["prespecified"]),
    ("power_new_data_moderate", Q_REJECT, m["new_data"]),
    ("power_agent_p", "The agent's own $p$-value rejects 100\\% of the time", min(m["agent"], s["agent"])),
    ("power_null_leads_moderate", "doing so on 12 out of 99 datasets for the smaller effect", m["null_leads"]),
    ("power_null_leads_strong", "and 1 out of 99 for the larger effect", s["null_leads"]),
    ("power_replay_given_true_moderate", Q_TRUE, m["replay_given_true"]),
    ("power_replay_given_true_strong", Q_TRUE, s["replay_given_true"]),
    ("power_tau_moderate", "with $\\tau=0.28$ on 100 datasets", m["tau"]),
    ("power_tau_strong", "and $\\tau=0.43$ on the other 100", s["tau"]),
    ("power_tau_datasets", Q_TAU, [m["attempted"], s["attempted"]]),
])
