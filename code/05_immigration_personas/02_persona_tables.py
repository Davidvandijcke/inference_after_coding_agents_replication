#!/usr/bin/env python3
"""Tables tab:miao_personas (corrected ISSP file) and tab:miao_personas_original (Miao et al.'s file, with a last row
for the runs of their Supplementary Figure 3c), and the persona design's in-text numbers.

Inputs:  data/agent_runs/persona_*/records.csv, data/derived/05_immigration_personas/miao_sfig3c_z.csv
Outputs: paper/tabs/miao_personas.tex, paper/tabs/miao_personas_original.tex,
         results/numbers/05_immigration_personas.json
"""
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from paper_numbers import write_numbers  # noqa: E402

RUNS = ROOT / "data" / "agent_runs"
DERIVED = ROOT / "data" / "derived" / "05_immigration_personas"
TABS = ROOT / "paper" / "tabs"

# cohort folder, table label, seed label of the resampling interval
MAIN = [("persona_sonnet46_corrected", "Claude Sonnet 4.6", "s46_clean"),
        ("persona_sonnet55_corrected", "Claude Sonnet 5.5", "s55_clean"),
        ("persona_luna_corrected", "GPT-6-Luna", "luna_clean")]
APPENDIX = [("persona_sonnet46_miaofile", "Claude Sonnet 4.6", "s46_miao"),
            ("persona_sonnet55_miaofile", "Claude Sonnet 5.5", "s55_miao"),
            ("persona_luna_miaofile", "GPT-6-Luna", "luna_miao_exact")]
MIAO_RUNS_PER_PERSONA = 20  # Miao et al. ran 20 agents per persona; their figure plots the review-passing ones


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def clopper_pearson(k, n):
    return [float(beta.ppf(.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0]


def gap(anti, pro, seed):
    rng = random.Random(seed)
    draws = sorted(np.mean([rng.choice(anti) for _ in anti]) - np.mean([rng.choice(pro) for _ in pro])
                   for _ in range(4000))
    return {"mean_z_anti": float(np.mean(anti)), "mean_z_pro": float(np.mean(pro)),
            "gap": float(np.mean(anti) - np.mean(pro)), "gap_ci": [float(draws[99]), float(draws[3899])]}


def kept_runs(folder):
    """All runs of a cohort and those with at least two of three reviews PASS."""
    rows = list(csv.DictReader(open(RUNS / folder / "records.csv", encoding="utf-8")))
    odd = {r[k] for r in rows for k in ("review_1", "review_2", "review_3")} - {"PASS", "FAIL", ""}
    if odd:
        sys.exit(f"{folder}: review verdicts other than PASS, FAIL or empty: {sorted(odd)}")
    kept = [r for r in rows if [r["review_1"], r["review_2"], r["review_3"]].count("PASS") >= 2]
    for r in kept:
        r["ame"], r["p"] = num(r["repaired_ame"]), num(r["repaired_ame_pvalue"])
        if r["ame"] is None or r["p"] is None or not 0 <= r["p"] <= 1:
            sys.exit(f"{folder}: kept run {r['session_id']} has no usable repaired estimate or p-value")
    return rows, kept


def cohort_summary(folder, seed):
    rows, kept = kept_runs(folder)
    for r in kept:  # z from the repaired p-value; a run with p = 0 counts as significant but has no z
        r["z"] = math.copysign(norm.isf(r["p"] / 2), r["ame"]) if r["p"] > 0 else None
    anti = [r for r in kept if r["persona"] == "anti"]
    pro = [r for r in kept if r["persona"] == "pro"]
    k_anti = sum(r["p"] < .05 and r["ame"] < 0 for r in anti)
    k_pro = sum(r["p"] < .05 and r["ame"] > 0 for r in pro)
    return {"runs": len(rows), "runs_anti": sum(r["persona"] == "anti" for r in rows),
            "runs_pro": sum(r["persona"] == "pro" for r in rows), "kept_anti": len(anti), "kept_pro": len(pro),
            "sig_anti": k_anti, "sig_pro": k_pro, "share": (k_anti + k_pro) / len(kept),
            "share_ci": clopper_pearson(k_anti + k_pro, len(kept)),
            **gap([r["z"] for r in anti if r["z"] is not None], [r["z"] for r in pro if r["z"] is not None],
                  f"aligned-{seed}")}


def figure_summary():
    rows = list(csv.DictReader(open(DERIVED / "miao_sfig3c_z.csv", encoding="utf-8")))
    anti = [float(r["z"]) for r in rows if r["persona"] == "anti"]
    pro = [float(r["z"]) for r in rows if r["persona"] == "pro"]
    k_anti, k_pro = sum(z < -1.96 for z in anti), sum(z > 1.96 for z in pro)
    n = len(anti) + len(pro)
    return {"runs": 2 * MIAO_RUNS_PER_PERSONA, "kept_anti": len(anti), "kept_pro": len(pro), "sig_anti": k_anti,
            "sig_pro": k_pro, "share": (k_anti + k_pro) / n, "share_ci": clopper_pearson(k_anti + k_pro, n),
            **gap(anti, pro, "aligned-miao-figure")}


def signed(x):
    return f"${x:+.1f}$".replace("+0.0$", "0.0$").replace("-0.0$", "0.0$")


def row(label, g):
    lo, hi = g["share_ci"]
    zlo, zhi = g["gap_ci"]
    return (f"{label} & {g['kept_anti'] + g['kept_pro']}/{g['runs']} & {g['sig_anti']}/{g['kept_anti']} & "
            f"{g['sig_pro']}/{g['kept_pro']} & {100 * g['share']:.0f}\\% [{100 * lo:.0f}, {100 * hi:.0f}] & "
            f"{signed(g['mean_z_anti'])} & {signed(g['mean_z_pro'])} & "
            f"{signed(g['gap'])} [{signed(zlo)}, {signed(zhi)}] \\\\")


HEAD = [r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lrrrcrrc}", r"\toprule",
        r" & & \multicolumn{3}{c}{Significant in the persona\textquoteright s direction} "
        r"& \multicolumn{3}{c}{Mean $z$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Model & Passed & Anti & Pro & Share [95\% CI] & Anti & Pro & $\Delta$ [95\% CI] \\", r"\midrule"]
FOOT = [r"\bottomrule", r"\end{tabular}"]


def main():
    summary = {folder: cohort_summary(folder, seed) for folder, _, seed in MAIN + APPENDIX}
    summary["miao_sfig3c"] = figure_summary()
    TABS.mkdir(parents=True, exist_ok=True)
    (TABS / "miao_personas.tex").write_text(
        "\n".join(HEAD + [row(label, summary[f]) for f, label, _ in MAIN] + FOOT) + "\n", encoding="utf-8")
    (TABS / "miao_personas_original.tex").write_text(
        "\n".join(HEAD + [row(label, summary[f]) for f, label, _ in APPENDIX]
                  + [r"\midrule", row(r"\citet{miao2026agentic}", summary["miao_sfig3c"])] + FOOT) + "\n",
        encoding="utf-8")

    s46, fig = summary["persona_sonnet46_corrected"], summary["miao_sfig3c"]
    s55, luna = summary["persona_sonnet55_corrected"], summary["persona_luna_corrected"]
    s46_miao, s55_miao, luna_miao = (summary[f] for f, _, _ in APPENDIX)
    records = [r for f, _, _ in MAIN + APPENDIX
               for r in csv.DictReader(open(RUNS / f / "records.csv", encoding="utf-8"))]
    reviews = {sum(r[k] in ("PASS", "FAIL") for k in ("review_1", "review_2", "review_3")) for r in records}
    gap_quote = r"compared to a difference of $-4.2$ (95\% CI $[-5.5, -2.9]$) in Supplementary Figure 3c"
    new_quote = (r"the average difference in $z$-statistics reported by the anti- and pro-immigration agents is $0.4$ "
                 r"for Sonnet 5.5 and $0.2$ for GPT-6-Luna")
    write_numbers("05_immigration_personas", [
        ("persona_runs_per_persona", "we ran 20 pro- and 20 anti-immigration agents on the same data",
         min(s46["runs_anti"], s46["runs_pro"])),
        ("persona_runs_per_model", r"\textit{Passed} indicates how many out of the 40 agents' analyses",
         sorted({summary[f]["runs"] for f, _, _ in MAIN})),
        ("persona_s46_anti_sig", "13/18 anti-immigration agents find significant, negative effects",
         f"{s46['sig_anti']}/{s46['kept_anti']}"),
        ("persona_s46_pro_sig", "while 9/19 pro-immigration agents find significant, positive effects",
         f"{s46['sig_pro']}/{s46['kept_pro']}"),
        ("persona_miao_anti_sig", r"compared to 12/17 and 9/17 in \citet{miao2026agentic}",
         f"{fig['sig_anti']}/{fig['kept_anti']}"),
        ("persona_miao_pro_sig", r"compared to 12/17 and 9/17 in \citet{miao2026agentic}",
         f"{fig['sig_pro']}/{fig['kept_pro']}"),
        ("persona_s46_gap", "The average difference in $z$-statistic reported by anti- and pro-immigration agents is "
         "$-4.5$", round(s46["gap"], 4)),
        ("persona_miao_gap", gap_quote, round(fig["gap"], 4)),
        ("persona_miao_gap_ci_lower", gap_quote, round(fig["gap_ci"][0], 4)),
        ("persona_miao_gap_ci_upper", gap_quote, round(fig["gap_ci"][1], 4)),
        ("persona_miao_figure_gap_footnote", "but the results in this figure instead indicate $-4.2$, as read from the "
         "figure's vector graphics", round(fig["gap"], 4)),
        ("persona_newer_models_sig", r"For both Claude Sonnet 5.5 and GPT-6-Luna, \textit{none} of the anti- and "
         "pro-immigration agents report significant results aligned with their persona",
         s55["sig_anti"] + s55["sig_pro"] + luna["sig_anti"] + luna["sig_pro"]),
        ("persona_s55_gap", new_quote, round(s55["gap"], 4)),
        ("persona_luna_gap", new_quote, round(luna["gap"], 4)),
        ("persona_s46_gap_both_files", r"a gap in mean $z$ of $-4.5$, against $-3.8$ on their file and $-4.2$ in their "
         "paper", [round(s46["gap"], 4), round(s46_miao["gap"], 4), round(fig["gap"], 4)]),
        ("persona_newer_runs_their_file", "10 runs per persona for Claude Sonnet 5.5 and 4 for GPT-6-Luna",
         [s55_miao["runs_anti"], s55_miao["runs_pro"], luna_miao["runs_anti"], luna_miao["runs_pro"]]),
        ("persona_repair_model", "a separate, independent reviewer agent (same model as the original)",
         sorted({r["repair_model"] for r in records})),
        ("persona_reviews_per_run", "run three times with the review prompt provided in the replication package",
         sorted(reviews)),
    ])


if __name__ == "__main__":
    main()
