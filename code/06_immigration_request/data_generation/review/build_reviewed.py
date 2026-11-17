"""Reviewed secondary tally (Claude, 28 October 2026). Raw model_null_match labels stay primary.
Each listed replay's comparison reason was read; the replay chose the original's outcome, exposure, controls,
waves and complete-case rule, and the comparator's only stated differences are wording of assumptions the
original did not declare as part of its model. Categories:
  dependence  - replay declares within-country (or country-wave) dependence / independent clusters;
                the original declared no dependence structure (its clustering lives in the p-value procedure)
  linearity   - replay says 'linear conditional mean', original says 'linear projection'
  redundant   - extra complete-case requirement on cntry_wave (defined from year and iso_country, already required),
                or a reporting rescaling of the contrast
  uncertain   - population wording that could be read as restricting the target to realized units
Usage: python3 review/build_reviewed.py RUN_DIR"""
import glob, json, re, subprocess, sys
from pathlib import Path
run = Path(sys.argv[1]).resolve()
rows = {r["i"]: r for r in map(json.loads, open(run / "review/wording_candidates.jsonl"))}
WORDING = [16,18,20,25,34,38,39,43,55,56,66,74,77,91,98,101,126,145,157,165,167,214,216,221,231,242,248,258,259,270,
           277,288,294,295,296,301,303,314,320,332,347,363,370,376,402,413,420,423,426,468,485,
           96,99,103,123,180,190,223,246,255,272,306,355,397,443,455,465,473,37,445,484,47]
UNCERTAIN = [80, 268, 481, 459]
EXTRA = {"dd475b": "redundant"}  # non-match whose reason does not mention dependence or linearity
res = {}
for p in glob.glob(str(run / "node-results/issp-rep*.results.jsonl")):
    for l in open(p):
        r = json.loads(l); res[r["case_id"]] = r
DEP = re.compile(r"dependen|independen|cluster", re.I)
# linearity wording: the reason contrasts a stated conditional mean with the original's linear projection
CM, PROJ = re.compile(r"conditional[- ]mean|E\[Y|E\(Y|conditional expectation|linear mean", re.I), re.compile(r"projection", re.I)
def categories(t):
    cats = (["dependence"] if DEP.search(t) else []) + (["linearity"] if CM.search(t) and PROJ.search(t) else [])
    return cats or ["redundant"]
out = {}
for i in WORDING + UNCERTAIN:
    cats = categories(rows[i]["reason"])
    out[rows[i]["case_id"]] = {"stratum": rows[i]["stratum"], "raw": False,
                               "reviewed": "uncertain" if i in UNCERTAIN else True, "categories": cats}
for pre, cat in EXTRA.items():
    cid = next(c for c in res if c.startswith(pre))
    out[cid] = {"stratum": res[cid]["meta"]["stratum"], "raw": False, "reviewed": True, "categories": [cat]}
json.dump(out, open(run / "review/reviewed_verdicts.json", "w"), indent=1)
u = json.load(open(run / "banks/7737e9064880f41217902b7cce1562f6.json"))["u"]
raw = {s: sum(r["comparison"]["model_null_match"] is True and r["meta"]["stratum"] == s for r in res.values())
       for s in ("tail", "body")}
def add(keep):
    return {s: raw[s] + sum(1 for v in out.values() if v["stratum"] == s and keep(v)) for s in raw}
versions = {
    "raw labels": raw,
    "+ dependence wording, redundant rules": add(lambda v: v["reviewed"] is True and v["categories"] in (["dependence"], ["redundant"])),
    "+ dependence and linearity wording": add(lambda v: v["reviewed"] is True),
    "+ uncertain as matches": add(lambda v: v["reviewed"] in (True, "uncertain")),
}
R = run.parents[1] / "r-package" / "R"
for name, x in versions.items():
    rc = (f'for (f in list.files("{R}", full.names=TRUE)) source(f); r <- replay_interval_ratio(u={u!r}, x1={x["tail"]}, '
          f'b1=500, x0={x["body"]}, b0=500, gamma=.05); cat(r$estimate$p_value, r$estimate$multiplier, r$p_interval, r$multiplier_interval)')
    v = list(map(float, subprocess.run(["Rscript", "-e", rc], capture_output=True, text=True).stdout.split()))
    print(f"{name:38s} tail {x['tail']:3d}/500  body {x['body']:3d}/500  corrected p {v[0]:.4f} [{v[2]:.4f}, {v[3]:.4f}]  "
          f"lambda {v[1]:.2f} [{v[4]:.2f}, {v[5]:.2f}]")
from collections import Counter
print(Counter((v["stratum"], v["reviewed"], tuple(v["categories"])) for v in out.values()))
