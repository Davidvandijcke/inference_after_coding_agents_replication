"""Code each original's headline from its frozen program and commitment. Reads no p-value.

The outcome and follow-up are the one outcome column named in score.py (one column in
100 of the main run's 100 programs). The type comes from the commitment text by rule and
is checked by hand, reading the commitments only.

family      primary     employed_m12, the outcome with the treatment effect, any
                        population or specification
            structural  hours_m12, earnings_m12 or training_m12, which depend on it
            unaffected  every other outcome, where the effect is exactly zero
            unknown     the rule found no single outcome: read the commitment by hand
type        ate         one treatment coefficient on all participants
            subgroup    one treatment coefficient on an analyst-imposed subgroup
            interaction a contrast of treatment effects between groups

Run this after the originals and before the banks; its output is frozen for the analysis.

Usage: EXP_ROOT=... python code_headlines.py ORIGINALS.jsonl OUT.json
"""
import json
import re
import sys

COLUMN = re.compile(r"\b(employed|hours|earnings|fin_security|food_insecure|debt_collections|distress|health|training)"
                    r"_m(6|12)\b")
INTERACTION_NULL = (r"interaction|×|differential|differs|difference between|equal for|equal across|equals the"
                    r"|is the same for|does not vary|heterogene|moderat|gap between|\bvs\.?\b|versus|β_?T[A-Z]")


def outcome_of(program):
    """The one outcome column the frozen program reads, or None if it reads several."""
    found = sorted(set(COLUMN.findall(program)))
    return (found[0][0], int(found[0][1])) if len(found) == 1 else (None, None)


def type_of(c):
    restricted = not (c.get("sample_restrictions") or "").strip().lower().startswith("no")
    interaction = re.search(INTERACTION_NULL, c.get("null") or "", re.I)
    return "interaction" if interaction else "subgroup" if restricted else "ate"


def family_of(outcome, followup):
    if outcome is None or followup is None:
        return "unknown"
    if followup == 12 and outcome == "employed":
        return "primary"
    if followup == 12 and outcome in ("hours", "earnings", "training"):
        return "structural"
    return "unaffected"


def code(commitment, program):
    o, f = outcome_of(program)
    return {"outcome": o, "followup": f, "family": family_of(o, f), "type": type_of(commitment)}


if __name__ == "__main__":
    import runner
    cases = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    results = runner.load_results()
    out = []
    for c in sorted(cases, key=lambda c: c["meta"]["slot"]):
        cm = results.get(c["case_id"], {}).get("commitment")
        p = runner.EXP / "originals" / c["case_id"] / "workspace" / "score.py"
        rec = {"slot": c["meta"]["slot"], "case_id": c["case_id"], "arm": "strong" if c["meta"]["slot"] % 2 else "moderate"}
        rec.update(code(cm, p.read_text(errors="replace")) if cm and p.exists() else {"family": "no_program"})
        rec["commitment_outcome"] = (cm or {}).get("outcome", "")[:160]  # for the hand check of the type code
        out.append(rec)
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=0))
