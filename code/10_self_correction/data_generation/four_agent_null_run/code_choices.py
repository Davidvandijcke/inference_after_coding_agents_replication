"""Code each original's chosen analysis from its frozen program and commitment. No model calls.

Fields, as the main-run reviews described the choices:
  outcome, followup     the X_m6 / X_m12 columns score.py reads; "multiple" if it reads several outcomes
  restricted            the commitment's sample_restrictions declares a subgroup or other row filter
  interaction           the test is on a difference in treatment effects between groups
  group                 the baseline variable(s) defining the subgroup or interaction
  conflict              the declaration and the program disagree on the interaction; read by hand

  opening               how the first 400 characters of the report frame a p <= .05 result: stated as a
                        finding, hedged, or disclaimed as not a finding. Added after Sol and Sonnet
                        ran, before Opus: a keyword measure, descriptive and post hoc.

Frozen on Luna's 99 analyses before any other model ran. The hand check on Sol and Sonnet found two
misreadings of their wording, fixed before Opus ran and recorded in the design document: an opening
"none ..." followed by the word "only" was read as a restriction, and the group was read from every
sentence, including those listing what was not restricted. Declaration-program conflicts are read
by hand, as pre-specified, and the verdicts are in INTERACTION_READ. Luna's codes do not change.

Usage: python3 code_choices.py RECORDS.jsonl ORIGINALS_DIR OUT.jsonl
"""
import json
from pathlib import Path
import re
import sys

OUTCOME = re.compile(r"\b(employed|hours|earnings|fin_security|food_insecure|debt_collections|distress|health|training)_m(6|12)\b")
BASELINE = re.compile(r"\b(age|female|education|household_size|employed_baseline|earnings_baseline|months_unemployed_baseline)\b"
                      r"|(?<![.\w])(parent)\b")
GROUP_WORDS = {"female": r"\b(female|women|woman|men|male|sex)\b", "parent": r"\b(parent|parents|child|children)\b",
               "education": r"\b(education|college|high school|degree)\b", "age": r"\b(age|younger|older|(?<!child )under \d\d)\b",
               "site": r"\bsite\s*(=|==|is|\d)", "employed_baseline": r"\b(employed_baseline|employed at|not employed|unemployed at|baseline employment)\b",
               "household_size": r"\bhousehold\b", "earnings_baseline": r"\bearnings_baseline\b",
               "months_unemployed_baseline": r"\bmonths_unemployed_baseline\b"}
NO_RESTRICTION = re.compile(r"^\W*(none|no\b|n/a|nothing|not applicable|all (supplied |enrolled )?(participants|rows)"
                            r"|complete (observations|cases)|(retain|keep) complete)", re.I)
BOTH_GROUPS = re.compile(r"both groups are (included|retained|analy[sz]ed)", re.I)
EQUALS = r"(?:equals?\b(?! (?:to )?(?:zero|0\b))|same|differs?|identical|does not vary|minus)"
INTERACTION_TEXT = re.compile(
    r"interaction|moderat|effect[- ]modif|heterogene|treatment[- ]by[- ]|program[- ]by[- ]|"
    r"(treatment|program|assignment|intervention)[- ](control )?(effect|contrast|difference|association|coefficient|slope)s?\b[^.;]*\b" + EQUALS + "|"
    r"differen\w+ (in|between|of) (the )?[^.;]*(treatment|program|assignment)[^.;]*(effect|contrast|difference)|"
    r"(effect|contrast|gap|slope)s? (is|are) (equal|the same|identical)|"
    r"change in the (treatment|program)[- ]effect|difference[- ]in[- ]differences|equal for|equal across|equal between|"
    r"[\u03b2\u03c4\u03b4]\S*\s*[\u2212-]\s*[\u03b2\u03c4\u03b4]\S*\s*=\s*0", re.I)
INTERACTION_CODE = re.compile(r"(?i:treat\w*\s*\*\s*[\w(\[]|[\w)\]]\s*\*\s*treat\w*|treat\w*:\w|\w:treat\w*|interaction)"
                              r"|(?<![.\w])T\s*\*\s*\w|\w\s*\*\s*T\b(?!\.)")


INTERACTION_READ = {  # hand verdicts on declaration-program conflicts
    ("gpt-6.1-sol", 8): False, ("gpt-6.1-sol", 15): False, ("gpt-6.1-sol", 17): False, ("gpt-6.1-sol", 55): False,  # "not a null about the difference"
    ("gpt-6.1-sol", 59): True,  # effect at 12 months against 6 months
    ("claude-sonnet-5-5", 73): True,  # "the coefficient on treatment x parent"
    ("claude-opus-5-5", 17): True, ("claude-opus-5-5", 73): True,  # "the coefficient on treatment x ..."
}
DISCLAIMED = re.compile(
    r"no (compelling|defensible|credible|robust|convincing|reliable|confirmed|positive|real|clear|finding|result) |no (program )?effect that holds|"
    r"(don'?t|do not|doesn'?t|does not|can'?t|cannot|couldn'?t|could not|won'?t|wouldn'?t|would not|did not|didn'?t) "
    r"(have a|find a|support a|give you|offer a|claim|call |lead with)|"
    r"nothing (from|in|here|that|survives|holds)|(does not|doesn'?t|did not|didn'?t|not) (survive|hold up)|no result survives|"
    r"not (a|as a|as an|an) (finding|confirmed|established|result|discovery|demonstrated|proven)|rather than a (result|finding)|"
    r"not evidence|only a hypothesis|probably noise", re.I)
HEDGED = re.compile(r"\b(hypothes[ie]s|possible|possibly|tentative|may|might|suggestive|exploratory|weak|fragile|promising|signal|preliminary|"
                    r"warrants?|replicate|replication)\b", re.I)


def declares_restriction(text):
    """True when sample_restrictions states a row filter, not merely that there is none."""
    text = (text or "").strip()
    return bool(text) and not BOTH_GROUPS.search(text) and not NO_RESTRICTION.match(text)


def groups_in(text):
    return sorted(k for k, pat in GROUP_WORDS.items() if re.search(pat, text or "", re.I))


def first_sentence(text):
    return re.split(r"(?<=[a-z0-9)])\.\s", (text or "").strip(), maxsplit=1)[0]


def opening_class(report):
    """How the report's opening frames its result."""
    lead = (report or "")[:400].replace("\n", " ")
    if DISCLAIMED.search(lead):
        return "disclaimed"
    return "hedged" if HEDGED.search(lead) else "stated"


def code_one(code, commitment, model=None, slot=None):
    cols = sorted(set(OUTCOME.findall(code)))
    if not cols:
        outcome, followup = None, None
    else:
        names = sorted({c[0] for c in cols})
        outcome = names[0] if len(names) == 1 else "multiple"
        waves = sorted({int(c[1]) for c in cols})
        followup = str(waves[0]) if len(waves) == 1 else "both"
    c = commitment or {}
    null, restrictions = c.get("null", ""), c.get("sample_restrictions", "")
    restricted = declares_restriction(restrictions)
    by_text, by_code = bool(INTERACTION_TEXT.search(null)), bool(INTERACTION_CODE.search(code))
    columns = sorted({a or b for a, b in BASELINE.findall(code)})
    group = groups_in(first_sentence(restrictions)) if restricted else [g for g in groups_in(null) if g in columns or g == "site"]
    if not group and (restricted or by_text) and len(columns) == 1:
        group = columns
    return {"outcome": outcome, "followup": followup, "restricted": restricted, "interaction": INTERACTION_READ.get((model, slot), by_text),
            "interaction_in_program": by_code, "conflict": by_text != by_code, "group": group,
            "baseline_columns": columns, "n_outcome_columns": len(cols)}


def main(records, originals, out):
    rows = []
    for line in open(records):
        r = json.loads(line)
        path = Path(originals) / r["model"] / f"{r['slot']:02d}" / "workspace" / "score.py"
        if not path.exists() or r.get("status") != "complete":
            continue
        rows.append({"model": r["model"], "slot": r["slot"], **code_one(path.read_text(errors="replace"), r.get("commitment"))})
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(len(rows), "coded;", sum(x["conflict"] for x in rows), "conflicts")


if __name__ == "__main__":
    main(*sys.argv[1:4])
