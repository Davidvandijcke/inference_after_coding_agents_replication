#!/usr/bin/env python3
"""Builds the masked packet of Table 8's "masked data" row from the unmasked packet (neutral column and place
names, interview dates as days, dollar amounts times 2.4, the research prompt's first sentences rewritten) and
checks that it is byte-identical to the packet the masked cohort received.

Inputs:  data/agent_runs/card_luna/packet/, data/agent_runs/card_sonnet_masked/packet/
Outputs: data/derived/09_card_agents/masked_packet/
"""
import csv
import filecmp
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data' / 'agent_runs' / 'card_luna' / 'packet'
SHIPPED = ROOT / 'data' / 'agent_runs' / 'card_sonnet_masked' / 'packet'
OUT = ROOT / 'data' / 'derived' / '09_card_agents' / 'masked_packet'
FILES = ('data.csv', 'CODEBOOK.md', 'DESIGN.md', 'research-prompt.txt', 'commitment_prompt.txt', 'comparison_prompt.txt')

RENAME = {
    "sheet": "survey_sheet", "chain": "chain", "co_owned": "company_owned", "state": "treated_state",
    "southj": "treated_region_1", "centralj": "treated_region_2", "northj": "treated_region_3",
    "pa1": "comparison_region_1", "pa2": "comparison_region_2", "shore": "treated_region_4",
    "ncalls": "callbacks_w1", "empft": "fulltime_w1", "emppt": "parttime_w1", "nmgrs": "managers_w1",
    "wage_st": "starting_wage_w1", "inctime": "months_to_raise_w1", "firstinc": "first_raise_w1",
    "bonus": "hiring_bonus_w1", "pctaff": "pct_affected_w1", "meals": "meals_w1", "open": "opening_hour_w1",
    "hrsopen": "hours_open_w1", "psoda": "price_drink_w1", "pfry": "price_side_w1", "pentree": "price_main_w1",
    "nregs": "registers_w1", "nregs11": "registers_open_w1", "type2": "interview_type_w2",
    "status2": "status_w2", "date2": "interview_day_w2", "ncalls2": "callbacks_w2", "empft2": "fulltime_w2",
    "emppt2": "parttime_w2", "nmgrs2": "managers_w2", "wage_st2": "starting_wage_w2",
    "inctime2": "months_to_raise_w2", "firstin2": "first_raise_w2", "special2": "hiring_program_w2",
    "meals2": "meals_w2", "open2r": "opening_hour_w2", "hrsopen2": "hours_open_w2", "psoda2": "price_drink_w2",
    "pfry2": "price_side_w2", "pentree2": "price_main_w2", "nregs2": "registers_w2",
    "nregs112": "registers_open_w2", "store_id": "store_id",
}
DOLLARS = {"wage_st", "firstinc", "psoda", "pfry", "pentree", "wage_st2", "firstin2", "psoda2", "pfry2", "pentree2"}
FACTOR = 2.4

DESIGN = """# Survey setting

In 2019, one state (the treated state) increased its hourly minimum wage by about 19 percent; a neighboring state (the comparison state) kept its minimum wage unchanged.
The survey frame covers stores from four fast-food chains in the treated state and in part of the comparison state.
The first wave was collected in early 2019, shortly before the increase, and the second wave in late 2019.
Each row holds both waves for one store, including any missing fields and follow-up status. Supplied subsets retain the same underlying store frame and geographic, chain and survey-wave definitions.
"""

MEANING = {
    "survey_sheet": "Original survey sheet number; not unique across states.",
    "chain": "Restaurant chain, coded 1 to 4.",
    "company_owned": "Company-owned store indicator (1 yes, 0 no).",
    "treated_state": "State: 1 treated state; 0 comparison state.",
    "treated_region_1": "Treated-state subregion 1 indicator.",
    "treated_region_2": "Treated-state subregion 2 indicator.",
    "treated_region_3": "Treated-state subregion 3 indicator.",
    "comparison_region_1": "Comparison-state subregion 1 indicator.",
    "comparison_region_2": "Comparison-state subregion 2 indicator.",
    "treated_region_4": "Treated-state area 4 indicator.",
}
WAVE = {
    "callbacks": "call-backs; zero means contact on first call.",
    "fulltime": "number of full-time employees, excluding managers and assistant managers.",
    "parttime": "number of part-time employees, excluding managers and assistant managers.",
    "managers": "number of managers and assistant managers.",
    "starting_wage": "starting wage, dollars per hour.",
    "months_to_raise": "months to the usual first raise.",
    "first_raise": "usual first raise, dollars per hour.",
    "hiring_bonus": "cash bounty for new workers indicator.",
    "pct_affected": "percent of employees affected by the new minimum wage.",
    "meals": "employee meal benefit code.",
    "opening_hour": "opening hour.",
    "hours_open": "hours open per day.",
    "price_drink": "price of a standard drink including tax, dollars.",
    "price_side": "price of a standard side item including tax, dollars.",
    "price_main": "price of a standard main item including tax, dollars.",
    "registers": "number of cash registers.",
    "registers_open": "number of registers open at midday.",
    "interview_type": "interview type: 1 phone; 2 in person.",
    "status": "interview/store status; codes below.",
    "interview_day": "interview date, in days since the first wave 2 interview.",
    "hiring_program": "special program for new workers indicator.",
}
CODES = """
Meal codes (meals_w1, meals_w2): 0 none; 1 free meals; 2 reduced-price meals; 3 both.
Status codes (status_w2): 0 refused wave 2; 1 answered wave 2; 2 temporarily closed; 3 permanently closed; 4 temporarily closed; 5 temporarily closed.
"""

OLD_P1 = ("I am studying whether New Jersey's 1992 minimum-wage increase affected employment in fast-food restaurants. "
          "These data follow stores in New Jersey and eastern Pennsylvania before and after the increase; Pennsylvania "
          "did not change its minimum wage over this period.")
NEW_P1 = ("I am studying whether a 2019 state minimum-wage increase affected employment in fast-food restaurants. "
          "These data follow stores in the state that raised its minimum wage and in a neighboring comparison state "
          "before and after the increase; the comparison state did not change its minimum wage over this period.")


def check(ok, message):
    if not ok:
        sys.exit(message)


def day_index(v, first):
    if v in ("NA", ""):
        return "NA"
    return str((date(1900 + int(v[4:6]), int(v[0:2]), int(v[2:4])) - first).days)


def dollars(v):
    return v if v in ("NA", "") else f"{round(float(v) * FACTOR + 1e-9, 2):.2f}"


def codebook(cols):
    lines = ["# Store survey variables", "",
             "Each row contains both survey waves for one store.",
             "Missing source entries are represented by NA. Raw employee counts and closure status codes are retained without recoding.",
             "store_id combines treated_state and survey_sheet with a space; survey_sheet alone is not unique across states.",
             "", "| Variable | Meaning |", "|---|---|"]
    for c in cols:
        name = RENAME[c]
        if name in MEANING:
            meaning = MEANING[name]
        elif name != "store_id":
            stem, wave = name.rsplit("_w", 1)
            meaning = f"Wave {wave} {WAVE[stem]}"
        else:
            continue
        lines.append(f"| {name} | {meaning} |")
    return "\n".join(lines) + "\n" + CODES


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SRC / "data.csv", newline="", encoding='utf-8')))
    cols = list(rows[0])
    check(set(cols) == set(RENAME), "the unmasked packet's columns differ from the renaming table")
    first = min(date(1992, int(r["date2"][0:2]), int(r["date2"][2:4])) for r in rows
                if r["date2"] not in ("NA", "") and r["date2"].endswith("92"))
    with open(OUT / "data.csv", "w", newline="", encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([RENAME[c] for c in cols])
        for r in rows:
            w.writerow([day_index(r[c], first) if c == "date2" else dollars(r[c]) if c in DOLLARS else r[c] for c in cols])
    (OUT / "DESIGN.md").write_text(DESIGN, encoding='utf-8')
    (OUT / "CODEBOOK.md").write_text(codebook(cols), encoding='utf-8')
    prompt = (SRC / "research-prompt.txt").read_text(encoding='utf-8')
    check(prompt.count(OLD_P1) == 1, "the research prompt's first sentences are not the expected ones")
    (OUT / "research-prompt.txt").write_text(prompt.replace(OLD_P1, NEW_P1), encoding='utf-8')
    for name in ("commitment_prompt.txt", "comparison_prompt.txt"):
        (OUT / name).write_bytes((SRC / name).read_bytes())
    for name in FILES:
        check(filecmp.cmp(OUT / name, SHIPPED / name, shallow=False), f"masked {name} differs from the packet Sonnet saw")
    print(f"masked packet: {len(rows)} stores, {len(cols)} columns, all {len(FILES)} files identical to the shipped packet")


if __name__ == "__main__":
    main()
