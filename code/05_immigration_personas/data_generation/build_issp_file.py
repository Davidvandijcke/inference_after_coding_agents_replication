#!/usr/bin/env python3
"""Build the respondent-level ISSP Role of Government file (1985-2016) with country macro data.

This is the restricted analysis file (issp_clean.csv) given to every agent on the corrected data (modules 05, 06
and 07). It is not shipped; data/raw/issp/SOURCE.md describes how to obtain the inputs.

Inputs (read-only): the five ISSP Role of Government modules (ZA1490, ZA1950, ZA2900, ZA4700 v1.0.0,
ZA6900 v2.0.0, Stata format) and the CRI country-year panel cri_macro.csv, all in one folder.

Usage, from the package root (about 3 seconds):
    python -I code/05_immigration_personas/data_generation/build_issp_file.py --src data/raw/issp \
        --out data/derived/05_immigration_personas/issp_clean.csv

Deterministic, no network. Every country code, item and category mapping is checked against the
value and variable labels stored in the Stata files; the build stops if a label differs from what
the mapping assumes or if the output's SHA-256 differs from that of the file the agents received
(checked before anything is written). An input whose SHA-256 differs from the copy used for the
paper gives a warning and the output check decides: those copies are the ones distributed with
the Crowdsourced Replication Initiative's materials, and downloads from GESIS were not tested.
Tested with Python 3.13.5, pandas 3.1.0 and numpy 2.5.4.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# SHA-256 of the inputs used for the paper and of the file the agents received.
INPUT_SHA256 = {
    "ZA1490.dta": "fe54f0eea7d8c9ea327f3175e7dc08882929e6d4335e69d43ebdb7b34ebbb80a",
    "ZA1950.dta": "dae4ba2929759da6c1fecc4abe4f3534f54aec6563b94bbcd35e4d9e24b75e30",
    "ZA2900.dta": "c26fa85da920f95f84d86ff3e7061dfcdd1508e5d78b24e73f272e15e8885b46",
    "ZA4700.dta": "6a188efd74a817e7fecbf073ca001ded2319eb4d875d23caaa666935156f561a",
    "ZA6900_v2-0-0.dta": "af39261f95b717e86d2cc4ff9d50a396d741ee58253639cdf903779ff7627ebc",
    "cri_macro.csv": "639fe8d1097b9e6e4d0312cb07ceba3079c21de901bea98870e23d7d468920e0",
}
OUTPUT_SHA256 = "bd2363cec654ea710d9bdbd389f23993c87b6ee7df5bf48a27394b64a4884491"

OUTCOMES = ["jobs", "healthcare", "old_age", "unemployed", "income_diff", "housing"]
MACRO = ["migstock_wb", "migstock_un", "migstock_oecd", "mignet_un", "pop_wb", "gdp_wb",
         "ginid_solt", "gini_wb", "wdi_unempilo", "socx_oecd", "al_ethnic"]
COLUMNS = (["wave", "year", "iso_country"] + OUTCOMES
           + ["age", "educ", "topbot", "female", "employed"]
           + MACRO + ["migstock", "migflow_pct", "log_gdp", "cntry_wave"])

# ISSP country abbreviations (1985-1996 integrated files) -> numeric ISO 3166 code.
# Split samples go to the national code: West/East Germany -> 276, Northern Ireland -> 826
# (United Kingdom), Israel Jews/Arabs -> 376.
ABBREV_ISO = {
    "aus": 36, "d": 276, "D-W": 276, "D-E": 276, "gb": 826, "nirl": 826, "usa": 840, "a": 40,
    "i": 380, "h": 348, "irl": 372, "n": 578, "il": 376, "IL-J": 376, "IL-A": 376, "s": 752,
    "cz": 203, "slo": 705, "pl": 616, "bg": 100, "rus": 643, "nz": 554, "cdn": 124, "rp": 608,
    "j": 392, "e": 724, "lv": 428, "f": 250, "cy": 196, "ch": 756,
}
# ISO numeric -> ISO alpha-2 prefix used in the 2006/2016 value labels ("AU-Australia").
ISO_ALPHA2 = {
    36: "AU", 56: "BE", 124: "CA", 152: "CL", 158: "TW", 191: "HR", 203: "CZ", 208: "DK",
    214: "DO", 246: "FI", 250: "FR", 268: "GE", 276: "DE", 348: "HU", 352: "IS", 356: "IN",
    372: "IE", 376: "IL", 392: "JP", 410: "KR", 428: "LV", 440: "LT", 528: "NL", 554: "NZ",
    578: "NO", 608: "PH", 616: "PL", 620: "PT", 643: "RU", 703: "SK", 705: "SI", 710: "ZA",
    724: "ES", 740: "SR", 752: "SE", 756: "CH", 764: "TH", 792: "TR", 826: "GB", 840: "US",
    858: "UY", 862: "VE",
}

# Module specifications. Item entries: (variable, substring that must occur in its variable label).
MODULES = [
    dict(wave="ZA1490", year=1985, file="ZA1490.dta", country="V3", country_kind="abbrev",
         country_labels={1: "aus", 2: "d", 3: "gb", 4: "usa", 5: "a", 8: "i"},
         items={"jobs": ("V101", "JOBS FOR ALL"), "healthcare": ("V103", "HEALTH CARE F SICK"),
                "old_age": ("V104", "PROVIDE FOR ELDERLY"), "unemployed": ("V106", "PROVIDE F UNEPMLOYD"),
                "income_diff": ("V107", "REDUCE INCOME DIFF"), "housing": None},
         age=("V117", "AGE"), sex=("V118", "SEX"), work=("V109", "CURRENT EMPLOYM STATUS"),
         work_kind="za1490", educ=None, topbot=None,
         age_categorical_codes={8}),  # Italy 1985 reports age in six unlabelled bands
    dict(wave="ZA1950", year=1990, file="ZA1950.dta", country="v3", country_kind="abbrev",
         country_labels={1: "aus", 2: "D-W", 3: "D-E", 4: "gb", 5: "nirl", 6: "usa", 7: "h",
                         8: "i", 9: "irl", 10: "n", 11: "il"},
         items={"jobs": ("v49", "JOBS FOR ALL"), "healthcare": ("v51", "HEALTH CARE F SICK"),
                "old_age": ("v52", "LIVING STANDARD FOR OLD"), "unemployed": ("v54", "PROVIDE F UNEMPLOYD"),
                "income_diff": ("v55", "REDUCE INCOME DIFF"), "housing": ("v57", "PROVIDE DECENT HOUSING")},
         age=("v60", "AGE OF RESPONDENT"), sex=("v59", "SEX OF RESPONDENT"),
         work=("v63", "CURRENT EMPLOYMENT STATUS"), work_kind="wrkst", educ=None, topbot=None,
         age_categorical_codes=set()),
    dict(wave="ZA2900", year=1996, file="ZA2900.dta", country="v3", country_kind="abbrev",
         country_labels={1: "aus", 2: "D-W", 3: "D-E", 4: "gb", 6: "usa", 8: "h", 9: "i", 10: "irl",
                         12: "n", 13: "s", 14: "cz", 15: "slo", 16: "pl", 17: "bg", 18: "rus",
                         19: "nz", 20: "cdn", 21: "rp", 22: "IL-J", 23: "IL-A", 24: "j", 25: "e",
                         26: "lv", 27: "f", 28: "cy", 30: "ch"},
         items={"jobs": ("v36", "provid jobs for all"), "healthcare": ("v38", "health care for sick"),
                "old_age": ("v39", "provide for elderly"), "unemployed": ("v41", "provide for unemployed"),
                "income_diff": ("v42", "reduce income differences"),
                "housing": ("v44", "provide decent housing")},
         age=("v201", "R: Age"), sex=("v200", "R: sex"), work=("v206", "R: Current employment status"),
         work_kind="wrkst", educ=("v205", "Education II"), educ_kind="za2900",
         topbot=None, age_categorical_codes=set()),
    dict(wave="ZA4700", year=2006, file="ZA4700.dta", country="V3a", country_kind="iso",
         items={"jobs": ("V25", "Provide job for everyone"), "healthcare": ("V27", "Provide health care for sick"),
                "old_age": ("V28", "Provide living standard for the old"),
                "unemployed": ("V30", "Provide living standard for unemployed"),
                "income_diff": ("V31", "Reduce income differences"),
                "housing": ("V33", "Provide decent housing")},
         age=("age", "R: Age"), sex=("sex", "R: Sex"), work=("wrkst", "R: Current employment status"),
         work_kind="wrkst", educ=("degree", "highest education level"), educ_kind="degree2006",
         topbot=("topbot", "Top Bottom self-placement"), age_categorical_codes=set()),
    dict(wave="ZA6900", year=2016, file="ZA6900_v2-0-0.dta", country="country", country_kind="iso",
         items={"jobs": ("v21", "provide job for everyone"), "healthcare": ("v23", "provide health care for sick"),
                "old_age": ("v24", "provide living standard for the old"),
                "unemployed": ("v26", "provide living standard for unemployed"),
                "income_diff": ("v27", "reduce income differences"),
                "housing": ("v29", "provide decent housing")},
         age=("AGE", "Age of respondent"), sex=("SEX", "Sex of Respondent"),
         work=("WORK", "Currently, formerly, or never in paid work"), work_kind="work2016",
         educ=("DEGREE", "Highest completed education level"), educ_kind="degree2016",
         topbot=("TOPBOT", "Top-Bottom self-placement"), age_categorical_codes=set()),
]

# Harmonized education (educ), 0-5:
# 0 no formal qualification / incomplete primary; 1 primary completed (lowest formal qualification);
# 2 lower secondary / incomplete secondary (above lowest qualification); 3 upper secondary completed;
# 4 post-secondary below a tertiary degree; 5 tertiary (university-level) degree completed.
EDUC_MAPS = {
    # ZA2900 v205 "Education II: categories"
    "za2900": {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5},
    # ZA4700 DEGREE (ISSP 1996-2010 definition)
    "degree2006": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    # ZA6900 DEGREE (ISCED-2011-based ISSP definition); lower and upper tertiary both -> 5
    "degree2016": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5},
}
EDUC_LABEL_CHECK = {
    "za2900": {1: "none", 2: "incpl primary", 3: "primary compl", 4: "incpl secondary",
               5: "secondary compl", 6: "semi-higher", 7: "university compl"},
    "degree2006": {0: "no formal qualification", 1: "lowest formal qualification",
                   2: "above lowest qualification", 3: "higher secondary completed",
                   4: "above higher secondary", 5: "university degree completed"},
    "degree2016": {0: "no formal education", 1: "primary school", 2: "lower secondary",
                   3: "upper secondary", 4: "post secondary, non-tertiary",
                   5: "lower level tertiary", 6: "upper level tertiary"},
}


def fail(msg: str) -> None:
    sys.exit(f"BUILD STOPPED: {msg}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_module(path: Path, cols: list[str]):
    with pd.io.stata.StataReader(path) as rdr:
        var_labels = rdr.variable_labels()
        lbl_names = dict(zip(rdr._varlist, rdr._lbllist))
        val_labels = rdr.value_labels()
    # Stata missing values (".", ".a", ...) become NaN; numeric non-substantive codes stay numeric.
    df = pd.read_stata(path, convert_categoricals=False, convert_missing=False, columns=cols)
    vlab = {c: {int(k): str(v) for k, v in val_labels.get(lbl_names.get(c, ""), {}).items()} for c in cols}
    return df, var_labels, vlab


def check_var_label(var_labels: dict, var: str, needle: str, wave: str) -> None:
    lab = str(var_labels.get(var, ""))
    if needle.lower() not in lab.lower():
        fail(f"{wave}: variable {var} has label '{lab.strip()}', expected it to contain '{needle}'")


def check_value_labels(vlab: dict, expected: dict, what: str) -> None:
    """expected: code -> regular expression (case-insensitive) the stripped label must match."""
    for code, pattern in expected.items():
        lab = vlab.get(code, "").strip()
        if not re.search(pattern, lab, flags=re.IGNORECASE):
            fail(f"{what}: code {code} is labelled '{lab}', expected /{pattern}/")


def country_codes(df, m, vlab) -> pd.Series:
    cv = m["country"]
    if m["country_kind"] == "abbrev":
        labels = vlab[cv]
        present = sorted(int(x) for x in df[cv].dropna().unique())
        if present != sorted(m["country_labels"]):
            fail(f"{m['wave']}: country codes present {present} differ from the specification")
        for code, abbrev in m["country_labels"].items():
            if labels.get(code) != abbrev:
                fail(f"{m['wave']}: country code {code} labelled '{labels.get(code)}', expected '{abbrev}'")
        iso = df[cv].map({c: ABBREV_ISO[a] for c, a in m["country_labels"].items()})
    else:
        labels = vlab[cv]
        iso = df[cv]
        for code in sorted(int(x) for x in iso.dropna().unique()):
            lab = labels.get(code, "")
            if code not in ISO_ALPHA2 or not lab.startswith(ISO_ALPHA2[code] + "-"):
                fail(f"{m['wave']}: ISO code {code} labelled '{lab}', not the expected country")
    if iso.isna().any():
        fail(f"{m['wave']}: {int(iso.isna().sum())} respondents without a country code")
    return iso.astype(int)


def check_split_samples(df, m) -> None:
    """The sample variable must nest within the national ISO code."""
    if m["wave"] == "ZA4700":
        s = pd.read_stata(Path(m["_path"]), convert_categoricals=False, columns=["V3"])["V3"]
        if not (np.floor(s.values) == df["V3a"].values).all():
            fail("ZA4700: V3 sample code does not nest within V3a")
    if m["wave"] == "ZA6900":
        s = pd.read_stata(Path(m["_path"]), convert_categoricals=False, columns=["c_sample"])["c_sample"]
        nat = np.where(s.values > 1000, s.values // 100, s.values)
        if not (nat == df["country"].values).all():
            fail("ZA6900: c_sample does not nest within country")


def recode_item(x: pd.Series) -> pd.Series:
    """1 definitely should -> 1, 2 probably should -> 2/3, 3 probably should not -> 1/3,
    4 definitely should not -> 0; every other code (can't choose, no answer, not asked) -> NaN."""
    return ((4 - x) / 3).where(x.isin([1, 2, 3, 4]))


def build_module(src: Path, m: dict) -> pd.DataFrame:
    path = src / m["file"]
    m["_path"] = str(path)
    cols = [m["country"]]
    for spec in m["items"].values():
        if spec:
            cols.append(spec[0])
    for key in ("age", "sex", "work", "educ", "topbot"):
        if m.get(key):
            cols.append(m[key][0])
    df, var_labels, vlab = read_module(path, cols)
    w = m["wave"]

    out = pd.DataFrame(index=df.index)
    out["wave"] = w
    out["year"] = m["year"]
    out["iso_country"] = country_codes(df, m, vlab)
    check_split_samples(df, m)

    # Outcomes
    for name, spec in m["items"].items():
        if spec is None:
            out[name] = np.nan
            continue
        var, needle = spec
        check_var_label(var_labels, var, needle, w)
        check_value_labels(vlab[var], {1: r"^definitely should( be)?$", 2: r"^probably should( be)?$",
                                       3: r"^probably (should )?not( be)?$",
                                       4: r"^definitely (should )?not( be)?$"}, f"{w} {var}")
        out[name] = recode_item(df[var])

    # Age in years
    var, needle = m["age"]
    check_var_label(var_labels, var, needle, w)
    nonresponse = [c for c, lab in vlab[var].items() if re.search(r"\bna\b|nap|refused|no answer|dk", lab, re.I)]
    age = df[var].where((df[var] >= 15) & (df[var] <= 99) & ~df[var].isin(nonresponse))
    if m["age_categorical_codes"]:
        bad = df[m["country"]].isin(m["age_categorical_codes"])
        age = age.where(~bad)
    out["age"] = age

    # Female
    var, needle = m["sex"]
    check_var_label(var_labels, var, needle, w)
    check_value_labels(vlab[var], {1: r"^male$", 2: r"^female$"}, f"{w} {var}")
    out["female"] = df[var].map({1: 0, 2: 1})

    # Employed (currently in paid work)
    var, needle = m["work"]
    check_var_label(var_labels, var, needle, w)
    x = df[var]
    if m["work_kind"] == "za1490":
        check_value_labels(vlab[var], {1: r"^unemployed$", 2: r"^employed$", 8: r"^not in labor force$"},
                           f"{w} {var}")
        # Not-in-labour-force and item non-response share Stata's system missing in this file;
        # no respondent with a missing status reports weekly hours. Coded 0 (not in paid work).
        out["employed"] = x.map({2: 1, 1: 0}).fillna(0)
    elif m["work_kind"] == "wrkst":
        check_value_labels(vlab[var], {1: r"full|f-time", 2: r"part|p-t", 3: r"less", 4: r"help",
                                       5: r"^unemployed$", 6: r"stud", 7: r"^retired$", 8: r"house",
                                       9: r"disabled", 10: r"^oth"}, f"{w} {var}")
        out["employed"] = x.map({1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0})
    elif m["work_kind"] == "work2016":
        check_value_labels(vlab[var], {1: r"^currently in paid work", 2: r"^currently not in paid work",
                                       3: r"^never had paid work", 9: r"^no answer"}, f"{w} {var}")
        out["employed"] = x.map({1: 1, 2: 0, 3: 0})
    else:
        fail(f"unknown work_kind {m['work_kind']}")

    # Education
    if m.get("educ"):
        var, needle = m["educ"]
        check_var_label(var_labels, var, needle, w)
        check_value_labels(vlab[var], {k: re.escape(v) for k, v in EDUC_LABEL_CHECK[m["educ_kind"]].items()},
                           f"{w} {var}")
        out["educ"] = df[var].map(EDUC_MAPS[m["educ_kind"]])
    else:
        out["educ"] = np.nan

    # Top-bottom self-placement
    if m.get("topbot"):
        var, needle = m["topbot"]
        check_var_label(var_labels, var, needle, w)
        lab1, lab10 = vlab[var].get(1, "").lower(), vlab[var].get(10, "").lower()
        if "lowest" not in lab1 or "highest" not in lab10:
            fail(f"{w} {var}: 1/10 labelled '{lab1}'/'{lab10}', expected lowest/highest")
        out["topbot"] = df[var].where(df[var].between(1, 10))
    else:
        out["topbot"] = np.nan

    # Sanity: no country-wave may carry banded ages (all ages < 15) after recoding
    agemax = out.groupby("iso_country")["age"].max()
    if (agemax.dropna() < 30).any():
        fail(f"{w}: implausible age maxima {agemax[agemax < 30].to_dict()}")
    return out


def read_macro(path: Path) -> pd.DataFrame:
    mc = pd.read_csv(path, na_values=[".", "..", "#VALUE!"], keep_default_na=True, dtype={"country": str},
                     encoding="utf-8")
    for c in MACRO:
        if mc[c].dtype == object or str(mc[c].dtype).startswith("str"):
            mc[c] = pd.to_numeric(mc[c].astype("string").str.strip().replace({"": pd.NA, "..": pd.NA,
                                                                              "#VALUE!": pd.NA, ".": pd.NA}),
                                  errors="raise")
    if mc.duplicated(["iso_country", "year"]).any():
        fail("cri_macro.csv has duplicate (iso_country, year) rows")
    return mc[["iso_country", "year"] + MACRO]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="folder with the ISSP .dta files and cri_macro.csv")
    ap.add_argument("--out", required=True, type=Path, help="output CSV path")
    a = ap.parse_args()

    differing = []
    for name, expected in INPUT_SHA256.items():
        got = sha256(a.src / name)
        print(f"{name}  sha256 {got}")
        if got != expected:
            differing.append(name)
            print(f"WARNING: {name} differs from the copy used for the paper (data/raw/issp/SOURCE.md lists the "
                  "versions); the check of the output decides", file=sys.stderr)

    parts = [build_module(a.src, m) for m in MODULES]
    df = pd.concat(parts, ignore_index=True)
    n_resp = len(df)

    macro = read_macro(a.src / "cri_macro.csv")
    df = df.merge(macro, how="left", left_on=["iso_country", "year"], right_on=["iso_country", "year"],
                  validate="many_to_one")
    if len(df) != n_resp:
        fail("macro merge changed the number of rows")

    df["migstock"] = df["migstock_wb"].where(df["migstock_wb"].notna(), df["migstock_un"])
    df["migflow_pct"] = df["mignet_un"] / 10.0          # net migrants per 100 inhabitants
    if (df["gdp_wb"] <= 0).any():
        fail("non-positive gdp_wb")
    df["log_gdp"] = np.log(df["gdp_wb"])
    df["cntry_wave"] = df["iso_country"].astype(str) + "_" + df["year"].astype(str)

    for c in ["age", "educ", "topbot", "female", "employed"]:
        if not ((df[c].dropna() % 1) == 0).all():
            fail(f"{c} has non-integer values")
        df[c] = df[c].astype("Int64")
    if not ((df["pop_wb"].dropna() % 1) == 0).all():
        fail("pop_wb has non-integer values")
    df["pop_wb"] = df["pop_wb"].astype("Int64")

    df = df[COLUMNS]
    cw = df.groupby("cntry_wave").size()
    print(f"rows {len(df)}  countries {df['iso_country'].nunique()}  country-waves {len(cw)}")
    print(df.groupby("year").size().to_string())
    data = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    print(f"sha256 {digest}")
    if digest != OUTPUT_SHA256:
        fail(f"output SHA-256 differs from the file the agents received ({OUTPUT_SHA256}); nothing written"
             + (f"; inputs that differ from the copies used for the paper: {', '.join(differing)}" if differing else ""))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
