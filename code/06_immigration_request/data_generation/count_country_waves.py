#!/usr/bin/env python3
"""Respondents per country-wave in the restricted ISSP analysis file (an aggregate that may be shipped).

Usage, from the package root:
    python code/06_immigration_request/data_generation/count_country_waves.py <issp_clean.csv> <output folder>
Writes <output folder>/issp_country_waves.csv (iso_country, year, respondents) in the format of the shipped
data/intermediate/06_immigration_request/issp_country_waves.csv, from which run.sh takes the file's respondent,
country and country-wave counts. Standard library only; seconds.
"""
import collections
import csv
import sys
from pathlib import Path

with open(sys.argv[1], newline="", encoding="utf-8") as fh:
    counts = collections.Counter((int(r["iso_country"]), int(r["year"])) for r in csv.DictReader(fh))
out = Path(sys.argv[2]) / "issp_country_waves.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["iso_country", "year", "respondents"])
    w.writerows((c, y, n) for (c, y), n in sorted(counts.items()))
