#!/usr/bin/env python3
"""The z-statistic of every run plotted in panel c of Miao et al.'s Supplementary Figure 3, read from the page's
vector graphics (one marker per run: teal for pro-immigration, red for anti-immigration personas).

Input:  data/raw/miao2026/paper/supplementary_figure_3_page35.svg
Output: data/derived/05_immigration_personas/miao_sfig3c_z.csv (persona, z)
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SVG = ROOT / "data/raw/miao2026/paper/supplementary_figure_3_page35.svg"
OUT = ROOT / "data/derived/05_immigration_personas/miao_sfig3c_z.csv"

TEAL = "rgb(16.470337%, 61.569214%, 56.079102%)"
RED = "rgb(83.921814%, 15.686035%, 15.686035%)"
Y_ZERO, PT_PER_Z = 168.37, (209.98 - 105.97) / 10.0  # gridlines z = 6 at y = 105.97, z = -4 at y = 209.98
# Marker columns of panel c (x centres, page points).
COLUMNS = {"pro": (TEAL, 288), "anti": (RED, 342)}


def markers(svg_text):
    """Centres and fill colours of the small filled paths (plot markers) drawn after the glyph definitions."""
    body = svg_text[svg_text.find("</defs>"):]
    out = []
    for attrs in re.findall(r"<path ([^>]*)/>", body):
        d = re.search(r'(?:^| )d="([^"]*)"', attrs)
        fill = re.search(r'fill="([^"]*)"', attrs)
        if not d or not fill or fill.group(1) not in (TEAL, RED):
            continue
        nums = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", d.group(1))]
        tr = re.search(r'transform="matrix\(([^)]*)\)"', attrs)
        a, b, c, e, f, g = map(float, tr.group(1).split(",")) if tr else (1, 0, 0, 1, 0, 0)
        pts = [(a * x + c * y + f, b * x + e * y + g) for x, y in zip(nums[0::2], nums[1::2])]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if max(xs) - min(xs) < 3 and max(ys) - min(ys) < 3:
            out.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, fill.group(1)))
    return out


def main():
    found = markers(SVG.read_text(encoding="utf-8"))
    rows = []
    for persona, (colour, x_centre) in COLUMNS.items():
        zs = sorted((Y_ZERO - y) / PT_PER_Z for x, y, fill in found
                    if fill == colour and abs(x - x_centre) < 3 and y < 230)
        rows += [(persona, z) for z in zs]
    assert sum(p == "anti" for p, _ in rows) == 17 and sum(p == "pro" for p, _ in rows) == 17, rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["persona", "z"])
        w.writerows((p, f"{z:.3f}") for p, z in sorted(rows, key=lambda r: (r[0], r[1])))


if __name__ == "__main__":
    main()
