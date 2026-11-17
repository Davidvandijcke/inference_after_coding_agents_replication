# Figure 1: a stylized illustration of selection and the replay correction, with simulated values.
# 480 null p-values are spread evenly over (0, 1); a fresh agent run re-selects the original analysis with
# probability 0.035 + 0.6 exp(-p / 0.12); the right panel maps each re-selected p-value to its rank among them.
#
# Input:  fonts/Inter-*.otf    Output: paper/figs/lens_figure.pdf
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "paper" / "figs" / "lens_figure.pdf"

for face in ("Regular", "Italic", "SemiBold"):
    fm.fontManager.addfont(str(HERE / "fonts" / f"Inter-{face}.otf"))
# mathtext.cal is not used; setting it avoids a lookup of its default, a system cursive font
plt.rcParams.update({"font.family": "Inter", "mathtext.fontset": "custom", "mathtext.rm": "Inter",
                     "mathtext.it": "Inter:italic", "mathtext.cal": "Inter"})

INK, MUTED = "#1d1d1f", "#7a7a80"
cmap = LinearSegmentedColormap.from_list("p", ["#8b1e4f", "#c0587e", "#9aa7c7", "#2f5d8c"])

rng = np.random.default_rng(7)
N = 480
p = (np.arange(N) + 0.5) / N                 # uniform under the null
s = 0.035 + 0.6 * np.exp(-p / 0.12)          # agent re-picks q* more often when p is small
sel = rng.random(N) < s
ps = p[sel]
H = (np.argsort(np.argsort(ps)) + 0.5) / len(ps)   # empirical selected cdf -> corrected p

# geometry
Y0, Y1 = 8, 86
yp = lambda u: Y0 + (Y1 - Y0) * u
XL, BX0, BX1, XF, XR1, XR2 = 10, 60.5, 81.5, 126, 160, 190
BOX_Y0, BOX_Y1 = 39.5, 59.5
BYC = (BOX_Y0 + BOX_Y1) / 2
squeeze = lambda y: BYC + (y - 50) * (BOX_Y1 - BOX_Y0 - 6) / (Y1 - Y0)

fig = plt.figure(figsize=(12, 6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-2, 206); ax.set_ylim(1, 101); ax.axis("off")

def curve(a, b, c1=0.5, c2=0.5, **kw):
    (x0, y0), (x1, y1) = a, b
    dx = x1 - x0
    path = MplPath([(x0, y0), (x0 + c1 * dx, y0), (x1 - c2 * dx, y1), (x1, y1)],
                   [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(path, fc="none", capstyle="round", **kw))

# other analyses the agent can land on
nodes = []
while len(nodes) < 34:
    x, y = rng.uniform(98, 146), rng.uniform(8, 86)
    if np.hypot(x - XF, y - 50) > 12 and not (x > XF - 2 and y < 52) and all(np.hypot(x - a, y - b) > 6 for a, b in nodes):
        nodes.append((x, y))
nodes = np.array(nodes)

# rays: data -> agent
for u in p:
    curve((XL + 2, yp(u)), (BX0 + 4, squeeze(yp(u))), lw=0.45, color=cmap(u), alpha=0.55, zorder=2)

# rays: agent -> analyses
for u, k in zip(p, sel):
    ye = squeeze(yp(u))
    if k:
        curve((BX1 - 4, ye), (XF, 50), lw=0.7, color=cmap(u), alpha=0.75, zorder=3)
    else:
        nx, ny = nodes[rng.integers(len(nodes))]
        curve((BX1 - 4, ye), (nx, ny), lw=0.35, color="#b9b9c0", alpha=0.35, zorder=2)

ax.scatter(nodes[:, 0], nodes[:, 1], s=16, fc="white", ec="#a9a9b0", lw=0.8, zorder=2.5)

# q* -> conditional p (bunched) -> corrected p (uniform)
for u, h in zip(ps, H):
    c = cmap(u)
    curve((XF, 50), (XR1, yp(u)), lw=0.8, color=c, alpha=0.8, zorder=3)
    curve((XR1 + 1.5, yp(u)), (XR2 - 1.5, yp(h)), lw=0.8, color=c, alpha=0.8, zorder=3)
    ax.plot([XR1 - 0.2, XR1 + 1.5], [yp(u)] * 2, color=c, lw=1.1, zorder=4, solid_capstyle="butt")
    ax.plot([XR2 - 1.5, XR2 + 0.3], [yp(h)] * 2, color=c, lw=1.1, zorder=4, solid_capstyle="butt")

# left rug: one tick per subsampled dataset, evenly spread
for u in p:
    ax.plot([XL - 0.5, XL + 2], [yp(u)] * 2, color=cmap(u), lw=0.8, solid_capstyle="butt", zorder=4)

# focal analysis q*
ax.scatter([XF], [50], s=120, fc=INK, ec="white", lw=2, zorder=6)
ax.text(XF, 54.5, r"$q^{*}$", fontsize=15, color=INK, ha="center", va="bottom", zorder=6)

# the agent: an opaque robot head
AX = 1.04                       # x-units per y-unit, keeps circles round
BXC = (BX0 + BX1) / 2
W, Hh = BX1 - BX0, BOX_Y1 - BOX_Y0
BODY = "#2e3440"
def rbox(x, y, w, h, r, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **kw))
def dot(x, y, d, **kw):
    ax.add_patch(Ellipse((x, y), d * AX, d, **kw))

ax.plot([BXC, BXC], [BOX_Y1, BOX_Y1 + 3.2], color=BODY, lw=1.6, zorder=4)
dot(BXC, BOX_Y1 + 3.6, 1.8, fc=BODY, ec="none", zorder=5)
rbox(BX0, BOX_Y0, W, Hh, 3.2, fc=BODY, ec="none", zorder=5)
for xe in (BXC - 4.2, BXC + 4.2):
    dot(xe, BYC + 1.5, 3.0, fc="white", ec="none", zorder=6)
ax.text(BXC, BOX_Y0 - 2.2, "coding agent", color=INK, fontsize=10, weight="semibold",
        ha="center", va="top", zorder=6, bbox=dict(fc="white", ec="none", pad=1.0))

# axis labels (minimal)
for x in [XL + 0.75, XR1 + 0.6, XR2 - 0.6]:
    ax.text(x, Y0 - 2.6, "0", fontsize=8.5, color=MUTED, ha="center", va="center")
    ax.text(x, Y1 + 2.6, "1", fontsize=8.5, color=MUTED, ha="center", va="center")
lab = dict(fontsize=10, color=MUTED, va="bottom", ha="center", linespacing=1.3)
ax.text(XL - 1, 92, "resampled datasets,\nsorted by their $p$", **{**lab, "ha": "left"})
ax.text((BX0 + BX1) / 2, 92, "same prompt,\nfresh run", **lab)
ax.text(XR1 + 0.6, 92, "$p$ when the agent\nreselects $q^{*}$", **lab)
ax.text(XR2 + 1, 92, "corrected $p$", **{**lab, "ha": "right"})

# the reported result, carried through the correction
u0 = 0.04
h0 = np.searchsorted(np.sort(ps), u0) / len(ps)
curve((XR1 + 1.5, yp(u0)), (XR2 - 1.5, yp(h0)), lw=2.2, color=INK, zorder=7)
ax.scatter([XR1 + 0.6, XR2 - 0.6], [yp(u0), yp(h0)], s=34, c=INK, zorder=8, ec="white", lw=1.2)
ax.text(XR1 - 1.5, yp(u0), "reported", fontsize=9.5, color=INK, ha="right", va="center", zorder=8,
        bbox=dict(fc="white", ec="none", pad=1.2))
ax.text(XR2 + 1.8, yp(h0), "corrected", fontsize=9.5, color=INK, ha="left", va="center", zorder=8)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white", metadata={"CreationDate": None})
