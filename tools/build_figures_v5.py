"""Write paper/figures.ipynb - the visualisation-only notebook for the virtual-articulography article.

It reads the small saved outputs of steps 19-22 (CSV) and step 24 (signal assets, .npz) - no audio, no models, no
analysis - so it runs in about a minute on Kaggle and in seconds locally. Every figure is checked automatically for
overlapping text (collision checker) before it is saved. Figure cells live in tools/figure_nature_cells.py.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Bradykinesia of the mouth - article figures, draft 5 (visualisation only)

Reads the saved outputs of the finished analysis notebooks (steps 19-22 result CSVs, step 24 signal assets) and only
draws. Light summaries (z-scores, correlations, PCA for display) take well under a second; nothing is re-analysed.

* Palette 'Navy & Crimson': deep navy = healthy, crimson = Parkinson's, steel-blue accent; no pink or orange anywhere.
* Every figure goes through `check_collisions()` (overlapping, clipped or cross-panel text). Target: 0 everywhere.
* Output: `figures_v5/` (PNG 300 dpi + vector PDF).
""")

code(r'''
# ---------------------------------------------------------------- setup
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch, Polygon, Wedge
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.stats import norm, spearmanr, gaussian_kde
from scipy.interpolate import make_interp_spline

def _find_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "results").is_dir():
            return p
    return Path.cwd()

ON_KAGGLE = Path("/kaggle/input").is_dir()
ROOT = Path("/kaggle/working") if ON_KAGGLE else _find_root()
SEARCH = [Path("/kaggle/input")] if ON_KAGGLE else [ROOT / "results"]
FIG_DIR = (ROOT / "figures_v5") if ON_KAGGLE else (ROOT / "paper" / "figures_v5")
TAB_DIR = (ROOT / "tables") if ON_KAGGLE else (ROOT / "paper" / "tables")
FIG_DIR.mkdir(parents=True, exist_ok=True); TAB_DIR.mkdir(parents=True, exist_ok=True)

_cache = {}
def _find(name, step=None):
    tag = None if step is None else step.replace("-", "_")
    hits = [p for base in SEARCH for p in base.rglob(name) if tag is None or tag in str(p).replace("-", "_")]
    if not hits:
        raise FileNotFoundError(name)
    return sorted(hits)[0]

def load(name, step=None):
    key = (name, step)
    if key not in _cache:
        _cache[key] = pd.read_csv(_find(name, step))
    return _cache[key].copy()

def load_npz(name):
    return np.load(_find(name), allow_pickle=True)

print("reading from", SEARCH, "| figures ->", FIG_DIR)
''')

md("## Style - edit here and re-run")

code(r'''
MM = 1 / 25.4
STYLE = dict(
    healthy="#1F3A5F", pd="#B22234",                                  # groups: deep navy / crimson
    english="#3A7CA5", italian="#6B5B95", phone="#8C8FA3",            # datasets: steel blue / muted violet / grey violet
    ink="#1B2333", ink2="#4E5668", grey="#A2A8B3", light="#E8EAEE", panel="#F4F5F7",
    passed="#2E5A87", failed="#DADDE3", band="#EEF2F7", grid="#E3E6EB", accent="#7FA6CF",
    fonts=["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], size=6.5, small=5.6, letter=8.5,
    w1=88 * MM, w2=180 * MM, dpi=300, formats=("png", "pdf"),
)
ART = {"UL": "Upper lip", "LL": "Lower lip", "LI": "Jaw", "TT": "Tongue tip", "TB": "Tongue body",
       "TD": "Tongue back", "LA": "Lip opening"}
ACOL = {"TT": "#2563EB", "TB": "#7C3AED", "TD": "#1E3A8A", "LI": "#475569", "LL": "#0D9488", "UL": "#0EA5E9",
        "LA": "#65A30D"}
VALIDATED = {"TT_range", "TB_space", "LL_range", "LL_speed", "LA_speed"}          # passed the real-sensor check (step 21)
SPEC_CMAP = LinearSegmentedColormap.from_list("spec", ["#05081A", "#14244F", "#2B4C8C", "#5B54A8", "#8E3B7E", "#C0303F", "#F6E7E4"])

def label(m):
    a, q = m.split("_", 1)
    return f"{ART.get(a, a)} {'working space' if q == 'space' else q}"

def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": STYLE["fonts"], "font.size": STYLE["size"],
        "axes.titlesize": STYLE["size"], "axes.titleweight": "normal", "axes.labelsize": STYLE["size"],
        "xtick.labelsize": STYLE["small"], "ytick.labelsize": STYLE["small"], "legend.fontsize": STYLE["small"],
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderaxespad": 0.2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5, "axes.edgecolor": STYLE["ink2"],
        "axes.labelcolor": STYLE["ink"], "xtick.color": STYLE["ink2"], "ytick.color": STYLE["ink2"],
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.titlepad": 3, "axes.labelpad": 2,
        "lines.linewidth": 0.9, "patch.linewidth": 0.5, "savefig.dpi": STYLE["dpi"], "figure.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white",
    })
apply_style()
print("fonts available:", [f for f in STYLE["fonts"] if any(f == x.name for x in mpl.font_manager.fontManager.ttflist)])
''')

md("## Helpers - collision checker, drawing primitives, mouth illustration")

code(r'''
# ---------------------------------------------------------------- quality control: text collisions
def _texts(fig, r):
    out = []
    def add(t, owner):
        if t is None or not t.get_visible() or not t.get_text().strip():
            return
        try:
            bb = t.get_window_extent(r)
        except Exception:
            return
        if bb.width > 0.5 and bb.height > 0.5:
            out.append((t.get_text().strip().replace("\n", " ")[:40], owner, bb))
    for ax in fig.axes:
        for t in (ax.title, ax._left_title, ax._right_title):
            add(t, ax)
        if ax.axison:                       # hidden axes (schematics, insets) carry no visible tick labels
            add(ax.xaxis.label, ax); add(ax.yaxis.label, ax)
            for axis in (ax.xaxis, ax.yaxis):
                lo, hi = sorted(axis.get_view_interval())
                for tk in axis.get_major_ticks():
                    if lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:   # ticks outside the limits are never drawn
                        add(tk.label1, ax); add(tk.label2, ax)
        for t in ax.texts:
            add(t, ax)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                add(t, ax)
    for t in fig.texts:
        add(t, None)
    return out

def check_collisions(fig, pad=0.6, verbose=True):
    """Report overlapping text, text running into another panel, and text cut off at the figure edge."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    T = _texts(fig, r)
    fb = fig.bbox
    issues = []
    ov = lambda a, b: a.x0 + pad < b.x1 and b.x0 + pad < a.x1 and a.y0 + pad < b.y1 and b.y0 + pad < a.y1
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            if ov(T[i][2], T[j][2]):
                issues.append(f"text overlap: '{T[i][0]}' x '{T[j][0]}'")
    for s, owner, bb in T:
        if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1 or bb.y0 < fb.y0 - 1 or bb.y1 > fb.y1 + 1:
            issues.append(f"cut off at edge: '{s}' at ({bb.x0:.0f},{bb.y0:.0f})-({bb.x1:.0f},{bb.y1:.0f}) of {fb.x1:.0f}x{fb.y1:.0f}")
        for ax in fig.axes:
            if ax is owner or not ax.get_visible() or not ax.axison:
                continue
            if ov(bb, ax.get_window_extent(r)):
                issues.append(f"text in another panel: '{s}'")
    if verbose:
        print(f"  collisions: {len(issues)}" + ("" if not issues else "\n    " + "\n    ".join(issues[:25])))
    return issues

QC = {}
def save(fig, name):
    issues = check_collisions(fig)
    QC[name] = len(issues)
    for fmt in STYLE["formats"]:
        fig.savefig(FIG_DIR / f"{name}.{fmt}", dpi=STYLE["dpi"], facecolor="white", bbox_inches="tight", pad_inches=0.03)
    print(f"saved {name}  ({len(issues)} collisions)")
    return issues

def letter(fig, ax, s, dx=0.0, dy=0.004):
    """Panel letter just above the top-left of the axes' full extent (labels included), in figure coordinates.
    The figure is drawn once per figure (not once per letter) to keep the notebook fast."""
    if not getattr(fig, "_drawn_once", False):
        fig.canvas.draw(); fig._drawn_once = True
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(max(bb.x0 + dx, 0.002), min(bb.y1 + dy, 0.985), s, fontsize=STYLE["letter"], fontweight="bold",
             va="bottom", ha="left")

def stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"

def se_from_p(beta, p):
    z = norm.isf(np.clip(np.asarray(p, float), 1e-12, 1) / 2)
    return np.abs(beta) / np.maximum(z, 1e-6)

def zs(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)

def raincloud(ax, data, pos, colors, width=0.32, s=5, seed=0):
    """Half violin (right) + interquartile bar and median (centre) + jittered points (left)."""
    for k, (d, p, c) in enumerate(zip(data, pos, colors)):
        d = np.asarray(d, float); d = d[np.isfinite(d)]
        if len(d) < 3:
            continue
        sd = d.std() or 1
        ys = np.linspace(d.min() - 0.15 * sd, d.max() + 0.15 * sd, 120)     # violin stays within the data range
        dens = gaussian_kde(d)(ys); dens = dens / dens.max() * width
        ax.fill_betweenx(ys, p + 0.04, p + 0.04 + dens, color="#E6E9EE", alpha=0.95, lw=0)
        ax.plot(p + 0.04 + dens, ys, color=c, lw=0.6)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([p, p], [q1, q3], color=c, lw=2.6, solid_capstyle="butt", zorder=4)
        ax.scatter([p], [med], s=9, color="white", edgecolor=c, lw=0.7, zorder=5)
        jit = np.random.default_rng(seed + k).uniform(-0.30, -0.10, len(d))
        ax.scatter(p + jit, d, s=s, color=c, alpha=0.8, lw=0, zorder=3)

def bracket(ax, x0, x1, y, text, dy):
    ax.plot([x0, x0, x1, x1], [y, y + dy, y + dy, y], color=STYLE["ink2"], lw=0.5, clip_on=False)
    ax.text((x0 + x1) / 2, y + dy * 1.15, text, ha="center", va="bottom", fontsize=STYLE["small"], clip_on=False)

def show_spec(ax, spec, fmax=5000, floor_pct=45, sr=16000, hop=0.01, ylabel=True):
    S = np.asarray(spec, float)
    f = np.linspace(0, sr / 2, S.shape[0]); keep = f <= fmax
    t = np.arange(S.shape[1]) * hop
    v = S[keep]
    ax.imshow(v, origin="lower", aspect="auto", cmap=SPEC_CMAP, extent=[t[0], t[-1], 0, fmax / 1000],
              vmin=np.percentile(v, floor_pct), vmax=np.percentile(v, 99.7), interpolation="antialiased")
    if ylabel:
        ax.set_ylabel("Frequency (kHz)")
    ax.set_yticks(np.arange(0, fmax / 1000 + 0.1, 1 if fmax <= 4000 else 2))

def ell_from_cov(cxx, cxy, cyy, k=1.0):
    w, v = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    return 2 * k * np.sqrt(max(w[1], 0)), 2 * k * np.sqrt(max(w[0], 0)), np.degrees(np.arctan2(v[1, 1], v[0, 1]))

# ---------------------------------------------------------------- mid-sagittal mouth illustration
CHN = ["TDX", "TDY", "TBX", "TBY", "TTX", "TTY", "LIX", "LIY", "ULX", "ULY", "LLX", "LLY"]
CI_ = {c: i for i, c in enumerate(CHN)}
FS_EMA = 50.0
ANCHOR = {"UL": (-3.45, 1.25), "LL": (-3.45, -0.75), "LI": (-2.45, -2.0), "TT": (-1.75, -0.05),
          "TB": (0.0, 0.55), "TD": (1.7, -0.15)}
SCALE = 0.45

def smooth(pts, n=200, closed=False):
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    t = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    k = min(3, len(pts) - 1)
    bc = "periodic" if closed and len(pts) > 3 else None
    return make_interp_spline(t, pts, k=k, bc_type=bc)(np.linspace(0, t[-1], n))

def draw_mouth(ax, tongue=True, fill="#F2F4F7", edge="#B3BAC4", lw=0.8, labels=False, label_size=None):
    """Stylised mid-sagittal head (schematic): face, hard palate and pharynx, teeth, lips, tongue."""
    face = [(-1.0, 5.6), (-3.3, 4.7), (-4.75, 3.3), (-4.1, 2.55), (-4.25, 1.8), (-3.95, 1.25), (-4.05, 0.15),
            (-3.95, -0.85), (-4.15, -1.65), (-3.75, -2.75), (-2.4, -3.55), (-0.3, -3.65), (1.2, -3.95), (1.9, -5.5),
            (4.6, -5.5), (4.9, -1.0), (4.6, 2.5), (3.2, 4.8), (1.0, 5.8)]
    ax.add_patch(Polygon(smooth(face, 400, closed=True), closed=True, fc=fill, ec=edge, lw=lw, zorder=0.5))
    palate = smooth([(-3.05, 1.3), (-2.2, 2.15), (-0.6, 2.55), (1.0, 2.35), (2.25, 1.75), (3.0, 0.6), (3.15, -1.3),
                     (2.95, -3.6)])
    ax.plot(palate[:, 0], palate[:, 1], color=edge, lw=lw * 1.4, zorder=1, solid_capstyle="round")
    jaw = smooth([(-3.1, -1.35), (-2.6, -2.15), (-1.0, -2.6), (0.9, -2.45), (2.05, -2.05)])
    ax.plot(jaw[:, 0], jaw[:, 1], color=edge, lw=lw, zorder=1)
    if tongue:
        tg = smooth([(-2.6, -0.25), (-1.75, 0.15), (0.0, 0.85), (1.7, 0.3), (2.55, -1.2), (2.5, -2.6), (0.5, -2.35),
                     (-1.6, -1.5)], 300, closed=True)
        ax.add_patch(Polygon(tg, closed=True, fc="#E3E8EF", ec="#BCC5D2", lw=0.5, zorder=0.8))
    for xy in ([(-3.3, 1.38), (-2.95, 1.38), (-2.9, 0.82), (-3.2, 0.78)],
               [(-2.95, -1.3), (-2.62, -1.3), (-2.58, -1.85), (-2.9, -1.88)]):
        ax.add_patch(Polygon(xy, closed=True, fc="white", ec=edge, lw=0.5, zorder=1.2))
    ax.add_patch(Ellipse((-3.75, 1.05), 0.75, 0.42, angle=-12, fc="#D5DCE6", ec="#AEB8C6", lw=0.4, zorder=1.1))
    ax.add_patch(Ellipse((-3.75, -0.62), 0.75, 0.42, angle=12, fc="#D5DCE6", ec="#AEB8C6", lw=0.4, zorder=1.1))
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-5.2, 5.0); ax.set_ylim(-4.6, 4.4)
    if labels:
        offs = {"UL": (-0.3, 1.4), "LL": (-0.6, -1.25), "LI": (0.9, -1.55), "TT": (-0.1, 1.45), "TB": (0.3, 1.5),
                "TD": (1.6, 1.25)}
        for a, (x, y) in ANCHOR.items():
            ax.scatter([x], [y], s=10, color=ACOL[a], zorder=6, ec="white", lw=0.4)
            dx, dy = offs[a]
            ax.annotate(ART[a], (x, y), xytext=(x + dx, y + dy), fontsize=label_size or STYLE["small"], color=ACOL[a],
                        ha="center", va="center", zorder=7,
                        arrowprops=dict(arrowstyle="-", color=ACOL[a], lw=0.4, shrinkA=0, shrinkB=2))

print("helpers ready")
''')

exec((ROOT / "tools" / "figure_v5_cells.py").read_text())

code(r'''
print("QUALITY CHECK - text collisions per figure (target 0):")
for k, v in QC.items():
    print(f"  {k:40s} {v}")
print("figures written:", sorted(p.name for p in FIG_DIR.glob("*.png")))
''')

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "paper" / "figures_v5.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
