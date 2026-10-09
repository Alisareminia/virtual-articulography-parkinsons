# Article figures, draft 4. Exec'd by tools/build_figure_notebook.py (provides md(), code() and the helper cell).
# Data: step 19-22 result CSVs and step 24 signal assets. Everything here only DRAWS (small summaries at most).

code(r'''
# ---------------------------------------------------------------- extra drawing helpers for draft 4
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_ind

DIV = LinearSegmentedColormap.from_list("div", ["#0B1730", "#1F3A5F", "#4F72A8", "#E9E9EC", "#C2565C", "#B22234", "#6E1020"])

def speed_line(ax, x, y, cmap="viridis", lw=0.9, alpha=0.9, vmax=None, z=4):
    """Trajectory coloured by instantaneous speed."""
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    sp = np.hypot(np.diff(x), np.diff(y))
    lc = LineCollection(seg, cmap=cmap, norm=Normalize(0, vmax or np.percentile(sp, 98)), lw=lw, alpha=alpha, zorder=z,
                        capstyle="round")
    lc.set_array(sp); ax.add_collection(lc)
    return lc

def nn_glyph(ax, x0, y0, w, h, layers=(4, 6, 6, 5, 3), col="#5B6C8F"):
    """A small neural-network drawing (nodes + edges) inside the box (x0, y0, w, h) in data coordinates."""
    xs = np.linspace(x0, x0 + w, len(layers))
    pos = [[(x, y) for y in np.linspace(y0 + h * 0.1, y0 + h * 0.9, n)] for x, n in zip(xs, layers)]
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=0.25, alpha=0.35, zorder=2)
    for layer in pos:
        for (x, y) in layer:
            ax.add_patch(plt.Circle((x, y), h * 0.045, fc="white", ec=col, lw=0.6, zorder=3))

def badge(ax, x, y, text, col, size=None):
    ax.text(x, y, text, ha="center", va="center", fontsize=size or STYLE["small"] - 0.3, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc=col, ec="none"), zorder=8)

def mic_icon(ax, x, y, s=1.0, col=STYLE["ink2"]):
    ax.add_patch(FancyBboxPatch((x - 0.9 * s, y - 0.2 * s), 1.8 * s, 3.0 * s, boxstyle="round,pad=0,rounding_size=0.9",
                                fc="white", ec=col, lw=0.6, zorder=5))
    ax.plot([x - 1.4 * s, x - 1.4 * s, x + 1.4 * s, x + 1.4 * s], [y + 0.9 * s, y - 0.6 * s, y - 0.6 * s, y + 0.9 * s],
            color=col, lw=0.6, zorder=5)
    ax.plot([x, x], [y - 0.6 * s, y - 1.5 * s], color=col, lw=0.6, zorder=5)
    ax.plot([x - 0.8 * s, x + 0.8 * s], [y - 1.5 * s, y - 1.5 * s], color=col, lw=0.6, zorder=5)

def phone_icon(ax, x, y, s=1.0, col=STYLE["ink2"]):
    ax.add_patch(FancyBboxPatch((x - 1.0 * s, y - 1.7 * s), 2.0 * s, 3.4 * s, boxstyle="round,pad=0,rounding_size=0.4",
                                fc="white", ec=col, lw=0.6, zorder=5))
    ax.plot([x - 0.35 * s, x + 0.35 * s], [y - 1.3 * s, y - 1.3 * s], color=col, lw=0.6, zorder=6)

def sensor_crossed(ax, x, y, s=1.0, col="#8E1B2A"):
    ax.add_patch(plt.Circle((x, y), 1.3 * s, fc="white", ec=col, lw=0.8, zorder=6))
    ax.add_patch(plt.Rectangle((x - 0.45 * s, y - 0.3 * s), 0.9 * s, 0.6 * s, fc="#CFCAC2", ec=STYLE["ink2"], lw=0.4, zorder=7))
    ax.plot([x + 0.45 * s, x + 0.9 * s], [y, y + 0.5 * s], color=STYLE["ink2"], lw=0.4, zorder=7)
    ax.plot([x - 0.92 * s, x + 0.92 * s], [y - 0.92 * s, y + 0.92 * s], color=col, lw=0.9, zorder=8)

def glow_ellipse(ax, xy, w, h, ang, col, k=(1.0, 1.2, 1.4), alphas=(0.55, 0.22, 0.10), z=3):
    for kk, al in zip(k[::-1], alphas[::-1]):
        ax.add_patch(Ellipse(xy, w * kk, h * kk, angle=ang, fc=col, ec="none", alpha=al, zorder=z))

def group_density(CL, cohort, label, art):
    n = json.loads(str(CL["counts"]))[f"{cohort}_{label}"]
    return CL[f"{cohort}_{label}_{art}"].astype(float) / n

def hdr_level(h, frac=0.9):
    """Density level whose super-level set holds `frac` of the mass."""
    v = np.sort(h.ravel())[::-1]; c = np.cumsum(v) / v.sum()
    return v[min(np.searchsorted(c, frac), len(v) - 1)]

def density_difference(CL, ctr, radius=2.4, sigma=1.6):
    """PD minus healthy movement density per articulator (mean of both reading cohorts), relative to the healthy peak."""
    XX, YY = np.meshgrid(ctr, ctr, indexing="ij"); R = np.hypot(XX, YY)
    D, HS, vlim = {}, {}, 0
    for a in ANCHOR:
        d = np.mean([gaussian_filter(group_density(CL, c, 1, a) - group_density(CL, c, 0, a), sigma) for c in ("MDVR", "IPVS")], axis=0)
        b = np.mean([gaussian_filter(group_density(CL, c, 0, a), sigma) for c in ("MDVR", "IPVS")], axis=0)
        D[a] = np.where(R <= radius, d / (b.max() + 1e-12), np.nan); HS[a] = b
        vlim = max(vlim, np.nanpercentile(np.abs(D[a]), 99))
    return D, HS, vlim

print("draft-4 helpers ready")
''')


code(r'''
# ---------------------------------------------------------------- draft-5 palette and primitives
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

H_, P_ = STYLE["healthy"], STYLE["pd"]
DIV = LinearSegmentedColormap.from_list("div", ["#0B1730", "#1F3A5F", "#4F72A8", "#E9E9EC", "#C2565C", "#B22234", "#6E1020"])
DIV_R = DIV.reversed()                      # for z of movement: red = less movement than healthy
SEQ_G = LinearSegmentedColormap.from_list("seqg", ["#FFFFFF", "#DCE6F2", "#7FA6CF", "#1F3A5F", "#0F1F3D"])
HEX_G = LinearSegmentedColormap.from_list("hexg", ["#EEF3F9", "#7FA6CF", "#1F3A5F", "#0F1F3D"])
SEVC = {0: "#CDD2DA", 1: "#B84A4E", 2: "#962030", 3: "#5E0E1C"}
DLM = ["TT_range", "TT_speed", "TB_range", "TB_speed", "TB_space", "TD_range", "TD_speed", "LI_range", "LI_speed",
       "LL_range", "LL_speed", "UL_range", "UL_speed", "LA_range", "LA_speed"]
SHORT = {"TT": "Tongue tip", "TB": "Tongue body", "TD": "Tongue back", "LI": "Jaw", "LL": "Lower lip", "UL": "Upper lip",
         "LA": "Lip opening"}
def short(m):
    a, q = m.split("_", 1)
    return f"{SHORT.get(a, a)} {'space' if q == 'space' else q}"
CLASSIC = {"Timing": ["speech_rate_syl_s", "artic_rate_syl_s", "pause_ratio", "pause_mean_s", "pause_rate_min", "run_mean_s", "rate_cv"],
           "Prosody": ["f0_sd_st", "f0_range_st", "intensity_sd_db", "stress_int_sd_db", "stress_f0_sd_st"],
           "Voice quality": ["hnr_db", "cpps_db", "h1h2_db", "alpha_ratio_db", "hammarberg_db", "hf_ratio_db", "centroid_hz",
                             "cons_centroid_hz", "cons_hf_db"],
           "Formants": ["F1_range", "F1_speed", "F2_range", "F2_speed", "F1F2_space"]}
CL_LAB = {"speech_rate_syl_s": "Speech rate", "artic_rate_syl_s": "Articulation rate", "pause_ratio": "Pause ratio",
          "pause_mean_s": "Pause length", "pause_rate_min": "Pause rate", "run_mean_s": "Run length", "rate_cv": "Rate variability",
          "f0_sd_st": "Pitch SD", "f0_range_st": "Pitch range", "intensity_sd_db": "Loudness SD", "stress_int_sd_db": "Stress loudness",
          "stress_f0_sd_st": "Stress pitch", "hnr_db": "HNR", "cpps_db": "CPPS", "h1h2_db": "H1-H2", "alpha_ratio_db": "Alpha ratio",
          "hammarberg_db": "Hammarberg index", "hf_ratio_db": "HF ratio", "centroid_hz": "Spectral centroid",
          "cons_centroid_hz": "Consonant centroid", "cons_hf_db": "Consonant HF", "F1_range": "F1 range", "F1_speed": "F1 speed",
          "F2_range": "F2 range", "F2_speed": "F2 speed", "F1F2_space": "F1-F2 space", "dl_composite": "DL composite",
          "TB_space": "Tongue space", "TT_range": "Tongue-tip range"}
FAMC = {"Deep learning": P_, "Timing": "#3A7CA5", "Prosody": "#8AB6C9", "Voice quality": "#7B6FA6", "Formants": "#2A9D8F"}

def rho_txt(x, y, lead="rho"):
    d = pd.DataFrame({"x": x, "y": y}).dropna(); r, p = spearmanr(d.x, d.y)
    pt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    return f"{lead} = {r:.2f}, {pt}, n = {len(d)}"

def tile_aspect(fig, ax, nx, ny):
    """mutation_aspect that makes FancyBboxPatch corners round when x and y data units differ in size."""
    pos = ax.get_position()
    return (pos.width * fig.get_figwidth() / nx) / (pos.height * fig.get_figheight() / ny)

def rtile(ax, x, y, w, h, color, r, aspect, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=aspect,
                                fc=color, ec="none", zorder=z))

def ygrid(ax, axis="y"):
    ax.grid(axis=axis, color=STYLE["grid"], lw=0.45, zorder=0); ax.set_axisbelow(True)

def eye(ax, pos, mu, lo, hi, col, h=0.30, alpha=0.38, vertical=False, z=2):
    """Interval drawn as a normal-shaped 'eye' (density of the estimate) instead of a bar."""
    sd = max((hi - lo) / 3.92, 1e-6); xs = np.linspace(lo, hi, 60); w = np.exp(-0.5 * ((xs - mu) / sd) ** 2) * h
    if vertical:
        ax.fill_betweenx(xs, pos - w, pos + w, color=col, alpha=alpha, lw=0, zorder=z)
    else:
        ax.fill_between(xs, pos - w, pos + w, color=col, alpha=alpha, lw=0, zorder=z)

def auc_of(x1, x0):
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float); x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    return mannwhitneyu(x1, x0).statistic / (len(x1) * len(x0))

def z_vs_healthy(S, cols):
    """z relative to healthy speakers of the same dataset and sex (falls back to all healthy if < 4)."""
    Z = S.copy()
    for coh in Z.cohort.unique():
        for mv in (0.0, 1.0):
            ref = Z[(Z.cohort == coh) & (Z.label == 0) & (Z.male == mv)]
            if len(ref) < 4:
                ref = Z[(Z.cohort == coh) & (Z.label == 0)]
            msk = (Z.cohort == coh) & (Z.male == mv)
            for c in cols:
                Z.loc[msk, c] = (S.loc[msk, c] - ref[c].mean()) / ref[c].std()
    return Z

def strip(ax, codes, cmap, vertical=True):
    a = np.asarray(codes, float)[:, None] if vertical else np.asarray(codes, float)[None, :]
    ax.imshow(np.ma.masked_invalid(a), cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)

from scipy.stats import mannwhitneyu
print("draft-5 palette ready")
''')

md("""
## Figure 1 - From an ordinary recording to a moving mouth
**a** study design: 147 speakers from three public datasets (two languages, telephone and microphone audio) are passed
through a pretrained acoustic-to-articulatory inversion network (self-supervised speech encoder + inversion head; no sensors,
no fine-tuning), giving 12 articulator channels at 50 Hz (virtual articulography), from which range, speed, working space
and tremor are measured for five analyses;
**b** waveform and **c** spectrogram with formant tracks of a typical healthy English reader; **d** the six articulator
movements the network estimates from the audio alone; **e** their mid-sagittal movement paths, coloured by speed
(anchor positions schematic, movements measured).
""")

code(r'''
apply_style()
X = load_npz("example_reading.npz")
fig = plt.figure(figsize=(STYLE["w2"], 134 * MM))
ax0 = fig.add_axes([0.005, 0.643, 0.99, 0.327]); ax0.set_xlim(0, 180); ax0.set_ylim(0, 48); ax0.axis("off")
T5 = load_npz("tremor_spectra.npz"); IX5 = load("tremor_spectra_index.csv")
INK, INK2 = STYLE["ink"], STYLE["ink2"]
cols5 = [(1, 39, "Recordings"), (43, 94, "Acoustic-to-articulatory inversion"), (98, 128, "Virtual articulography"),
         (132, 153, "Kinematics"), (157, 180, "Clinical analyses")]
for k, (x0, x1, t) in enumerate(cols5):
    ax0.text(x0, 47.6, f"{k + 1}", fontsize=STYLE["size"], fontweight="bold", color=STYLE["pd"], va="top")
    ax0.text(x0 + 2.6, 47.6, t.upper(), fontsize=STYLE["small"] - 0.6, fontweight="bold", color=INK, va="top")
    ax0.plot([x0, x1], [44.4, 44.4], color=INK, lw=0.6, solid_capstyle="butt")
for xa, xb in ((39.6, 42.4), (94.6, 97.4), (128.6, 131.4), (153.6, 156.4)):
    ax0.annotate("", xy=(xb, 24), xytext=(xa, 24), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, mutation_scale=6))
# 1: cohort table (booktabs style)
hx = [(1, "Dataset", "left"), (15.5, "Task / audio", "left"), (35.0, "PD", "right"), (39.0, "HC", "right")]
for x, h, ha in hx:
    ax0.text(x, 42.6, h, fontsize=STYLE["small"] - 0.7, color=INK2, ha=ha, va="top", fontstyle="italic")
ax0.plot([1, 39], [39.9, 39.9], color=INK, lw=0.35)
rows = [("MDVR-KCL", "English", "read passage", "phone", 16, 21, STYLE["english"]),
        ("IPVS", "Italian", "reading, vowels", "microphone", 25, 21, STYLE["italian"]),
        ("Figshare", "vowel only", "sustained /a/", "phone", 40, 24, STYLE["phone"])]
for k, (nm, lang, task, au, npd, nhc, c) in enumerate(rows):
    y = 38.6 - 5.9 * k
    ax0.add_patch(Rectangle((1, y - 3.2), 0.9, 3.2, fc=c, ec="none"))
    ax0.text(2.6, y, nm, fontsize=STYLE["small"] - 0.4, fontweight="bold", color=INK, va="top")
    ax0.text(2.6, y - 2.2, lang, fontsize=STYLE["small"] - 1.0, color=INK2, va="top")
    ax0.text(15.5, y, task, fontsize=STYLE["small"] - 0.9, color=INK, va="top")
    ax0.text(15.5, y - 2.2, au, fontsize=STYLE["small"] - 1.0, color=INK2, va="top")
    ax0.text(35.0, y, f"{npd}", fontsize=STYLE["small"] - 0.4, color=STYLE["pd"], ha="right", va="top", fontweight="bold")
    ax0.text(39.0, y, f"{nhc}", fontsize=STYLE["small"] - 0.4, color=STYLE["healthy"], ha="right", va="top", fontweight="bold")
ax0.plot([1, 39], [21.3, 21.3], color=INK, lw=0.35)
ax0.text(2.6, 20.4, "Total", fontsize=STYLE["small"] - 0.4, fontweight="bold", color=INK, va="top")
ax0.text(35.0, 20.4, "81", fontsize=STYLE["small"] - 0.4, color=STYLE["pd"], ha="right", va="top", fontweight="bold")
ax0.text(39.0, 20.4, "66", fontsize=STYLE["small"] - 0.4, color=STYLE["healthy"], ha="right", va="top", fontweight="bold")
ax0.plot([1, 39], [17.3, 17.3], color=INK, lw=0.6)
ax0.text(1, 15.6, "IPVS reading: 25 PD / 20 HC; retest: two readings.\nMDVR: H&Y, UPDRS speech items II-5 and III-18.",
         fontsize=STYLE["small"] - 1.1, color=INK2, va="top", linespacing=1.25)
wav = X["healthy_audio"]; tw = np.linspace(1, 39, 1400); env = np.abs(wav[: len(wav) // 1400 * 1400]).reshape(1400, -1).max(1)
env = env / env.max() * 2.3
ax0.fill_between(tw, 7.0 - env, 7.0 + env, color=INK, lw=0, alpha=0.75)
ax0.text(1, 1.2, "example: English reading", fontsize=STYLE["small"] - 1.1, color=INK2, va="bottom")
# 2: inversion network
ins1 = ax0.inset_axes([43, 17, 12.5, 19], transform=ax0.transData)
show_spec(ins1, X["healthy_spec"][:, :180], fmax=4000, ylabel=False); ins1.set_xticks([]); ins1.set_yticks([]); ins1.set_xlabel("")
for s_ in ins1.spines.values(): s_.set_visible(False)
ax0.annotate("", xy=(58.6, 26.5), xytext=(56.2, 26.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.5, mutation_scale=5))
nl = 9
for i in range(nl):
    x = 59.2 + i * 1.65; sh = i / (nl - 1)
    fc = LinearSegmentedColormap.from_list("enc", ["#DCE6F2", STYLE["healthy"]])(0.15 + 0.75 * sh)
    ax0.add_patch(Polygon([(x, 17.5), (x + 1.25, 19.6), (x + 1.25, 37.0), (x, 34.9)], closed=True, fc=fc, ec="white", lw=0.4, zorder=3))
ax0.text(66.6, 15.6, "self-supervised\nspeech encoder", fontsize=STYLE["small"] - 0.9, color=INK, ha="center", va="top", linespacing=1.15)
ax0.annotate("", xy=(76.6, 26.5), xytext=(74.4, 26.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.5, mutation_scale=5))
ax0.add_patch(Rectangle((76.9, 21.5), 3.2, 10, fc=STYLE["pd"], ec="none", alpha=0.9, zorder=3))
ax0.text(78.5, 15.6, "inversion\nhead", fontsize=STYLE["small"] - 0.9, color=INK, ha="center", va="top", linespacing=1.15)
ax0.annotate("", xy=(82.6, 26.5), xytext=(80.4, 26.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.5, mutation_scale=5))
ins2 = ax0.inset_axes([83, 17, 11, 19.5], transform=ax0.transData)
E0 = X["healthy_ema"][:120]
for k, c in enumerate(["TTY", "TTX", "TBY", "TBX", "TDY", "TDX", "LIY", "LIX", "LLY", "LLX", "ULY", "ULX"]):
    ins2.plot(zs(E0[:, CI_[c]]) * 0.32 - k, color=ACOL[c[:2]], lw=0.5)
ins2.axis("off")
ax0.text(88.5, 15.6, "12 channels\n50 Hz", fontsize=STYLE["small"] - 0.9, color=INK, ha="center", va="top", linespacing=1.15)
ax0.plot([43, 94], [8.6, 8.6], color=STYLE["light"], lw=0.5)
ax0.text(43, 7.4, "Pretrained on articulograph (EMA) recordings with matching audio.", fontsize=STYLE["small"] - 1.1, color=INK2, va="top")
ax0.text(43, 4.6, "Applied to every recording as is: no patient data, no fine-tuning, no sensors.", fontsize=STYLE["small"] - 1.1,
         color=INK2, va="top")
ax0.text(43, 42.6, "audio  ->  tongue, jaw and lip trajectories", fontsize=STYLE["small"] - 0.7, color=INK2, va="top", fontstyle="italic")
# 3: virtual articulography
ins3 = ax0.inset_axes([97.5, 2.5, 31, 39.5], transform=ax0.transData)
draw_mouth(ins3, lw=0.5); ins3.axis("off")
Eh = X["healthy_ema"]
for a, (cx, cy) in ANCHOR.items():
    xy = Eh[:, [CI_[f"{a}X"], CI_[f"{a}Y"]]]; d = (xy - xy.mean(0)) * SCALE
    ins3.plot(cx + d[:, 0], cy + d[:, 1], color=ACOL[a], lw=0.3, alpha=0.75, zorder=4)
    ins3.scatter([cx], [cy], s=9, color=ACOL[a], zorder=6, ec="white", lw=0.4)
# 4: kinematic glyphs from real data
yy = Eh[:240, CI_["TTY"]]; vv = np.abs(np.gradient(Eh[:240, CI_["TTY"]]) * FS_EMA)
g_specs = [("Range", "p5-p95 excursion"), ("Speed", "p95 velocity"), ("Working space", "x-y spread"), ("Tremor", "3-7 Hz power")]
for k, (t, sub) in enumerate(g_specs):
    yb = 34.6 - 10.0 * k
    gi = ax0.inset_axes([132, yb, 8.5, 7.6], transform=ax0.transData); gi.axis("off")
    if k == 0:
        gi.plot(yy, color=ACOL["TT"], lw=0.5); lo_, hi_ = np.percentile(yy, [5, 95])
        for v_ in (lo_, hi_):
            gi.axhline(v_, color=INK, lw=0.4, ls=(0, (1.5, 1)))
        gi.annotate("", xy=(len(yy) * 1.04, hi_), xytext=(len(yy) * 1.04, lo_), arrowprops=dict(arrowstyle="<->", lw=0.5, color=INK,
                                                                                                mutation_scale=4))
        gi.set_xlim(0, len(yy) * 1.1)
    elif k == 1:
        gi.fill_between(np.arange(len(vv)), 0, vv, color=ACOL["LL"], alpha=0.35, lw=0); gi.plot(vv, color=ACOL["LL"], lw=0.45)
        gi.axhline(np.percentile(vv, 95), color=INK, lw=0.4, ls=(0, (1.5, 1)))
    elif k == 2:
        xy = Eh[:, [CI_["TBX"], CI_["TBY"]]]; xy = xy - xy.mean(0)
        gi.scatter(xy[::2, 0], xy[::2, 1], s=0.6, color=ACOL["TB"], alpha=0.5, lw=0)
        cv = np.cov(xy.T); w_, h_, a_ = ell_from_cov(cv[0, 0], cv[0, 1], cv[1, 1], k=1.6)
        gi.add_patch(Ellipse((0, 0), w_, h_, angle=a_, fill=False, ec=INK, lw=0.6)); gi.set_aspect("equal", adjustable="datalim")
    else:
        fq = T5["freq"]; sel = (IX5.cohort == "IPVS") & (IX5.label == 1)
        sp = T5["jaw"][sel.to_numpy()].mean(0); keep_ = (fq >= 1) & (fq <= 12)
        gi.axvspan(3, 7, color=STYLE["grey"], alpha=0.3, lw=0)
        gi.plot(fq[keep_], 10 * np.log10(sp[keep_] / sp[keep_].max()), color=ACOL["LI"], lw=0.6); gi.set_xlim(1, 12)
    ax0.text(141.5, yb + 5.6, t, fontsize=STYLE["small"] - 0.7, fontweight="bold", color=INK, va="center")
    ax0.text(141.5, yb + 2.6, sub, fontsize=STYLE["small"] - 1.2, color=INK2, va="center")
# 5: analyses with their sample sizes
an5 = [("Patients vs healthy", "82 readers, 2 languages"), ("Severity", "16 rated patients"), ("Tremor", "110 vowel speakers"),
       ("Sensor validation", "1,378 sentences, 3 speakers"), ("Added value", "vs 26 classic features")]
for k, (t, sub) in enumerate(an5):
    y = 42.2 - 8.25 * k
    ax0.text(157, y, t, fontsize=STYLE["small"] - 0.5, fontweight="bold", color=INK, va="top")
    ax0.text(157, y - 2.7, sub, fontsize=STYLE["small"] - 1.1, color=INK2, va="top")
    if k < len(an5) - 1:
        ax0.plot([157, 180], [y - 6.1, y - 6.1], color=STYLE["light"], lw=0.5)
gs = fig.add_gridspec(3, 2, left=0.095, right=0.985, bottom=0.065, top=0.62, width_ratios=[1.65, 1],
                      height_ratios=[0.42, 1, 1.45], hspace=0.16, wspace=0.05)
x, spec, E = X["healthy_audio"], X["healthy_spec"], X["healthy_ema"]
ft, F = X["healthy_ftime"], X["healthy_formants"]
axb = fig.add_subplot(gs[0, 0]); t = np.arange(len(x)) / 16000
env = np.abs(x); axb.fill_between(t, -env, env, color=STYLE["ink"], lw=0, alpha=0.85); axb.set_xlim(0, t[-1]); axb.axis("off")
axc = fig.add_subplot(gs[1, 0], sharex=axb); show_spec(axc, spec)
for k, c in enumerate(["#FFFFFF", "#A8E0D6", "#C9B8E8"]):
    axc.plot(ft, F[k] / 1000, ".", ms=1.0, color=c, mew=0)
for k, (nm, c) in enumerate((("F1", "#FFFFFF"), ("F2", "#A8E0D6"), ("F3", "#C9B8E8"))):
    axc.text(0.89 + 0.035 * k, 0.94, nm, transform=axc.transAxes, ha="left", va="top", color=c, fontsize=STYLE["small"],
             fontweight="bold")
axc.tick_params(labelbottom=False); axc.set_xlabel("")
axd = fig.add_subplot(gs[2, 0], sharex=axb); te = np.arange(len(E)) / FS_EMA
for k, a in enumerate(["TT", "TB", "TD", "LI", "LL", "UL"]):
    yv = zs(E[:, CI_[f"{a}Y"]]) * 0.3 - k
    axd.fill_between(te, -k, yv, color=ACOL[a], alpha=0.12, lw=0)
    axd.plot(te, yv, color=ACOL[a], lw=0.7)
    axd.text(-0.012, -k, ART[a], color=ACOL[a], ha="right", va="center", fontsize=STYLE["small"],
             transform=axd.get_yaxis_transform())
axd.set_yticks([]); axd.spines["left"].set_visible(False); axd.set_xlabel("Time (s)"); axd.set_ylim(-5.9, 0.9)
axe = fig.add_subplot(gs[:, 1]); draw_mouth(axe, labels=True)
vmax = 0
for a in ANCHOR:
    xy = E[:, [CI_[f"{a}X"], CI_[f"{a}Y"]]] * SCALE
    vmax = max(vmax, np.percentile(np.hypot(*np.diff(xy, axis=0).T), 98))
for a, (cx, cy) in ANCHOR.items():
    xy = E[:, [CI_[f"{a}X"], CI_[f"{a}Y"]]]; d = (xy - xy.mean(0)) * SCALE
    lc = speed_line(axe, cx + d[:, 0], cy + d[:, 1], cmap=LinearSegmentedColormap.from_list("spd", ["#14244F", "#2B4C8C", "#5B54A8", "#8E3B7E", "#B22234"]), lw=0.55, vmax=vmax)
cax = axe.inset_axes([0.60, 0.07, 0.32, 0.025])
cb = fig.colorbar(lc, cax=cax, orientation="horizontal"); cb.set_ticks([]); cb.outline.set_linewidth(0.3)
cax.set_title("movement speed", fontsize=STYLE["small"] - 0.4, pad=1.5)
cax.text(0, -1.4, "slow", transform=cax.transAxes, fontsize=STYLE["small"] - 0.6, ha="left", va="top")
cax.text(1, -1.4, "fast", transform=cax.transAxes, fontsize=STYLE["small"] - 0.6, ha="right", va="top")
fig.text(0.004, 0.996, "a", fontsize=STYLE["letter"], fontweight="bold", va="top")
for axx, s in ((axb, "b"), (axc, "c"), (axd, "d")):
    fig.text(0.008, axx.get_position().y1 - 0.004, s, fontsize=STYLE["letter"], fontweight="bold", va="top")
fig.text(axe.get_position().x0 + 0.005, 0.62, "e", fontsize=STYLE["letter"], fontweight="bold", va="top")
save(fig, "Fig1_overview_voice_to_mouth"); plt.show()
''')

md("""
## Figure 2 - The network's movements match real articulograph sensors; vowel postures are preserved in PD
**a-b** a typical MOCHA-TIMIT sentence (median agreement): spectrogram, real sensor trace (grey band) and network
estimate (colour); **c** frame-by-frame agreement over all sentences per channel (violin, interquartile bar, median);
**d** vowel tongue shapes of healthy speakers (solid) and patients (dashed) overlap: static postures are preserved;
**e** vowel identified from posture alone (bubble area = proportion); **f** agreement of every movement summary with the
real sensors (filled = median of the three MOCHA speakers, open = worst speaker; bold, ringed = passes the pre-specified
rule, median rho >= 0.5: 5 of 15 measures; lip-opening speed was not among the 10 measures of the main analysis and
shows no group difference; per-speaker values in Supplementary Fig. S5); **g** every sentence of the four validated
measures used in the main analysis: real sensor vs network estimate (z within speaker; hexagon shade = number of sentences, line = fit).
""")

code(r'''
apply_style()
M = load_npz("mocha_example.npz"); FR = load("mocha_frame_correlations.csv"); MS = load("mocha_sentence_summaries.csv")
P = load("pilot_ema_recordings.csv"); CM = load("pilot_vowel_confusion_multi.csv").set_index("vowel")
SV = load("summary_measure_validity.csv"); SVS = load("summary_measure_validity_per_speaker.csv")
VCOL = {"a": "#B22234", "e": "#7B6FA6", "i": "#3A7CA5", "o": "#1B2333", "u": "#2A9D8F"}
fig = plt.figure(figsize=(STYLE["w2"], 232 * MM))
outer = fig.add_gridspec(3, 1, left=0.11, right=0.985, bottom=0.045, top=0.955, height_ratios=[1.3, 1.08, 0.62], hspace=0.27)
top = outer[0].subgridspec(1, 2, width_ratios=[1.7, 1], wspace=0.22)
left = top[0].subgridspec(5, 1, height_ratios=[0.8, 1, 1, 1, 1], hspace=0.14)
axa = fig.add_subplot(left[0]); show_spec(axa, M["spec"], fmax=4000); axa.tick_params(labelbottom=False)
axa.set_title("MOCHA-TIMIT sentence:  grey band = real sensor (EMA),  colour = network estimate from audio", loc="left",
              fontsize=STYLE["small"], color=STYLE["ink2"])
te = np.arange(M["real"].shape[0]) / FS_EMA
axes_b = []
for k, (c, a) in enumerate((("TTY", "TT"), ("TBY", "TB"), ("LIY", "LI"), ("LLY", "LL"))):
    ax = fig.add_subplot(left[k + 1], sharex=axa); axes_b.append(ax)
    ax.fill_between(te, M["real"][:, CI_[c]] - 0.2, M["real"][:, CI_[c]] + 0.2, color="#CFCFCF", lw=0)
    ax.plot(te, M["est"][:, CI_[c]], color=ACOL[a], lw=0.9)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlim(0, te[-1])
    ax.text(-0.012, 0.5, ART[a], transform=ax.transAxes, ha="right", va="center", color=ACOL[a], fontsize=STYLE["small"],
            fontweight="bold")
    ax.text(1.0, 0.98, f"r = {M['r'][CI_[c]]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=STYLE["small"],
            color=STYLE["ink2"])
    if k < 3:
        ax.tick_params(labelbottom=False, length=0); ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel("Time (s)")
right = top[1].subgridspec(2, 1, height_ratios=[0.9, 1.1], hspace=0.38)
axc = fig.add_subplot(right[0]); ygrid(axc)
order = ["TTY", "TBY", "TDY", "LIY", "LLY", "ULY", "TTX", "TBX", "TDX", "LIX", "LLX", "ULX"]
data = [FR[FR.channel == c].r.dropna().clip(-1, 1).to_numpy() for c in order]
vp = axc.violinplot(data, positions=range(len(order)), widths=0.85, showextrema=False, showmedians=False)
for body, c in zip(vp["bodies"], order):
    body.set_facecolor(ACOL[c[:2]]); body.set_alpha(0.55); body.set_edgecolor("none")
for i, d in enumerate(data):
    q1, md_, q3 = np.percentile(d, [25, 50, 75])
    axc.plot([i, i], [q1, q3], color=STYLE["ink"], lw=1.6, solid_capstyle="butt", zorder=3)
    axc.scatter(i, md_, s=7, color="white", edgecolor=STYLE["ink"], lw=0.5, zorder=4)
axc.set_xticks(range(len(order))); axc.set_xticklabels([c[:2] + ("v" if c[2] == "Y" else "h") for c in order], rotation=90)
for tl, c in zip(axc.get_xticklabels(), order):
    tl.set_color(ACOL[c[:2]])
axc.set_ylim(-0.2, 1.0); axc.set_ylabel("Frame agreement (r)"); axc.axhline(0, color=STYLE["grey"], lw=0.4)
axc.text(0.0, 1.03, f"{FR.groupby(['speaker', 'sentence']).ngroups:,} sentences; median r = {FR.r.median():.2f} (v vertical, h horizontal)",
         transform=axc.transAxes, fontsize=STYLE["small"] - 0.4, color=STYLE["ink2"], va="bottom")
axd = fig.add_subplot(right[1]); draw_mouth(axd, tongue=False)
V = P[(P.model == "multi") & (P.kind == "vowel") & (P.cond == "clean")].copy()
V[CHN] = V[CHN] - V.groupby("sid")[CHN].transform("mean")
for lb, ls, lw in ((0, "-", 1.7), (1, (0, (2.2, 1.2)), 1.2)):
    VM = V[V.label == lb].groupby("vowel")[CHN].mean()
    for vw in "aeiou":
        p = {a: (ANCHOR[a][0] + 1.15 * VM.loc[vw, f"{a}X"], ANCHOR[a][1] + 1.15 * VM.loc[vw, f"{a}Y"]) for a in ANCHOR}
        tg = smooth([p["TD"], p["TB"], p["TT"]], 80)
        axd.plot(tg[:, 0], tg[:, 1], color=VCOL[vw], lw=lw, ls=ls, solid_capstyle="round")
for j, vw in enumerate("aeiou"):
    axd.text(0.03 + 0.11 * j, 0.02, f"/{vw}/", transform=axd.transAxes, color=VCOL[vw], fontsize=STYLE["size"], fontweight="bold")
axd.plot([], [], color=STYLE["ink2"], lw=1.7, label="Healthy"); axd.plot([], [], color=STYLE["ink2"], lw=1.2, ls=(0, (2.2, 1.2)), label="PD")
axd.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02))
row = outer[1].subgridspec(1, 2, width_ratios=[0.7, 1.3], wspace=0.62)
axe = fig.add_subplot(row[0])
for i, v in enumerate("aeiou"):
    for j, u in enumerate("aeiou"):
        val = CM.loc[v, u]
        axe.scatter(j, i, s=560 * val + 2, color=STYLE["passed"] if i == j else "#A7A7A7", alpha=0.9 if i == j else 0.65,
                    edgecolor="white", lw=0.5)
        if val >= 0.05:
            axe.text(j, i, f"{val:.2f}".lstrip("0"), ha="center", va="center", fontsize=STYLE["small"] - 0.5,
                     color="white" if i == j else STYLE["ink"])
axe.set_xticks(range(5)); axe.set_xticklabels([f"/{v}/" for v in "aeiou"]); axe.set_yticks(range(5))
axe.set_yticklabels([f"/{v}/" for v in "aeiou"]); axe.set_xlim(-0.6, 4.6); axe.set_ylim(4.6, -0.6)
axe.set_xlabel("Identified from posture"); axe.set_ylabel("Spoken vowel"); axe.tick_params(length=0)
for s_ in axe.spines.values(): s_.set_visible(False)
axe.set_title(f"Accuracy {np.mean(np.diag(CM.values)):.0%} (chance 20%)", loc="left", fontsize=STYLE["small"])
# f: validity per measure (median of the three sensor speakers, line to the worst speaker)
axf = fig.add_subplot(row[1]); ygrid(axf, "x")
SVi = SV.set_index("measure"); meas = SV.sort_values("median_spearman", ascending=False).measure.tolist()
axf.axvspan(0.5, 0.75, color=STYLE["passed"], alpha=0.07, lw=0)
for i, m in enumerate(meas):
    c = ACOL[m.split("_")[0]]; md_, mn_ = SVi.loc[m, "median_spearman"], SVi.loc[m, "min_spearman"]
    axf.plot([mn_, md_], [i, i], color=c, lw=1.4, alpha=0.45, solid_capstyle="round")
    axf.scatter(mn_, i, s=12, facecolor="white", edgecolor=c, lw=0.8, zorder=3)
    axf.scatter(md_, i, s=26, color=c, edgecolor=STYLE["ink"] if SVi.loc[m, "passes"] else "white", lw=0.8 if SVi.loc[m, "passes"] else 0.4,
                zorder=4)
    axf.text(0.745, i, f"{md_:.2f}", ha="right", va="center", fontsize=STYLE["small"] - 0.7,
             color=STYLE["ink"] if SVi.loc[m, "passes"] else STYLE["grey"], fontweight="bold" if SVi.loc[m, "passes"] else "normal")
axf.axvline(0.5, color=STYLE["ink"], lw=0.6, ls=(0, (2, 2)))
axf.set_yticks(range(len(meas))); axf.set_yticklabels([label(m) for m in meas]); axf.set_ylim(len(meas) - 0.4, -0.9)
for tl, m in zip(axf.get_yticklabels(), meas):
    tl.set_fontweight("bold" if SVi.loc[m, "passes"] else "normal"); tl.set_color(ACOL[m.split("_")[0]])
axf.set_xlim(0, 0.75); axf.set_xlabel("Agreement with real sensors (Spearman rho per sentence)")
axf.tick_params(axis="y", length=0); axf.spines["left"].set_visible(False)
axf.text(0.505, -0.75, "pre-specified pass", fontsize=STYLE["small"] - 0.6, color=STYLE["ink2"], va="center")
axf.scatter([], [], s=24, color=STYLE["ink2"], label="median of 3 speakers"); axf.scatter([], [], s=11, facecolor="white", edgecolor=STYLE["ink2"], lw=0.8, label="worst speaker")
axf.legend(loc="lower left", fontsize=STYLE["small"] - 0.6, handletextpad=0.2)
# g: sentence-level hexbins
r3 = outer[2].subgridspec(1, 4, wspace=0.32)
axes_g = []
for k, m in enumerate(["TT_range", "TB_space", "LL_range", "LL_speed"]):
    ax = fig.add_subplot(r3[k]); axes_g.append(ax)
    zx = MS.groupby("speaker")[f"real_{m}"].transform(lambda v: (v - v.mean()) / v.std())
    zy = MS.groupby("speaker")[f"est_{m}"].transform(lambda v: (v - v.mean()) / v.std())
    ok = np.isfinite(zx) & np.isfinite(zy)
    ax.hexbin(zx[ok], zy[ok], gridsize=20, extent=(-3, 3, -3, 3), cmap=HEX_G, mincnt=1, linewidths=0.15, edgecolors="white")
    b = np.polyfit(zx[ok], zy[ok], 1); xx = np.array([-2.6, 2.6])
    ax.plot(xx, np.polyval(b, xx), color=STYLE["pd"], lw=1.0)
    ax.plot([-3, 3], [-3, 3], color=STYLE["grey"], lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.set_title(label(m), fontsize=STYLE["small"], color=ACOL[m.split("_")[0]], fontweight="bold")
    ax.text(0.04, 0.96, f"rho = {SVi.loc[m, 'median_spearman']:.2f}", transform=ax.transAxes, va="top", fontsize=STYLE["small"] - 0.3)
    ax.set_xlabel("Real sensor (z)")
    if k == 0:
        ax.set_ylabel("Network (z)")
    else:
        ax.tick_params(labelleft=False)
for ax_, s in ((axa, "a"), (axes_b[0], "b"), (axc, "c"), (axd, "d"), (axe, "e"), (axf, "f"), (axes_g[0], "g")):
    letter(fig, ax_, s)
save(fig, "Fig2_validation"); plt.show()
''')

md("""
## Figure 3 - The mouth moves less in Parkinson's disease
**a** where movement is lost: difference in movement density, Parkinson's minus healthy (speaker-centred positions
during reading, both datasets; dotted = region holding 90% of healthy movement); navy = space healthy speakers use that
patients do not reach, crimson = movement concentrated near the centre; **b** the same as a profile: time patients spend at
each distance from their average position, relative to healthy speakers (every articulator: more time near the centre,
fewer large excursions); **c** all measures, pooled over both languages (diamond, 95% CI drawn as density; open markers =
English and Italian estimates; number = pooled difference in SD), with FDR stars (bold = sensor-validated); **d** the four
sensor-validated measures per speaker (z vs healthy of the same dataset and sex; n healthy v PD under each language).
""")

code(r'''
apply_style()
CL = load_npz("movement_clouds.npz"); grid = CL["grid"]; ctr = (grid[:-1] + grid[1:]) / 2
X = load_npz("example_reading.npz")
PO = load("reading_pooled_EN_IT.csv").set_index("measure"); E = load("mouth_bradykinesia_effects.csv", "20_")
S = load("speaker_table_dl_and_classic.csv")
fig = plt.figure(figsize=(STYLE["w2"], 200 * MM))
outer = fig.add_gridspec(2, 1, left=0.075, right=0.985, bottom=0.075, top=0.975, height_ratios=[1.0, 1.15], hspace=0.22)
r1 = outer[0].subgridspec(1, 2, width_ratios=[1.3, 1], wspace=0.22)
axa = fig.add_subplot(r1[0]); draw_mouth(axa, tongue=False, fill="#F2F4F7")
D, HS, vlim = density_difference(CL, ctr)
for a, (cx, cy) in ANCHOR.items():
    gx = cx + ctr * SCALE * 0.95; gy = cy + ctr * SCALE * 0.95
    im = axa.pcolormesh(gx, gy, D[a].T, cmap=DIV, vmin=-vlim, vmax=vlim, shading="gouraud", zorder=4, rasterized=True)
    axa.contour(gx, gy, HS[a].T, levels=[hdr_level(HS[a], 0.9)], colors=[STYLE["ink2"]], linewidths=0.45, linestyles=":", zorder=5)
lab_off = {"UL": (-1.45, 0.75), "LL": (-1.45, -0.75), "LI": (0.0, -1.35), "TT": (0.0, 1.32), "TB": (0.0, 1.32), "TD": (0.0, 1.32)}
for a, (cx, cy) in ANCHOR.items():
    dx, dy = lab_off[a]
    axa.text(cx + dx, cy + dy, ART[a], ha="center", va="center", fontsize=STYLE["small"] - 0.3, color=ACOL[a], zorder=8,
             fontweight="bold")
cax = axa.inset_axes([0.60, 0.06, 0.34, 0.03])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_ticks([-vlim * 0.8, 0, vlim * 0.8])
cb.set_ticklabels(["patients\nreach less", "0", "patients\nmore"]); cb.outline.set_linewidth(0.3)
cax.tick_params(labelsize=STYLE["small"] - 0.8, length=1.5, pad=1)
cax.set_title("movement density, PD - healthy", fontsize=STYLE["small"] - 0.5, pad=2)
# ---- b: excursion-loss profile (share of time at each distance from the speaker's own average position)
axb = fig.add_subplot(r1[1])
XX, YY = np.meshgrid(ctr, ctr, indexing="ij"); RR = np.hypot(XX, YY)
rb = np.linspace(0, 2.85, 17); rc = (rb[:-1] + rb[1:]) / 2
axb.axhspan(0, 60, color="#F4F4F5", lw=0); axb.axhspan(-60, 0, color="#EDF1F6", lw=0)
allpr = []
for a in ["TT", "TB", "TD", "LI", "LL", "UL"]:
    lr = []
    for coh in ("MDVR", "IPVS"):
        h0, h1 = group_density(CL, coh, 0, a), group_density(CL, coh, 1, a)
        s0 = np.array([h0[(RR >= lo) & (RR < hi)].sum() for lo, hi in zip(rb[:-1], rb[1:])])
        s1 = np.array([h1[(RR >= lo) & (RR < hi)].sum() for lo, hi in zip(rb[:-1], rb[1:])])
        lr.append(np.log((s1 + 1e-4) / (s0 + 1e-4)))
    pr = 100 * (np.exp(np.mean(lr, 0)) - 1); allpr.append(pr)
    xs_ = np.linspace(rc[0], rc[-1], 120); ys_ = make_interp_spline(rc, pr, k=3)(xs_)
    axb.plot(xs_, ys_, color=ACOL[a], lw=1.3, label=ART[a], solid_capstyle="round")
    axb.scatter(rc, pr, s=5, color=ACOL[a], zorder=3, lw=0)
axb.axhline(0, color=STYLE["ink2"], lw=0.6)
axb.set_xlim(0, 2.9); axb.set_ylim(-55, 45)
axb.set_xlabel("Distance from the speaker's average position (a.u.)")
axb.set_ylabel("Time spent there, Parkinson's vs healthy (%)")
allpr = np.array(allpr); near_, far_ = allpr[:, rc < 0.55].mean(), allpr[:, rc > 2.2].mean()
axb.text(0.04, 0.96, f"patients stay near the centre\n{near_:+.0f}% time within 0.55", transform=axb.transAxes, va="top", fontsize=STYLE["small"], color=STYLE["pd"],
         fontweight="bold")
axb.text(0.96, 0.04, f"large excursions are lost\n{far_:+.0f}% time beyond 2.2", transform=axb.transAxes, va="bottom", ha="right", fontsize=STYLE["small"],
         color=STYLE["healthy"], fontweight="bold")
axb.legend(loc="lower left", ncol=2, columnspacing=0.8, handlelength=1.2)
r2 = outer[1].subgridspec(1, 3, width_ratios=[0.16, 1.0, 1.1], wspace=0.35)
subd = r2[2].subgridspec(2, 2, wspace=0.42, hspace=0.62)
feats = [("TT_range", "Tongue-tip\nrange"), ("TB_space", "Tongue\nworking space"), ("LI_range", "Jaw\nrange"), ("LL_speed", "Lower-lip\nspeed")]
Z = S.copy()
for coh in Z.cohort.unique():
    for mv in (0.0, 1.0):
        ref = Z[(Z.cohort == coh) & (Z.label == 0) & (Z.male == mv)]
        if len(ref) < 4:
            ref = Z[(Z.cohort == coh) & (Z.label == 0)]
        msk = (Z.cohort == coh) & (Z.male == mv)
        for f_, _ in feats:
            Z.loc[msk, f_ + "_z"] = (Z.loc[msk, f_] - ref[f_].mean()) / ref[f_].std()
axes_d = []
for k, (f_, nm) in enumerate(feats):
    ax = fig.add_subplot(subd[k // 2, k % 2]); axes_d.append(ax)
    data, pos, cols = [], [], []
    for coh, base in (("MDVR reading", 0.0), ("IPVS reading", 2.1)):
        for lb, off in ((0, 0.0), (1, 0.85)):
            data.append(Z[(Z.cohort == coh) & (Z.label == lb)][f_ + "_z"].dropna()); pos.append(base + off)
            cols.append(STYLE["healthy"] if lb == 0 else STYLE["pd"])
    raincloud(ax, data, pos, cols, width=0.30, s=3.2, seed=k * 10)
    ax.axhline(0, color=STYLE["grey"], lw=0.4, ls=":", zorder=0)
    lo, hi = np.nanpercentile(pd.concat(data), [1, 99]); ax.set_ylim(lo - 0.6, hi + 1.5)
    for sec, base in (("B MDVR reading", 0.0), ("B IPVS reading", 2.1)):
        e = E[(E.section == sec) & (E.measure == f_)]
        if len(e):
            bracket(ax, base, base + 0.85, hi + 0.45, stars(e.q.iloc[0]), dy=0.25)
    ns_ = [len(d) for d in data]
    ax.set_xticks([0.42, 2.52]); ax.set_xticklabels([f"EN\n{ns_[0]} v {ns_[1]}", f"IT\n{ns_[2]} v {ns_[3]}"]); ax.set_xlim(-0.55, 3.35)
    ax.set_title(nm, fontsize=STYLE["small"], pad=6)
    if k % 2 == 0:
        ax.set_ylabel("z (vs healthy)")
axes_d[3].scatter([], [], s=8, color=STYLE["healthy"], label="Healthy"); axes_d[3].scatter([], [], s=8, color=STYLE["pd"], label="PD")
axes_d[3].legend(loc="upper right", ncol=2, bbox_to_anchor=(1.0, -0.3))
axe = fig.add_subplot(r2[1])
fam = [("Tongue", ["TT_range", "TT_speed", "TB_range", "TB_speed", "TB_space", "TD_range", "TD_speed"]), ("Jaw", ["LI_range", "LI_speed"]),
       ("Lips", ["LL_range", "LL_speed", "UL_range", "UL_speed", "LA_range", "LA_speed"])]
yk, heads, ycur = [], [], 0.0
for fname, ms in fam:
    heads.append((ycur, fname)); ycur -= 0.9
    for m in ms:
        yk.append((ycur, m)); ycur -= 0.82
    ycur -= 0.25
for y0, m in yk:
    if m not in PO.index:
        continue
    p = PO.loc[m]; col = ACOL[m.split("_")[0]]
    eye(axe, y0, p.beta, p.ci_lo, p.ci_hi, col, h=0.30, alpha=0.40)
    for sec_, mk_, cc_ in (("B MDVR reading", "o", STYLE["english"]), ("B IPVS reading", "s", STYLE["italian"])):
        e_ = E[(E.section == sec_) & (E.measure == m)]
        if len(e_):
            axe.scatter(e_.beta.iloc[0], y0, s=8, marker=mk_, facecolor="white", edgecolor=cc_, lw=0.7, zorder=4)
    axe.scatter(p.beta, y0, s=20, marker="D", color=col, edgecolor="white", lw=0.4, zorder=5)
    axe.text(0.92, y0, f"{p.beta:.2f}", transform=axe.get_yaxis_transform(), ha="right", va="center", fontsize=STYLE["small"] - 0.8,
             color=STYLE["ink2"])
    axe.text(1.0, y0, stars(p.q), transform=axe.get_yaxis_transform(), ha="right", va="center", fontsize=STYLE["small"] - 0.3,
             color=STYLE["ink"] if p.q < 0.05 else STYLE["grey"])
axe.axvline(0, color=STYLE["grey"], lw=0.5); ygrid(axe, "x")
axe.scatter([], [], s=14, marker="D", color=STYLE["ink2"], label="Pooled")
axe.scatter([], [], s=9, marker="o", facecolor="white", edgecolor=STYLE["english"], lw=0.7, label="English")
axe.scatter([], [], s=9, marker="s", facecolor="white", edgecolor=STYLE["italian"], lw=0.7, label="Italian")
axe.legend(loc="lower left", fontsize=STYLE["small"] - 0.6, handletextpad=0.1)
axe.set_yticks([y for y, _ in yk])
axe.set_yticklabels([label(m).replace("Tongue ", "").replace("Lip opening", "Lip open.") for _, m in yk], fontsize=STYLE["small"] - 0.3)
for tl, (_, m) in zip(axe.get_yticklabels(), yk):
    tl.set_fontweight("bold" if m in VALIDATED else "normal")
for y0, fname in heads:
    axe.text(0.0, y0, fname, transform=axe.get_yaxis_transform(), ha="left", va="center", fontsize=STYLE["small"], fontweight="bold")
axe.set_ylim(ycur + 0.2, 0.6); axe.set_xlim(-1.8, 0.75); axe.set_xlabel("Parkinson's - healthy (SD, pooled, 95% CI)")
axe.spines["left"].set_visible(False); axe.tick_params(axis="y", length=0)
for ax_, s in ((axa, "a"), (axb, "b"), (axe, "c"), (axes_d[0], "d")):
    letter(fig, ax_, s)
save(fig, "Fig3_hypokinesia"); plt.show()
''')

md("""
## Figure 4 - The less the tongue moves, the worse the speech
**a** tongue-body movement of every English patient (1, 1.2 and 1.4 SD envelopes), ordered by the patient-reported
speech score (UPDRS II item 5, speech in daily life); dotted outline = median healthy speaker, number = area as % of healthy;
**b-c** tongue working space against the speech score (mean +/- SE; healthy reference band) and the examiner-rated speech item of the motor examination (UPDRS III item 18; least squares
with 95% bootstrap band; Hoehn & Yahr in Supplementary Fig. S6); **d** every movement measure against every severity
scale, before and after removing speech rate (bubble area = |rho|, colour = sign, ring = p < 0.05); **e** severity correlations of the
deep-learning measures and the classic acoustic measures, ranked; open circle = DL composite with speech rate partialled out.
""")

code(r'''
apply_style()
EL = load("movement_ellipses.csv"); S = load("speaker_table_dl_and_classic.csv"); H = load("h2h_severity.csv")
SEV = load("mouth_severity.csv", "20_")
fig = plt.figure(figsize=(STYLE["w2"], 200 * MM))
outer = fig.add_gridspec(3, 1, left=0.085, right=0.985, bottom=0.085, top=0.975, height_ratios=[0.27, 0.9, 1.15], hspace=0.28)
axa = fig.add_subplot(outer[0])
m = EL[EL.cohort == "MDVR"]; hc = m[m.label == 0]
wr, hr, ar = ell_from_cov(hc.TB_cxx.median(), hc.TB_cxy.median(), hc.TB_cyy.median())
pdx = m[(m.label == 1) & m.updrs2.notna()].sort_values(["updrs2", "TB_space"], ascending=[True, False]).reset_index(drop=True)
glow_ellipse(axa, (0, 0), wr, hr, ar, H_, alphas=(0.30, 0.12, 0.05))
axa.add_patch(Ellipse((0, 0), wr, hr, angle=ar, fill=False, ec=H_, ls=(0, (3, 1.5)), lw=0.9))
axa.text(0, 0, "100%", ha="center", va="center", fontsize=STYLE["small"] - 0.6, color=H_, fontweight="bold", zorder=7)
x, last, xpos = 3.1, None, []
for _, r in pdx.iterrows():
    if last is not None and r.updrs2 != last:
        x += 1.2
    w_, h_, a_ = ell_from_cov(r.TB_cxx, r.TB_cxy, r.TB_cyy)
    glow_ellipse(axa, (x, 0), w_, h_, a_, SEVC[int(r.updrs2)])
    axa.add_patch(Ellipse((x, 0), wr, hr, angle=ar, fill=False, ec=STYLE["ink2"], ls=":", lw=0.45, zorder=5))
    axa.text(x, 0, f"{r.TB_space / hc.TB_space.median():.0%}", ha="center", va="center", fontsize=STYLE["small"] - 0.6,
             color="white" if r.updrs2 >= 2 else STYLE["ink"], fontweight="bold", zorder=7)
    xpos.append((x, int(r.updrs2))); last = r.updrs2; x += 2.35
ylo = -max(hr, wr) * 0.62
axa.text(0, ylo - 0.25, "Healthy\n(median)", ha="center", va="top", fontsize=STYLE["small"], color=H_)
for sc in sorted({v for _, v in xpos}):
    xs_ = [p for p, v in xpos if v == sc]
    axa.plot([min(xs_) - 0.8, max(xs_) + 0.8], [ylo, ylo], color=SEVC[sc], lw=1.8, solid_capstyle="round")
    axa.text(np.mean(xs_), ylo - 0.25, f"Speech score {sc}", ha="center", va="top", fontsize=STYLE["small"],
             color=SEVC[max(sc, 1)], fontweight="bold")
yhi = max(hr, wr) * 0.62
axa.add_patch(FancyArrowPatch((3.0, yhi), (x - 1.6, yhi), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=STYLE["grey"]))
axa.text((3.0 + x - 1.6) / 2, yhi + 0.12, "worse patient-reported speech", ha="center", va="bottom", fontsize=STYLE["small"] - 0.2,
         color=STYLE["ink2"])
axa.set_xlim(-1.6, x + 0.1); axa.set_ylim(ylo - 0.95, yhi + 0.5); axa.set_aspect("equal", adjustable="datalim"); axa.axis("off")
pdm = S[(S.cohort == "MDVR reading") & (S.label == 1)]; hcm = S[(S.cohort == "MDVR reading") & (S.label == 0)]
r2 = outer[1].subgridspec(1, 2, wspace=0.28)
rng = np.random.default_rng(4)
axb = fig.add_subplot(r2[0]); ygrid(axb)
hm, hs_ = hcm.TB_space.mean(), hcm.TB_space.std()
axb.axhspan(hm - hs_, hm + hs_, color=H_, alpha=0.10, lw=0); axb.axhline(hm, color=H_, lw=0.7, ls="--")
axb.text(3.45, hm + hs_ * 0.95, "healthy\nmean +/- SD", fontsize=STYLE["small"] - 0.5, color=H_, ha="right", va="top")
means = []
for sc in range(4):
    v = pdm[pdm.updrs2 == sc].TB_space.dropna()
    if len(v):
        axb.scatter(sc + rng.uniform(-0.12, 0.12, len(v)), v, s=16, color=SEVC[sc], edgecolor=STYLE["ink2"], lw=0.3, zorder=3)
        se = v.std() / np.sqrt(len(v)) if len(v) > 1 else 0
        axb.errorbar(sc + 0.27, v.mean(), yerr=se, fmt="o", ms=3.5, color=STYLE["ink"], elinewidth=0.8, capsize=1.5, zorder=4)
        means.append((sc + 0.27, v.mean()))
axb.plot(*zip(*means), color=STYLE["ink"], lw=0.6, ls=(0, (2, 1.5)), zorder=3)
axb.text(0.03, 0.04, rho_txt(pdm.updrs2, pdm.TB_space), transform=axb.transAxes, fontsize=STYLE["small"] - 0.2)
axb.set_xticks(range(4)); axb.set_xlabel("UPDRS II-5: speech (daily life)"); axb.set_ylabel("Tongue working space"); axb.set_xlim(-0.4, 3.5)
def sev_scatter(ax, sc, meas, xlab, ylab=None):
    d = pdm[[meas, sc, "updrs2"]].dropna(); xs = np.linspace(d[sc].min(), d[sc].max(), 50)
    fits = []
    for _ in range(600):
        i = rng.integers(0, len(d), len(d))
        if d[sc].iloc[i].nunique() > 1:
            fits.append(np.polyval(np.polyfit(d[sc].iloc[i], d[meas].iloc[i], 1), xs))
    lo, hi = np.percentile(fits, [2.5, 97.5], axis=0)
    ax.fill_between(xs, lo, hi, color=STYLE["grey"], alpha=0.25, lw=0)
    ax.plot(xs, np.polyval(np.polyfit(d[sc], d[meas], 1), xs), color=P_, lw=1.0)
    ax.scatter(d[sc] + rng.uniform(-0.08, 0.08, len(d)), d[meas], s=16, c=[SEVC[int(v)] for v in d.updrs2], edgecolor=STYLE["ink2"],
               lw=0.3, zorder=3)
    ax.text(0.97, 0.97, rho_txt(d[sc], d[meas]), transform=ax.transAxes, ha="right", va="top", fontsize=STYLE["small"] - 0.2)
    ax.set_xlabel(xlab); ygrid(ax)
    if ylab:
        ax.set_ylabel(ylab)
axc = fig.add_subplot(r2[1]); sev_scatter(axc, "updrs3", "TB_space", "UPDRS III-18: speech (examiner)")
for sc in range(4):
    axc.scatter([], [], s=14, color=SEVC[sc], edgecolor=STYLE["ink2"], lw=0.3, label=f"speech score {sc}")
axc.legend(loc="lower left", ncol=2, fontsize=STYLE["small"] - 0.6, columnspacing=0.5, handletextpad=0.1)
r3 = outer[2].subgridspec(1, 3, width_ratios=[0.1, 1.15, 1], wspace=0.5)
axe = fig.add_subplot(r3[1])
scales = [("hy", "Hoehn &\nYahr"), ("updrs3", "UPDRS III-18\nspeech, exam"), ("updrs2", "UPDRS II-5\nspeech, daily")]
srt = SEV[SEV.scale == "updrs2"].set_index("measure").rho.reindex(DLM).sort_values().index.tolist()
xcol = []
for j, (sc, _) in enumerate(scales):
    xcol += [(j * 2.6, sc, "raw"), (j * 2.6 + 1, sc, "adj")]
for i, mm in enumerate(srt):
    axe.axhline(i, color=STYLE["grid"], lw=0.4, zorder=0)
    for x0, sc, kind in xcol:
        r_ = SEV[(SEV.scale == sc) & (SEV.measure == mm)].iloc[0]
        rho, p = (r_.rho, r_.p) if kind == "raw" else (r_.rho_adj_rate, r_.p_adj_rate)
        axe.scatter(x0, i, s=150 * abs(rho) + 1, color=DIV(0.5 + rho / 2), edgecolor=STYLE["ink"] if p < 0.05 else "none",
                    lw=0.7, zorder=3)
axe.set_yticks(range(len(srt))); axe.set_yticklabels([label(mm) for mm in srt])
for tl, mm in zip(axe.get_yticklabels(), srt):
    tl.set_color(ACOL[mm.split("_")[0]]); tl.set_fontweight("bold" if mm in VALIDATED else "normal")
axe.set_xticks([x0 for x0, _, _ in xcol]); axe.set_xticklabels(["raw", "rate\nadj."] * 3, fontsize=STYLE["small"] - 0.4)
axe.set_xlim(-0.7, 6.9); axe.set_ylim(len(srt) - 0.4, -0.6); axe.tick_params(length=0)
for s_ in axe.spines.values(): s_.set_visible(False)
for j, (_, nm) in enumerate(scales):
    axe.text(j * 2.6 + 0.5, -1.0, nm, ha="center", va="bottom", fontsize=STYLE["small"], fontweight="bold")
    axe.add_patch(Rectangle((j * 2.6 - 0.5, -0.6), 2.0, len(srt), fc=STYLE["panel"], ec="none", zorder=0))
cax = axe.inset_axes([0.25, -0.115, 0.5, 0.02])
cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(-1, 1), DIV), cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.3)
cb.set_ticks([-1, -0.5, 0, 0.5, 1]); cax.tick_params(labelsize=STYLE["small"] - 0.7, length=1.5, pad=1)
cax.set_xlabel("Spearman rho with severity", fontsize=STYLE["small"] - 0.3, labelpad=1)
axf = fig.add_subplot(r3[2]); ygrid(axf, "x")
NM = {"dl_composite": "DL composite", "TB_space": "Tongue space", "TT_range": "Tongue-tip range", "speech_rate_syl_s": "Speech rate",
      "pause_ratio": "Pause ratio", "f0_sd_st": "Pitch SD", "F1F2_space": "F1-F2 space", "F2_speed": "F2 speed",
      "cons_centroid_hz": "Consonant centroid"}
for i, (sc, nm) in enumerate(scales):
    y0 = 2 - i
    h = H[H.scale == sc].set_index("measure")
    axf.add_patch(Rectangle((0, y0 - 0.38), 1, 0.76, fc=STYLE["panel"], ec="none", zorder=0))
    for mm in NM:
        if mm not in h.index:
            continue
        kd = h.loc[mm, "kind"]; v = abs(h.loc[mm, "rho"]); dl = kd == "deep learning"
        axf.scatter(v, y0 + (0.13 if dl else -0.13), s=26 if mm in ("dl_composite", "speech_rate_syl_s") else 13,
                    color=P_ if dl else H_, edgecolor="white", lw=0.4, zorder=4, alpha=1 if mm in ("dl_composite", "speech_rate_syl_s") else 0.7)
    pc = abs(h.loc["dl_composite | controlling speech_rate_syl_s", "rho"]); vc = abs(h.loc["dl_composite", "rho"])
    axf.annotate("", xy=(pc, y0 + 0.30), xytext=(vc, y0 + 0.30), arrowprops=dict(arrowstyle="-|>", color=P_, lw=0.6, mutation_scale=5))
    axf.scatter(pc, y0 + 0.30, s=16, facecolor="white", edgecolor=P_, lw=0.8, zorder=5)
bd = H[H.measure.str.startswith("|rho") & (H.scale == "updrs2")]
if len(bd):
    b_ = bd.iloc[0]
    axf.text(0.02, 2.62, f"UPDRS II: |rho| DL composite - |rho| speech rate = {b_.rho:+.2f} [{b_.ci_lo:.2f}, {b_.ci_hi:.2f}]", fontsize=STYLE["small"] - 0.6,
             color=STYLE["ink2"], va="bottom")
axf.set_yticks([2, 1, 0]); axf.set_yticklabels([n for _, n in scales], fontsize=STYLE["small"] - 0.3); axf.set_xlim(0, 1.0); axf.set_ylim(-1.75, 2.95)
axf.set_xlabel("|Spearman rho| with severity"); axf.spines["left"].set_visible(False); axf.tick_params(axis="y", length=0)
axf.scatter([], [], s=24, color=P_, label="Deep learning (large = composite)")
axf.scatter([], [], s=24, color=H_, label="Classic (large = speech rate)")
axf.scatter([], [], s=16, facecolor="white", edgecolor=P_, lw=0.8, label="DL composite, rate removed")
axf.legend(loc="lower left", ncol=1, fontsize=STYLE["small"] - 0.3)
for ax_, s in ((axa, "a"), (axb, "b"), (axc, "c"), (axe, "d"), (axf, "e")):
    letter(fig, ax_, s)
save(fig, "Fig4_severity"); plt.show()
''')

md("""
## Figure 5 - Two tremor sources: the mouth and the voice box
**a-b** jaw movement during a sustained /a/ (typical healthy vs typical Parkinson's, Italian) and its time-frequency map
(dashed = 3-7 Hz tremor band); **c** spectral fingerprint of every Italian speaker: jaw-movement spectrum relative to the
healthy average, speakers sorted by 3-7 Hz tremor share (left strip: navy healthy, crimson Parkinson's); **d** group
spectra (mean +/- SE) and their difference with 95% CI; black bars = frequencies where the groups differ (Welch t-test,
p < 0.05, uncorrected); **e** 3-7 Hz jaw tremor per speaker; **f** loudness tremor per speaker, phone recordings;
**g** mouth and voice-box tremor are unrelated within speakers; **h** tremor AUC for every site, dataset and recording band
(open = Italian vowels band-limited to telephone quality; black ring = q < 0.05): jaw tremor survives the phone band, and
voice loudness tremor goes in opposite directions in the two datasets.
""")

code(r'''
apply_style()
T = load_npz("tremor_spectra.npz"); IX = load("tremor_spectra_index.csv")
TI, TF = load("tremor_speakers_IPVS.csv"), load("tremor_speakers_FIGSHARE.csv"); E = load("mouth_bradykinesia_effects.csv", "20_")
f = T["freq"]
fig = plt.figure(figsize=(STYLE["w2"], 214 * MM))
outer = fig.add_gridspec(3, 1, left=0.07, right=0.985, bottom=0.055, top=0.95, height_ratios=[1.25, 1, 0.66], hspace=0.36)
top = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.30)
tops = []
for j, (lb, nm, col) in enumerate(((0, "Healthy", STYLE["healthy"]), (1, "Parkinson's", STYLE["pd"]))):
    key = f"ex_IPVS_{lb}_"
    sub = top[j].subgridspec(2, 1, height_ratios=[0.42, 1], hspace=0.05)
    a1 = fig.add_subplot(sub[0]); a2 = fig.add_subplot(sub[1], sharex=a1); tops.append(a1)
    sig = T[key + "sig"]; tt = np.arange(len(sig)) / float(T[key + "fs"])
    a1.fill_between(tt, sig.min(), sig, color=STYLE["grey"], alpha=0.2, lw=0); a1.plot(tt, sig, color=col, lw=0.55)
    a1.set_yticks([]); a1.spines["left"].set_visible(False); a1.tick_params(labelbottom=False); a1.set_xlim(0, tt[-1])
    a1.set_title(f"{nm}, sustained /a/", loc="left", color=col, fontweight="bold")
    tf, ff, tm = T[key + "tf"], T[key + "f"], T[key + "t"]; keep = (ff >= 1) & (ff <= 12)
    Zt = tf[keep] / (np.percentile(tf[keep], 99) + 1e-12)
    a2.pcolormesh(tm, ff[keep], Zt, shading="gouraud", cmap=SPEC_CMAP, vmin=0, vmax=1, rasterized=True)
    a2.axhline(3, color="white", lw=0.5, ls=(0, (2, 2))); a2.axhline(7, color="white", lw=0.5, ls=(0, (2, 2)))
    a2.set_xlabel("Time (s)"); a2.set_yticks([2, 4, 6, 8, 10])
    if j == 0:
        a2.set_ylabel("Jaw movement frequency (Hz)")
    else:
        a2.tick_params(labelleft=False)
d_ = IX.assign(i=np.arange(len(IX)))
g = d_[d_.cohort == "IPVS"]
per = [(sid, int(gg.label.iloc[0]), T["jaw"][gg.i.to_numpy()].mean(0)) for sid, gg in g.groupby("sid")]
band = (f >= 3) & (f <= 7); keep = (f >= 1) & (f <= 12)
per.sort(key=lambda r: r[2][band].sum() / r[2].sum())
labs = np.array([lb for _, lb, _ in per])
Lall = np.array([10 * np.log10(np.clip(s / s.sum(), 1e-8, None)) for _, _, s in per])
Dm = gaussian_filter((Lall - Lall[labs == 0].mean(0))[:, keep], (0.6, 0.8)); vv = np.nanpercentile(np.abs(Dm), 97)
subc = top[2].subgridspec(2, 2, width_ratios=[0.07, 1], height_ratios=[0.045, 1], wspace=0.04, hspace=0.12)
axc = fig.add_subplot(subc[1, 1]); strip = fig.add_subplot(subc[1, 0], sharey=axc); cbx = fig.add_subplot(subc[0, 1])
im = axc.pcolormesh(f[keep], np.arange(len(per)), Dm, cmap=DIV, vmin=-vv, vmax=vv, shading="gouraud", rasterized=True)
for fb in (3, 7):
    axc.axvline(fb, color=STYLE["ink"], lw=0.5, ls=(0, (2, 2)))
axc.set_xlim(1, 12); axc.set_xlabel("Jaw movement frequency (Hz)"); axc.tick_params(labelleft=False, left=False)
axc.spines["left"].set_visible(False)
strip.imshow(labs[:, None], cmap=LinearSegmentedColormap.from_list("lab", [STYLE["healthy"], STYLE["pd"]]), aspect="auto",
             origin="lower", extent=(0, 1, -0.5, len(per) - 0.5))
strip.set_xticks([]); strip.set_yticks([]); strip.set_ylabel("Every Italian speaker, sorted by jaw tremor  ->")
for s_ in strip.spines.values(): s_.set_visible(False)
cb = fig.colorbar(im, cax=cbx, orientation="horizontal"); cb.outline.set_linewidth(0.3)
cbx.xaxis.set_ticks_position("top"); cbx.tick_params(labelsize=STYLE["small"] - 0.6, length=1.5, pad=1)
cbx.set_title("spectrum vs healthy average (dB)", fontsize=STYLE["small"] - 0.4, pad=8)
nh, npd = int((labs[-12:] == 1).sum()), int((labs[:12] == 1).sum())
for yy_, txt, cc in ((len(per) - 1.5, f"most tremor: {nh} of 12 are PD", STYLE["pd"]), (0.5, f"all speakers: {int(labs.sum())} of {len(labs)} are PD", STYLE["ink2"])):
    axc.text(11.7, yy_, txt, ha="right", va="top" if yy_ > 1 else "bottom", fontsize=STYLE["small"] - 0.4, color=cc, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
bot = outer[1].subgridspec(1, 4, width_ratios=[1.3, 0.8, 0.8, 1.0], wspace=0.62)
subd = bot[0].subgridspec(2, 1, height_ratios=[1, 0.55], hspace=0.1)
axd = fig.add_subplot(subd[0]); axd2 = fig.add_subplot(subd[1], sharex=axd)
L0, L1 = Lall[labs == 0], Lall[labs == 1]
for L, col, nm in ((L0, STYLE["healthy"], "Healthy"), (L1, STYLE["pd"], "Parkinson's")):
    mu, se = L.mean(0), L.std(0) / np.sqrt(len(L))
    axd.fill_between(f, mu - se, mu + se, color=col, alpha=0.15, lw=0); axd.plot(f, mu, color=col, lw=1.0, label=f"{nm} (n = {len(L)})")
mu_all = np.r_[L0.mean(0)[keep], L1.mean(0)[keep]]
axd.set_xlim(1, 12); axd.set_ylim(mu_all.min() - 3, mu_all.max() + 6); axd.set_ylabel("Jaw power share (dB)")
axd.axvspan(3, 7, color=STYLE["band"], lw=0, zorder=0); axd.tick_params(labelbottom=False)
axd.legend(loc="upper right", fontsize=STYLE["small"] - 0.3)
dd = L1.mean(0) - L0.mean(0); sed = np.sqrt(L1.var(0) / len(L1) + L0.var(0) / len(L0))
axd2.axvspan(3, 7, color=STYLE["band"], lw=0, zorder=0)
axd2.fill_between(f, dd - 1.96 * sed, dd + 1.96 * sed, color=STYLE["grey"], alpha=0.3, lw=0)
axd2.plot(f, dd, color=STYLE["pd"], lw=0.9); axd2.axhline(0, color=STYLE["ink2"], lw=0.5)
pv = ttest_ind(L1, L0, axis=0, equal_var=False).pvalue; df_ = np.median(np.diff(f))
m_ = np.nanmax(np.abs(np.r_[(dd - 1.96 * sed)[keep], (dd + 1.96 * sed)[keep]])) * 1.15
axd2.set_ylim(-m_ * 0.75, m_)
for i in np.where((pv < 0.05) & keep)[0]:
    axd2.plot([f[i] - df_ / 2, f[i] + df_ / 2], [-m_ * 0.62, -m_ * 0.62], color=STYLE["ink"], lw=2.0, solid_capstyle="butt")
axd2.set_ylabel("PD - healthy\n(dB, 95% CI)"); axd2.set_xlabel("Frequency (Hz)")
axd2.text(1.1, -m_ * 0.5, "p < 0.05", fontsize=STYLE["small"] - 0.6, va="bottom", color=STYLE["ink2"])
def rain_pair(ax, d, col, ylab, sec, meas):
    raincloud(ax, [d[d.label == 0][col].dropna(), d[d.label == 1][col].dropna()], [0, 1], [STYLE["healthy"], STYLE["pd"]], width=0.35, s=5)
    e = E[(E.section == sec) & (E.measure == meas)]; lo, hi = np.nanpercentile(d[col], [0, 100])
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.30 * (hi - lo))
    if len(e):
        bracket(ax, 0, 1, hi + 0.08 * (hi - lo), f"{stars(e.q.iloc[0])}  AUC {e.auc.iloc[0]:.2f}", dy=0.04 * (hi - lo))
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Healthy\nn = {int((d.label == 0).sum())}", f"PD\nn = {int((d.label == 1).sum())}"])
    ax.set_xlim(-0.55, 1.6); ax.set_ylabel(ylab)
axe = fig.add_subplot(bot[1]); rain_pair(axe, TI, "mouth_LI", "Jaw tremor (3-7 Hz share)", "D IPVS vowels (clean)", "mouth_LI")
axf = fig.add_subplot(bot[2]); rain_pair(axf, TF, "voice_loudness", "Loudness tremor, phone (log)", "D FIGSHARE vowels (clean)", "voice_loudness")
axg = fig.add_subplot(bot[3])
allz = pd.concat([pd.DataFrame({"v": zs(d.voice_loudness), "m": zs(d.mouth_tremor)}) for d in (TI, TF)]).dropna()
xx, yy = np.mgrid[-3:3:80j, -3:3:80j]
kde = gaussian_kde(allz[["v", "m"]].to_numpy().T)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
axg.contourf(xx, yy, kde, levels=8, cmap=LinearSegmentedColormap.from_list("g", ["#FFFFFF", "#D5DCE4", "#7D8796"]), zorder=0)
for d, col, nm, mk in ((TI, STYLE["italian"], "Italian /a/", "o"), (TF, STYLE["phone"], "Phone /a/", "s")):
    axg.scatter(zs(d.voice_loudness), zs(d.mouth_tremor), s=6, color=col, marker=mk, lw=0, alpha=0.9,
                label=f"{nm}\n" + rho_txt(d.mouth_tremor, d.voice_loudness))
axg.set_xlim(-3, 3); axg.set_ylim(-3, 3); axg.set_xlabel("Voice-box tremor (z)"); axg.set_ylabel("Mouth tremor (z)")
axg.legend(loc="upper left", ncol=1, fontsize=STYLE["small"] - 1.0, frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
r3 = outer[2].subgridspec(1, 3, width_ratios=[0.12, 1.0, 0.12])
def auc_e(sec, meas):
    e = E[(E.section == sec) & (E.measure == meas)]
    return (e.auc.iloc[0], e.q.iloc[0]) if len(e) else (np.nan, np.nan)
axi = fig.add_subplot(r3[1]); ygrid(axi, "x")
items = [("mouth_LI", "Jaw"), ("mouth_TT", "Tongue tip"), ("mouth_TB", "Tongue body"), ("mouth_TD", "Tongue back"),
         ("mouth_LL", "Lower lip"), ("voice_loudness", "Voice loudness"), ("voice_pitch", "Voice pitch")]
axi.axvspan(0.5, 1, color="#F4F4F5", lw=0); axi.axvspan(0, 0.5, color="#EDF1F6", lw=0)
for i, (mm, nm) in enumerate(items):
    y0 = len(items) - 1 - i
    a1, q1 = auc_e("D IPVS vowels (clean)", mm); a2, _ = auc_e("D IPVS vowels (phone)", mm); a3, q3 = auc_e("D FIGSHARE vowels (clean)", mm)
    axi.plot([a1, a2], [y0 + 0.13, y0 + 0.13], color=STYLE["italian"], lw=1.4, alpha=0.6)
    axi.scatter(a1, y0 + 0.13, s=26, color=STYLE["italian"], edgecolor=STYLE["ink"] if q1 < 0.05 else "white", lw=0.8 if q1 < 0.05 else 0.4, zorder=4)
    axi.scatter(a2, y0 + 0.13, s=20, facecolor="white", edgecolor=STYLE["italian"], lw=0.9, zorder=4)
    axi.scatter(a3, y0 - 0.16, s=24, marker="s", color=STYLE["phone"], edgecolor=STYLE["ink"] if q3 < 0.05 else "white",
                lw=0.8 if q3 < 0.05 else 0.4, zorder=4)
axi.axvline(0.5, color=STYLE["grey"], lw=0.6, ls=(0, (2, 2))); axi.axhline(1.5, color=STYLE["grey"], lw=0.5)
axi.set_yticks(range(len(items))[::-1]); axi.set_yticklabels([nm for _, nm in items])
for tl, (mm, _) in zip(axi.get_yticklabels(), items):
    tl.set_color(ACOL[mm.split("_")[1]] if mm.startswith("mouth") else STYLE["ink2"])
axi.set_xlim(0.1, 0.95); axi.set_ylim(-0.95, len(items) - 0.45); axi.set_xlabel("AUC, Parkinson's vs healthy (black ring = q < 0.05)")
axi.tick_params(axis="y", length=0); axi.spines["left"].set_visible(False)
axi.text(0.12, -0.75, "less tremor in PD", fontsize=STYLE["small"] - 0.5, color=H_, va="center")
axi.text(0.93, -0.75, "more tremor in PD", fontsize=STYLE["small"] - 0.5, color=P_, va="center", ha="right")
axi.scatter([], [], s=22, color=STYLE["italian"], label="Italian, microphone"); axi.scatter([], [], s=18, facecolor="white", edgecolor=STYLE["italian"], lw=0.9, label="Italian, phone band")
axi.scatter([], [], s=20, marker="s", color=STYLE["phone"], label="Figshare, phone")
axi.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=0.8, handletextpad=0.2)
for ax_, s in ((tops[0], "a"), (tops[1], "b"), (cbx, "c"), (axd, "d"), (axe, "e"), (axf, "f"), (axg, "g"), (axi, "h")):
    letter(fig, ax_, s)
save(fig, "Fig5_tremor"); plt.show()
''')

md("""
## Figure 6 - Why deep learning? Comparison with classic acoustics
**a** cross-validated AUC (patients vs healthy; mean and 95% range over repeats drawn as density) for English (filled,
MDVR) and Italian (open, IPVS; classic voice quality at ceiling because the groups were recorded in different rooms);
**b** the same mouth movement measured by a classic formant-based stand-in points the wrong way (larger in Parkinson's),
the network estimate the expected way (arrow: classic -> network); **c** share of the deep-learning composite not predictable
from all classic acoustics; **d** Spearman correlations among all 41 features across English speakers, clustered (top tree;
strip = feature family): the network's speed measures cluster with speech and articulation rate, whereas its range and
working-space measures form their own block, separate from every classic family; **e** every English speaker in the space
of the network measures and in the space of classic acoustics (first two principal components; contours hold 50% of
each group; AUC = separation along PC1).
""")

code(r'''
apply_style()
A = load("h2h_auc.csv"); F = load("h2h_formant_vs_dl.csv"); UQ = load("h2h_unique_information.csv"); LR = load("h2h_likelihood_ratio.csv")
S = load("speaker_table_dl_and_classic.csv")
fig = plt.figure(figsize=(STYLE["w2"], 192 * MM))
outer = fig.add_gridspec(2, 1, left=0.16, right=0.985, bottom=0.085, top=0.955, height_ratios=[0.8, 1.3], hspace=0.30)
r1 = outer[0].subgridspec(1, 3, width_ratios=[1.15, 1.0, 0.78], wspace=0.7)
NAME = {"classic timing/prosody": "Timing & prosody", "classic voice quality": "Voice quality",
        "classic articulation (incl. formant movement)": "Articulation (formants)", "all classic": "All classic",
        "deep learning (validated composite)": "Deep learning (1 number)", "deep learning (all 15)": "Deep learning (15)",
        "all classic + DL composite": "Classic + DL composite", "all classic + DL all": "Classic + DL (all)"}
order = ["classic timing/prosody", "classic voice quality", "classic articulation (incl. formant movement)", "all classic",
         "deep learning (validated composite)", "deep learning (all 15)", "all classic + DL composite", "all classic + DL all"]
order = [o for o in order if o in set(A.feature_set)]
axa = fig.add_subplot(r1[0]); ygrid(axa, "x")
y = np.arange(len(order))[::-1]
for i_, yi_ in enumerate(y):
    if i_ % 2 == 0:
        axa.axhspan(yi_ - 0.5, yi_ + 0.5, color="#F2F4F7", lw=0, zorder=0)
fam = lambda n: P_ if n.startswith("deep") else "#6E1020" if "+" in n else H_
for yi, n in zip(y, order):
    c = fam(n)
    for coh, dy, filled in (("MDVR reading", 0.17, True), ("IPVS reading", -0.17, False)):
        r_ = A[(A.cohort == coh) & (A.feature_set == n)]
        if not len(r_):
            continue
        r_ = r_.iloc[0]
        eye(axa, yi + dy, r_.auc, r_.auc_lo, r_.auc_hi, STYLE["grey"], h=0.15, alpha=0.45 if filled else 0.22)
        axa.scatter(r_.auc, yi + dy, s=15, color=c if filled else "white", edgecolor=c, lw=0.7, zorder=4)
        if filled:
            axa.text(min(r_.auc_hi + 0.01, 0.97), yi + dy, f"{r_.auc:.2f}", va="center", fontsize=STYLE["small"] - 0.7, color=STYLE["ink2"])
nclassic = sum(not n.startswith("deep") and "+" not in n for n in order)
axa.axhline(len(order) - nclassic - 0.5, color=STYLE["grey"], lw=0.5, ls=(0, (2, 2)))
axa.set_yticks(y); axa.set_yticklabels([NAME[n] for n in order]); axa.set_xlim(0.5, 1.0); axa.set_ylim(-0.6, len(order) - 0.4)
for tl, n in zip(axa.get_yticklabels(), order):
    tl.set_color(fam(n))
axa.set_xlabel("Cross-validated AUC"); axa.spines["left"].set_visible(False); axa.tick_params(axis="y", length=0)
axa.scatter([], [], s=15, color=STYLE["ink2"], label="English"); axa.scatter([], [], s=15, color="white", edgecolor=STYLE["ink2"], label="Italian")
axa.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
lr = LR[LR.cohort == "MDVR reading"]
if len(lr):
    axa.text(0.0, -0.2, f"English: DL composite adds to classic predictors, likelihood-ratio p = {lr.p.iloc[0]:.3f}",
             transform=axa.transAxes, fontsize=STYLE["small"] - 0.4, color=STYLE["ink2"], va="top")
axb = fig.add_subplot(r1[1])
rows, ys, yc = [], [], 0.0
heads = []
for coh, nm in (("MDVR reading", "English"), ("IPVS reading", "Italian")):
    heads.append((yc, nm)); yc -= 0.85
    for _, r_ in F[F.cohort == coh].iterrows():
        rows.append(r_); ys.append(yc); yc -= 0.78
    yc -= 0.2
axb.axvspan(-2, 0, color="#F2F4F7", lw=0, zorder=0)
for r_, y0 in zip(rows, ys):
    axb.annotate("", xy=(r_.dl_beta, y0), xytext=(r_.classic_beta, y0),
                 arrowprops=dict(arrowstyle="-|>", color="#B7B3C2", lw=0.9, mutation_scale=6, shrinkA=3.5, shrinkB=3.5))
    axb.scatter(r_.classic_beta, y0, s=20, color=H_, zorder=3, edgecolor="white", lw=0.4)
    axb.scatter(r_.dl_beta, y0, s=20, color=P_, zorder=3, edgecolor="white", lw=0.4)
axb.axvline(0, color=STYLE["grey"], lw=0.5)
axb.set_yticks(ys); axb.set_yticklabels([r_.movement.capitalize() for r_ in rows], fontsize=STYLE["small"] - 0.3)
for y0, nm in heads:
    axb.text(-1.55, y0, nm, ha="left", va="center", fontsize=STYLE["small"], fontweight="bold")
axb.set_xlim(-1.6, 1.6); axb.set_ylim(yc + 0.3, 0.55); axb.set_xlabel("Parkinson's - healthy (SD)")
axb.spines["left"].set_visible(False); axb.tick_params(axis="y", length=0)
axb.text(-0.8, 0.62, "smaller in PD", ha="center", va="bottom", fontsize=STYLE["small"] - 0.3, color=H_)
axb.text(0.8, 0.62, "larger in PD", ha="center", va="bottom", fontsize=STYLE["small"] - 0.3, color=P_)
axb.scatter([], [], s=18, color=H_, label="Classic formants"); axb.scatter([], [], s=18, color=P_, label="Network")
axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
axc = fig.add_subplot(r1[2]); axc.set_xlim(-2.05, 2.05); axc.set_ylim(-5.0, 2.15); axc.set_aspect("equal"); axc.axis("off")
MIX = "#4A4E69"
for k, (coh, nm) in enumerate((("MDVR reading", "English"), ("IPVS reading", "Italian"))):
    uu = UQ[UQ.cohort == coh]
    if not len(uu):
        continue
    u = float(uu.unique_share.iloc[0]); cy = 0.15 - 3.25 * k
    rC, rD = 1.0, 0.95; dist = rC + rD - 2 * rD * (1 - u)
    cC, cD = (-dist / 2, cy), (dist / 2, cy)
    circC = plt.Circle(cC, rC, fc=H_, alpha=0.16, ec="none", zorder=2)
    circD = plt.Circle(cD, rD, fc="#ECEEF2", alpha=1.0, ec="none", zorder=2)
    axc.add_patch(circC); axc.add_patch(circD)
    ov = plt.Circle(cD, rD, fc=MIX, alpha=0.30, ec="none", zorder=3); axc.add_patch(ov); ov.set_clip_path(circC)
    axc.add_patch(plt.Circle(cC, rC, fill=False, ec=H_, lw=1.0, zorder=4))
    axc.add_patch(plt.Circle(cD, rD, fill=False, ec=P_, lw=1.0, zorder=4))
    xo = (cD[0] - rD + cC[0] + rC) / 2
    axc.text(cC[0] - 0.42, cy, "classic\nacoustics", ha="center", va="center", fontsize=STYLE["small"] - 1.0, color=H_, zorder=5,
             linespacing=1.05)
    axc.text(xo, cy, f"{1 - u:.0%}", ha="center", va="center", fontsize=STYLE["small"] - 0.4, color="white", fontweight="bold", zorder=5)
    axc.text((cD[0] + rD + cC[0] + rC) / 2 + 0.02, cy, f"{u:.0%}", ha="center", va="center", fontsize=STYLE["small"] + 0.4,
             color=P_, fontweight="bold", zorder=5)
    axc.text(0, cy - 1.18, nm, ha="center", va="top", fontsize=STYLE["small"], fontweight="bold", color=STYLE["ink"])
axc.text(0, 2.12, "Network composite:\nshared vs unique", ha="center", va="top", fontsize=STYLE["small"] - 0.2, color=STYLE["ink2"])
# d: clustered correlation map (English speakers)
r2 = outer[1].subgridspec(1, 2, width_ratios=[1.45, 0.78], wspace=0.30)
en = S[S.cohort == "MDVR reading"].copy()
feats, fams = list(DLM), ["Deep learning"] * len(DLM)
for fnm, cols in CLASSIC.items():
    for c in cols:
        if c in en and en[c].notna().sum() > 10:
            feats.append(c); fams.append(fnm)
C = en[feats].corr(method="spearman").fillna(0).to_numpy()
Zl = linkage(squareform(1 - np.abs(C), checks=False), "average"); od = leaves_list(Zl)
dsub = r2[0].subgridspec(3, 2, width_ratios=[0.03, 1], height_ratios=[0.12, 0.03, 1], wspace=0.012, hspace=0.012)
axdn = fig.add_subplot(dsub[0, 1]); axts = fig.add_subplot(dsub[1, 1]); axls = fig.add_subplot(dsub[2, 0]); axd = fig.add_subplot(dsub[2, 1])
dendrogram(Zl, ax=axdn, no_labels=True, link_color_func=lambda k: "#8E92A8")
for ln in axdn.collections:
    ln.set_linewidth(0.6)
axdn.set_xlim(0, 10 * len(feats)); axdn.axis("off")
famlist = list(FAMC); fcm = ListedColormap([FAMC[f] for f in famlist])
fi = np.array([famlist.index(fams[i]) for i in od])
axts.imshow(fi[None, :], cmap=fcm, vmin=0, vmax=len(famlist) - 1, aspect="auto", interpolation="nearest"); axts.axis("off")
axls.imshow(fi[:, None], cmap=fcm, vmin=0, vmax=len(famlist) - 1, aspect="auto", interpolation="nearest")
axls.set_xticks([]); axls.set_yticks(range(len(od)))
labs_ = [short(feats[i]) if fams[i] == "Deep learning" else CL_LAB.get(feats[i], feats[i]) for i in od]
axls.set_yticklabels(labs_, fontsize=STYLE["small"] - 1.0); axls.tick_params(length=0, pad=1)
for tl, i in zip(axls.get_yticklabels(), od):
    tl.set_color(STYLE["ink2"])
for s_ in axls.spines.values(): s_.set_visible(False)
nfe = len(od)
im = axd.pcolormesh(np.arange(nfe + 1) - 0.5, np.arange(nfe + 1) - 0.5, C[np.ix_(od, od)], cmap=DIV, vmin=-1, vmax=1,
                    edgecolors="white", linewidth=0.2)
axd.set_xlim(-0.5, nfe - 0.5); axd.set_ylim(nfe - 0.5, -0.5)
axd.set_xticks(range(len(od))); axd.set_xticklabels(labs_, rotation=90, fontsize=STYLE["small"] - 1.0); axd.set_yticks([])
for tl, i in zip(axd.get_xticklabels(), od):
    tl.set_color(STYLE["ink2"])
axd.tick_params(length=0, pad=1)
for s_ in axd.spines.values(): s_.set_visible(False)
cax = axd.inset_axes([1.015, 0.6, 0.022, 0.38])
cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.3); cb.set_ticks([-1, 0, 1]); cax.tick_params(labelsize=STYLE["small"] - 0.7, length=1.5)
cb.set_label("Spearman r", fontsize=STYLE["small"] - 0.4)
for k, fnm in enumerate(famlist):
    axd.text(1.015, 0.45 - 0.065 * k, fnm, transform=axd.transAxes, fontsize=STYLE["small"] - 0.6, color=FAMC[fnm], fontweight="bold",
             va="center", ha="left")
# e: PCA maps
esub = r2[1].subgridspec(2, 1, hspace=0.45)
def pca_map(ax, cols, title):
    X_ = en[cols].apply(lambda c: c.fillna(c.median())); X_ = (X_ - X_.mean()) / X_.std(ddof=0)
    X_ = X_.loc[:, X_.std() > 0].to_numpy()
    U, s_, Vt = np.linalg.svd(X_, full_matrices=False); pcs = U[:, :2] * s_[:2]; ve = s_ ** 2 / (s_ ** 2).sum()
    lab = en.label.to_numpy()
    if pcs[lab == 1, 0].mean() < pcs[lab == 0, 0].mean():
        pcs[:, 0] *= -1
    (x0_, y0_), (x1_, y1_) = pcs.min(0), pcs.max(0); px, py = 0.12 * (x1_ - x0_), 0.15 * (y1_ - y0_)
    gx, gy = np.mgrid[x0_ - px:x1_ + px:80j, y0_ - py:y1_ + py:80j]
    for lb, col in ((0, H_), (1, P_)):
        kd = gaussian_kde(pcs[lab == lb].T, bw_method=0.75)(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        lv8 = hdr_level(kd, 0.8)
        ax.contourf(gx, gy, kd, levels=[lv8, kd.max() * 1.01], colors=[STYLE["grey"]], alpha=0.10)
        lv = hdr_level(kd, 0.5)
        ax.contourf(gx, gy, kd, levels=[lv, kd.max() * 1.01], colors=[STYLE["grey"]], alpha=0.16)
        ax.contour(gx, gy, kd, levels=[lv], colors=[col], linewidths=0.8)
        ax.scatter(pcs[lab == lb, 0], pcs[lab == lb, 1], s=10, color=col, edgecolor="white", lw=0.3, zorder=3)
        ax.scatter(*pcs[lab == lb].mean(0), s=40, marker="X", color=col, edgecolor="white", lw=0.5, zorder=4)
    a_ = auc_of(pcs[lab == 1, 0], pcs[lab == 0, 0])
    ax.set_title(title, loc="left", fontsize=STYLE["small"], fontweight="bold")
    ax.text(0.97, 0.04, f"PC1 AUC {a_:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=STYLE["small"] - 0.3)
    ax.set_xlim(x0_ - px, x1_ + px); ax.set_ylim(y0_ - py, y1_ + py)
    ax.set_xlabel(f"PC1 ({ve[0]:.0%})"); ax.set_ylabel(f"PC2 ({ve[1]:.0%})"); ygrid(ax, "both")
axe1 = fig.add_subplot(esub[0]); pca_map(axe1, DLM, "Network movement measures (15)")
cl_cols = [c for f_, c in zip(fams, feats) if f_ != "Deep learning"]
axe2 = fig.add_subplot(esub[1]); pca_map(axe2, cl_cols, f"Classic acoustics ({len(cl_cols)})")
axe1.scatter([], [], s=10, color=H_, label="Healthy"); axe1.scatter([], [], s=10, color=P_, label="Parkinson's")
axe1.legend(loc="upper left", ncol=1, fontsize=STYLE["small"] - 0.4)
for ax_, s in ((axa, "a"), (axb, "b"), (axc, "c"), (axdn, "d"), (axe1, "e")):
    letter(fig, ax_, s)
save(fig, "Fig6_why_deep_learning"); plt.show()
''')

md("""
## Figure 7 - Individual movement profiles
**a** every patient's value on each movement measure, as z relative to healthy speakers of the same dataset and sex (dot,
one patient; grey band, central 80% of healthy speakers; dark tick, patient median; number, share of patients below the
healthy 10th percentile; bold, sensor-validated); **b** composite of the four validated measures per speaker (higher = smaller
movements; dashed line, healthy 90th percentile; number, share of patients above it).
""")

code(r'''
apply_style()
S = load("speaker_table_dl_and_classic.csv")
INK, INK2 = STYLE["ink"], STYLE["ink2"]
cols = ["TT_range", "TT_speed", "TB_range", "TB_speed", "TB_space", "TD_range", "TD_speed", "LI_range", "LI_speed",
        "LL_range", "LL_speed", "UL_range", "UL_speed", "LA_range", "LA_speed"]
arts = [("Tongue tip", 2), ("Tongue body", 3), ("Tongue back", 2), ("Jaw", 2), ("Lower lip", 2), ("Upper lip", 2), ("Lip opening", 2)]
Z = z_vs_healthy(S, cols)
hc, pdz = Z[Z.label == 0], Z[Z.label == 1]
xpos, x, spans = [], 0.0, []
for nm, k in arts:
    spans.append((nm, x, x + k - 1)); xpos += list(x + np.arange(k)); x += k + 0.6
xpos = np.array(xpos)
fig = plt.figure(figsize=(STYLE["w2"], 92 * MM))
gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, bottom=0.2, top=0.86, width_ratios=[3.3, 1], wspace=0.22)
axa = fig.add_subplot(gs[0]); rng = np.random.default_rng(7)
for j, c in enumerate(cols):
    lo, hi = np.nanpercentile(hc[c], [10, 90])
    axa.add_patch(FancyBboxPatch((xpos[j] - 0.38, lo), 0.76, hi - lo, boxstyle="round,pad=0,rounding_size=0.12",
                                 mutation_aspect=0.6, fc="#E6E9EE", ec="none", zorder=1))
    v = pdz[c].dropna().to_numpy(); vc = np.clip(v, -4.2, 3.2)
    axa.scatter(xpos[j] + rng.uniform(-0.27, 0.27, len(v)), vc, s=5, color=P_, alpha=0.55, lw=0, zorder=3)
    md_ = np.median(v)
    axa.plot([xpos[j] - 0.32, xpos[j] + 0.32], [md_, md_], color=INK, lw=1.6, solid_capstyle="round", zorder=4)
    share = np.mean(v < lo)
    axa.text(xpos[j], 3.55, f"{share:.0%}", ha="center", va="bottom", fontsize=STYLE["small"] - 0.9,
             color=INK if c in VALIDATED else INK2, fontweight="bold" if c in VALIDATED else "normal")
axa.axhline(0, color=STYLE["grey"], lw=0.5, zorder=0)
axa.set_xlim(-0.6, xpos[-1] + 0.6); axa.set_ylim(-4.4, 4.6); axa.set_yticks([-4, -2, 0, 2])
axa.set_ylabel("z relative to healthy")
axa.set_xticks(xpos); axa.set_xticklabels([c.split("_")[1] for c in cols], rotation=90, fontsize=STYLE["small"] - 0.6)
for tl, c in zip(axa.get_xticklabels(), cols):
    tl.set_fontweight("bold" if c in VALIDATED else "normal"); tl.set_color(INK if c in VALIDATED else INK2)
axa.tick_params(axis="x", length=0); axa.spines["bottom"].set_visible(False)
for nm, x0, x1 in spans:
    axa.text((x0 + x1) / 2, -5.55, nm, ha="center", va="top", fontsize=STYLE["small"] - 0.5, color=INK,
             transform=axa.transData, clip_on=False)
axa.text(-0.45, 4.3, "patients below the healthy range (healthy: 10%)", ha="left", va="bottom", fontsize=STYLE["small"] - 0.9, color=INK2)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D as _L2
axa.legend(handles=[Patch(color="#E6E9EE", label="healthy, central 80%"),
                    _L2([], [], marker="o", ls="", color=P_, alpha=0.6, ms=3, label="patient"),
                    _L2([], [], color=INK, lw=1.6, label="patient median")],
           loc="lower left", bbox_to_anchor=(0.0, 1.05), ncol=3, fontsize=STYLE["small"] - 0.6, handletextpad=0.3)
axb = fig.add_subplot(gs[1])
grp = [("MDVR reading", 0, "EN\nhealthy"), ("MDVR reading", 1, "EN\nPD"), ("IPVS reading", 0, "IT\nhealthy"), ("IPVS reading", 1, "IT\nPD")]
data = [Z[(Z.cohort == c) & (Z.label == l)].dl_composite.dropna() for c, l, _ in grp]
pos = [0, 1, 2.4, 3.4]
raincloud(axb, data, pos, [H_, P_, H_, P_], width=0.32, s=4, seed=3)
for (c, l, _), d, p in zip(grp, data, pos):
    if l == 1:
        thr = np.percentile(data[pos.index(p) - 1], 90)
        axb.plot([p - 1.35, p + 0.45], [thr, thr], color=INK2, lw=0.6, ls=(0, (2, 1.5)), zorder=0)
        axb.text(p + 0.05, 5.3, f"{np.mean(d > thr):.0%}", ha="center", va="bottom", fontsize=STYLE["small"] - 0.5,
                 color=P_, fontweight="bold")
axb.set_xticks(pos); axb.set_xticklabels([g for _, _, g in grp], fontsize=STYLE["small"] - 0.5)
axb.set_ylim(-3, 6.3); axb.set_ylabel("Composite (higher = smaller movements)"); axb.set_xlim(-0.6, 3.95); ygrid(axb)
axb.text(1.7, 6.3, "PD above healthy 90th pct.", ha="center", va="top", fontsize=STYLE["small"] - 1.0, color=INK2)
for ax_, s in ((axa, "a"), (axb, "b")):
    letter(fig, ax_, s)
save(fig, "Fig7_kinematic_fingerprint"); plt.show()
''')

md("""
## Supplementary Figure S7 - The measures are repeatable and the effects are robust
**a** test-retest: the four sensor-validated measures from two separate readings of the same Italian speakers
(ICC(2,1); dashed = identity); **b** reliability of every measure (shaded zones: good >= 0.75, excellent >= 0.9);
**c** group differences before and after adjusting for speech rate (each point one measure in one language; on the
diagonal = unchanged by rate); **d** English vs Italian group differences per measure (each point one measure;
diagonal = identical effect; lower-left quadrant = smaller in PD in both languages).
""")

code(r'''
apply_style()
RR = load("reading_recordings.csv", "20_"); R = load("mouth_reliability.csv", "20_"); E = load("mouth_bradykinesia_effects.csv", "20_")
fig = plt.figure(figsize=(STYLE["w2"], 128 * MM))
outer = fig.add_gridspec(2, 1, left=0.075, right=0.985, bottom=0.08, top=0.95, height_ratios=[0.6, 1.12], hspace=0.34)
r1 = outer[0].subgridspec(1, 4, wspace=0.42)
it = RR[RR.cohort == "IPVS reading"]
axes_a = []
for k, m in enumerate(["TT_range", "TB_space", "LL_range", "LL_speed"]):
    ax = fig.add_subplot(r1[k]); axes_a.append(ax); ygrid(ax, "both")
    w = it.pivot_table(index=["sid", "label"], columns="take", values=m).dropna().reset_index()
    lo, hi = np.nanpercentile(np.r_[w.B1, w.B2], [0, 100]); pad = 0.08 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=STYLE["grey"], lw=0.6, ls=(0, (2, 2)))
    for lb, col in ((0, H_), (1, P_)):
        g = w[w.label == lb]
        ax.scatter(g.B1, g.B2, s=13, color=col, edgecolor="white", lw=0.3, alpha=0.9, zorder=3)
    icc = R[(R.measure == m) & R.section.str.startswith("B")].icc
    ax.text(0.04, 0.96, f"ICC {icc.iloc[0]:.2f}" if len(icc) else "", transform=ax.transAxes, va="top", fontsize=STYLE["small"])
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal")
    ax.set_title(label(m), fontsize=STYLE["small"], color=ACOL[m.split("_")[0]], fontweight="bold")
    ax.set_xlabel("Reading 1")
    if k == 0:
        ax.set_ylabel("Reading 2")
axes_a[-1].scatter([], [], s=12, color=H_, label="Healthy"); axes_a[-1].scatter([], [], s=12, color=P_, label="PD")
axes_a[-1].legend(loc="lower right", fontsize=STYLE["small"] - 0.6, handletextpad=0.1)
r2 = outer[1].subgridspec(1, 4, width_ratios=[0.12, 1.05, 1, 1], wspace=0.5)
axb = fig.add_subplot(r2[1])
r = R[R.section.str.startswith("B")].sort_values("icc"); y = np.arange(len(r))
axb.axvspan(0.75, 0.9, color=STYLE["english"], alpha=0.08, lw=0); axb.axvspan(0.9, 1.0, color=STYLE["english"], alpha=0.18, lw=0)
for yi, (_, rr) in zip(y, r.iterrows()):
    c = ACOL[rr.measure.split("_")[0]]
    axb.plot([0.5, rr.icc], [yi, yi], color=c, lw=1.6, alpha=0.35, solid_capstyle="round")
    axb.scatter(rr.icc, yi, s=20, color=c, edgecolor="white", lw=0.4, zorder=3)
axb.set_yticks(y); axb.set_yticklabels([label(m) for m in r.measure], fontsize=STYLE["small"] - 0.4)
for tl, m in zip(axb.get_yticklabels(), r.measure):
    tl.set_color(ACOL[m.split("_")[0]]); tl.set_fontweight("bold" if m in VALIDATED else "normal")
axb.set_xlim(0.5, 1.0); axb.set_xlabel("ICC(2,1), two readings"); axb.tick_params(axis="y", length=0); axb.spines["left"].set_visible(False)
axb.text(0.825, len(r) - 0.3, "good", ha="center", va="bottom", fontsize=STYLE["small"] - 0.4, color=STYLE["ink2"])
axb.text(0.95, len(r) - 0.3, "excellent", ha="center", va="bottom", fontsize=STYLE["small"] - 0.4, color=STYLE["ink2"])
axb.set_ylim(-0.6, len(r) + 0.6)
def effect_scatter(ax, xs, ys, cols_, mks, xl, yl, lim):
    ax.plot([-lim, lim], [-lim, lim], color=STYLE["grey"], lw=0.6, ls=(0, (2, 2)))
    ax.axhline(0, color=STYLE["grid"], lw=0.6); ax.axvline(0, color=STYLE["grid"], lw=0.6)
    for x_, y_, c, mk in zip(xs, ys, cols_, mks):
        ax.scatter(x_, y_, s=22, color=c, marker=mk, edgecolor="white", lw=0.4, zorder=3)
    ax.set_xlim(-lim, lim * 0.55); ax.set_ylim(-lim, lim * 0.55); ax.set_aspect("equal")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
axc = fig.add_subplot(r2[2])
xs, ys, cs, mk = [], [], [], []
for sec, m_ in (("B MDVR reading", "o"), ("B IPVS reading", "s")):
    e = E[(E.section == sec) & E.measure.isin(DLM)]
    xs += e.beta.tolist(); ys += e.beta_rate_adjusted.tolist(); cs += [ACOL[m.split("_")[0]] for m in e.measure]; mk += [m_] * len(e)
effect_scatter(axc, xs, ys, cs, mk, "Unadjusted (SD)", "Speech-rate adjusted (SD)", 1.6)
axc.scatter([], [], s=16, marker="o", color=STYLE["ink2"], label="English"); axc.scatter([], [], s=16, marker="s", color=STYLE["ink2"], label="Italian")
axc.legend(loc="upper left", fontsize=STYLE["small"] - 0.5)
axd = fig.add_subplot(r2[3])
en = E[E.section == "B MDVR reading"].set_index("measure"); itv = E[E.section == "B IPVS reading"].set_index("measure")
ms = [m for m in DLM if m in en.index and m in itv.index]
effect_scatter(axd, en.loc[ms, "beta"], itv.loc[ms, "beta"], [ACOL[m.split("_")[0]] for m in ms], ["D"] * len(ms),
               "English (SD)", "Italian (SD)", 1.6)
rho_ = spearmanr(en.loc[ms, "beta"], itv.loc[ms, "beta"])[0]
both = int(((en.loc[ms, "beta"] < 0) & (itv.loc[ms, "beta"] < 0)).sum())
axd.text(0.04, 0.96, f"{both} of {len(ms)} measures smaller in PD\nin both languages (rank rho {rho_:.2f})", transform=axd.transAxes,
         va="top", fontsize=STYLE["small"] - 0.4)
for a in ["TT", "TB", "TD", "LI", "LL", "UL", "LA"]:
    axd.scatter([], [], s=14, marker="D", color=ACOL[a], label=SHORT[a])
axd.legend(loc="lower right", fontsize=STYLE["small"] - 0.9, handletextpad=0.1, labelspacing=0.25)
for ax_, s in ((axes_a[0], "a"), (axb, "b"), (axc, "c"), (axd, "d")):
    letter(fig, ax_, s)
save(fig, "FigS7_robustness"); plt.show()
''')

md("## Supplementary figures")

code(r'''
apply_style()
R = load("mouth_reliability.csv", "20_"); FL = load("frame_level_agreement.csv").set_index("speaker"); A = load("h2h_auc.csv")
fig = plt.figure(figsize=(STYLE["w2"], 118 * MM))
gs = fig.add_gridspec(2, 2, left=0.19, right=0.97, bottom=0.08, top=0.90, height_ratios=[1.15, 0.85], width_ratios=[1, 1.05], hspace=0.5, wspace=0.85)
axa = fig.add_subplot(gs[0, 0]); r = R[R.section.str.startswith("B")].sort_values("icc"); y = np.arange(len(r))
axa.hlines(y, 0.5, r.icc, color=[ACOL[m.split("_")[0]] for m in r.measure], lw=1.4)
axa.scatter(r.icc, y, s=14, color=[ACOL[m.split("_")[0]] for m in r.measure], zorder=3)
axa.axvline(0.75, color=STYLE["ink"], lw=0.5, ls="--"); axa.set_xlim(0.5, 1.0)
axa.set_yticks(y); axa.set_yticklabels([label(m) for m in r.measure]); axa.set_xlabel("ICC(2,1), two reading takes")
axa.text(0.755, len(r) - 0.6, "good", fontsize=STYLE["small"], color=STYLE["ink2"])
axb = fig.add_subplot(gs[0, 1])
NAME = {"classic timing/prosody": "Timing & prosody", "classic voice quality": "Voice quality", "classic articulation (incl. formant movement)": "Articulation",
        "all classic": "All classic", "deep learning (validated composite)": "DL composite", "deep learning (all 15)": "DL (15)",
        "all classic + DL composite": "Classic + DL comp.", "all classic + DL all": "Classic + DL (all)"}
a = A[A.cohort == "IPVS reading"].reset_index(drop=True); y = np.arange(len(a))[::-1]
cols = [STYLE["pd"] if n.startswith("deep") else STYLE["healthy"] for n in a.feature_set]
axb.hlines(y, a.auc_lo, a.auc_hi, color=cols, lw=1.4, alpha=0.55); axb.scatter(a.auc, y, s=16, color=cols, zorder=3)
axb.set_yticks(y); axb.set_yticklabels([NAME.get(n, n) for n in a.feature_set]); axb.set_xlim(0.5, 1.0); axb.set_xlabel("Cross-validated AUC")
axb.set_title("Italian: classic voice quality at ceiling\n(patients and controls recorded in different rooms)", loc="left", fontsize=STYLE["small"])
axc = fig.add_subplot(gs[1, :])
im = axc.imshow(FL.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
for i in range(FL.shape[0]):
    for j in range(FL.shape[1]):
        axc.text(j, i, f"{FL.values[i, j]:.2f}", ha="center", va="center", fontsize=STYLE["small"] - 0.3,
                 color="white" if FL.values[i, j] < 0.6 else STYLE["ink"])
axc.set_xticks(range(FL.shape[1])); axc.set_xticklabels(FL.columns); axc.set_yticks(range(FL.shape[0])); axc.set_yticklabels(FL.index)
for s_ in axc.spines.values(): s_.set_visible(False)
axc.tick_params(length=0)
cb = fig.colorbar(im, ax=axc, fraction=0.025, pad=0.01); cb.set_label("Pearson r"); cb.outline.set_linewidth(0.4)
axc.set_xlabel(f"Channel (frame-by-frame agreement with real sensors, MOCHA-TIMIT; median r = {np.nanmedian(FL.values):.2f})")
for ax_, s in ((axa, "a"), (axb, "b"), (axc, "c")):
    letter(fig, ax_, s)
save(fig, "FigS1_reliability_ceiling_agreement"); plt.show()
''')

code(r'''
apply_style()
M = load_npz("mocha_example.npz")
fig = plt.figure(figsize=(STYLE["w2"], 120 * MM))
gs = fig.add_gridspec(6, 2, left=0.11, right=0.985, bottom=0.08, top=0.93, hspace=0.18, wspace=0.18)
te = np.arange(M["real"].shape[0]) / FS_EMA
pairs = [("TT", "Tongue tip"), ("TB", "Tongue body"), ("TD", "Tongue back"), ("LI", "Jaw"), ("LL", "Lower lip"), ("UL", "Upper lip")]
first = None
for i, (a, nm) in enumerate(pairs):
    for j, axn in enumerate(("X", "Y")):
        ax = fig.add_subplot(gs[i, j], sharex=first) if first else fig.add_subplot(gs[i, j]); first = first or ax
        c = f"{a}{axn}"
        ax.fill_between(te, M["real"][:, CI_[c]] - 0.18, M["real"][:, CI_[c]] + 0.18, color="#C9C9C9", lw=0)
        ax.plot(te, M["est"][:, CI_[c]], color=ACOL[a], lw=0.7)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlim(0, te[-1])
        ax.text(1.0, 0.95, f"r = {M['r'][CI_[c]]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=STYLE["small"], color=STYLE["ink2"])
        if j == 0:
            ax.text(-0.02, 0.5, nm, transform=ax.transAxes, ha="right", va="center", color=ACOL[a], fontsize=STYLE["small"])
        if i == 0:
            ax.set_title("Horizontal (front-back)" if axn == "X" else "Vertical (up-down)", fontsize=STYLE["size"])
        if i < 5:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (s)")
save(fig, "FigS2_all_channels_vs_sensors"); plt.show()
''')

code(r'''
apply_style()
E = load("mouth_bradykinesia_effects.csv", "20_")
ms = ["TT_range", "TT_speed", "TB_range", "TB_speed", "TB_space", "TD_range", "TD_speed", "LI_range", "LI_speed",
      "LL_range", "LL_speed", "UL_range", "UL_speed", "LA_range", "LA_speed"]
fig, axs = plt.subplots(1, 2, figsize=(STYLE["w2"], 92 * MM), sharey=True)
fig.subplots_adjust(left=0.17, right=0.985, bottom=0.13, top=0.9, wspace=0.2)
y = np.arange(len(ms))[::-1]
for ax, (sec, nm, col) in zip(axs, (("B MDVR reading", "English (MDVR)", STYLE["english"]), ("B IPVS reading", "Italian (IPVS)", STYLE["italian"]))):
    e = E[E.section == sec].set_index("measure").reindex(ms); se = se_from_p(e.beta, e.p)
    ax.hlines(y, e.beta - 1.96 * se, e.beta + 1.96 * se, color=col, lw=1.0)
    ax.scatter(e.beta, y, s=14, color=col, zorder=3)
    ax.scatter(e.beta_rate_adjusted, y, s=12, facecolor="white", edgecolor=col, lw=0.7, zorder=4)
    for yi, q in zip(y, e.q):
        ax.text(1.0, yi, stars(q), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=STYLE["small"])
    ax.axvline(0, color=STYLE["grey"], lw=0.5); ax.set_xlim(-2.2, 1.0); ax.set_title(nm, loc="left", color=col, fontweight="bold")
    ax.set_xlabel("Parkinson's - healthy (SD)")
axs[0].set_yticks(y); axs[0].set_yticklabels([label(m) for m in ms])
axs[1].scatter([], [], s=12, color=STYLE["grey"], label="Unadjusted"); axs[1].scatter([], [], s=12, facecolor="white", edgecolor=STYLE["grey"], label="Speech-rate adjusted")
axs[1].legend(loc="lower left")
for ax_, s in zip(axs, "ab"):
    letter(fig, ax_, s)
save(fig, "FigS3_forest_by_dataset"); plt.show()
''')

code(r'''
apply_style()
X = load_npz("example_reading.npz")
fig = plt.figure(figsize=(STYLE["w2"], 92 * MM))
sube = fig.add_gridspec(2, 2, left=0.085, right=0.985, bottom=0.11, top=0.92, height_ratios=[0.85, 1], hspace=0.08, wspace=0.05)
axes_e = []
for j, (tag, nm, col) in enumerate((("healthy", "Typical healthy reader", STYLE["healthy"]), ("pd", "Typical Parkinson's reader", STYLE["pd"]))):
    a1 = fig.add_subplot(sube[0, j]); show_spec(a1, X[f"{tag}_spec"], fmax=4000, ylabel=(j == 0))
    a1.tick_params(labelbottom=False); a1.set_xlabel(""); a1.set_title(nm, loc="left", color=col, fontweight="bold")
    if j:
        a1.tick_params(labelleft=False)
    a2 = fig.add_subplot(sube[1, j], sharex=a1)
    Ex = X[f"{tag}_ema"]; tx_ = np.arange(len(Ex)) / FS_EMA
    for a, off in (("TT", 0.0), ("LI", -4.6)):
        yv = Ex[:, CI_[f"{a}Y"]]
        a2.fill_between(tx_, off, yv - np.median(yv) + off, color=ACOL[a], alpha=0.15, lw=0)
        a2.plot(tx_, yv - np.median(yv) + off, color=ACOL[a], lw=0.7)
        if j == 0:
            a2.text(-0.01, off, ART[a], transform=a2.get_yaxis_transform(), ha="right", va="center", color=ACOL[a], fontsize=STYLE["small"])
    a2.set_ylim(-8.6, 3.6); a2.set_yticks([]); a2.spines["left"].set_visible(False); a2.set_xlabel("Time (s)")
    axes_e.append(a1)
for ax_, s in zip(axes_e, "ab"):
    letter(fig, ax_, s)
save(fig, "FigS4_example_readers"); plt.show()
''')


md("""
## Supplementary Figure S5 - Agreement with real sensors for each MOCHA speaker
Spearman rho between the network estimate and the real sensor, per movement measure and speaker (~460 sentences each);
ringed = passes the pre-specified rule (median of the three speakers >= 0.5).
""")

code(r'''
apply_style()
SV = load("summary_measure_validity.csv"); SVS = load("summary_measure_validity_per_speaker.csv")
fig = plt.figure(figsize=(STYLE["w1"] * 1.45, 92 * MM))
axf = fig.add_axes([0.30, 0.08, 0.55, 0.80])
SVi = SV.set_index("measure"); meas = SV.sort_values("median_spearman", ascending=False).measure.tolist()
spk = [("fsew0", "Speaker 1\n(female)"), ("msak0", "Speaker 2\n(male)"), ("maps0", "Speaker 3\n(male)")]
Mx = SVS.pivot(index="measure", columns="speaker", values="spearman")
cn = Normalize(0, 0.75)
for i, m in enumerate(meas):
    vals = [Mx.loc[m, s] for s, _ in spk] + [SVi.loc[m, "median_spearman"]]
    for j, v in enumerate(vals):
        x0 = j + (0.35 if j == 3 else 0)
        axf.add_patch(Rectangle((x0, i), 0.94, 0.9, fc=SEQ_G(cn(v)), ec="none"))
        axf.text(x0 + 0.47, i + 0.45, f"{v:.2f}", ha="center", va="center", fontsize=STYLE["small"] - 0.6,
                 color="white" if v > 0.45 else STYLE["ink"], fontweight="bold" if j == 3 else "normal")
    if SVi.loc[m, "passes"]:
        axf.add_patch(Rectangle((3.35, i), 0.94, 0.9, fc="none", ec=STYLE["ink"], lw=0.9))
axf.set_xlim(-0.05, 4.35); axf.set_ylim(len(meas) - 0.05, -0.05)
axf.set_yticks(np.arange(len(meas)) + 0.45); axf.set_yticklabels([label(m) for m in meas])
for tl, m in zip(axf.get_yticklabels(), meas):
    tl.set_fontweight("bold" if SVi.loc[m, "passes"] else "normal"); tl.set_color(ACOL[m.split("_")[0]])
axf.set_xticks([0.47, 1.47, 2.47, 3.82]); axf.set_xticklabels([s for _, s in spk] + ["Median"]); axf.xaxis.tick_top()
axf.tick_params(length=0)
for s_ in axf.spines.values(): s_.set_visible(False)
cax = axf.inset_axes([1.06, 0.25, 0.04, 0.5])
cb = fig.colorbar(plt.cm.ScalarMappable(cn, SEQ_G), cax=cax); cb.outline.set_linewidth(0.3); cb.set_ticks([0, 0.25, 0.5, 0.75])
cax.tick_params(labelsize=STYLE["small"] - 0.6, length=1.5); cb.set_label("rho with real sensors", fontsize=STYLE["small"] - 0.3)
save(fig, "FigS5_validity_per_speaker"); plt.show()
''')

md("""
## Supplementary Figure S6 - Lower-lip range and Hoehn & Yahr stage
English patients (n = 16); least squares with 95% bootstrap band; colour = UPDRS II speech item.
""")

code(r'''
apply_style()
S = load("speaker_table_dl_and_classic.csv"); pdm = S[(S.cohort == "MDVR reading") & (S.label == 1)]
rng = np.random.default_rng(4)
fig, ax = plt.subplots(figsize=(STYLE["w1"], 70 * MM)); fig.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.95)
d = pdm[["LL_range", "hy", "updrs2"]].dropna(); xs = np.linspace(d.hy.min(), d.hy.max(), 50); fits = []
for _ in range(600):
    i = rng.integers(0, len(d), len(d))
    if d.hy.iloc[i].nunique() > 1:
        fits.append(np.polyval(np.polyfit(d.hy.iloc[i], d.LL_range.iloc[i], 1), xs))
lo, hi = np.percentile(fits, [2.5, 97.5], axis=0)
ax.fill_between(xs, lo, hi, color=STYLE["grey"], alpha=0.25, lw=0); ax.plot(xs, np.polyval(np.polyfit(d.hy, d.LL_range, 1), xs), color=P_, lw=1.0)
ax.scatter(d.hy + rng.uniform(-0.06, 0.06, len(d)), d.LL_range, s=18, c=[SEVC[int(v)] for v in d.updrs2], edgecolor=STYLE["ink2"], lw=0.3, zorder=3)
ax.text(0.97, 0.97, rho_txt(d.hy, d.LL_range), transform=ax.transAxes, ha="right", va="top", fontsize=STYLE["small"] - 0.2)
ax.set_xlabel("Hoehn & Yahr stage"); ax.set_ylabel("Lower-lip range"); ygrid(ax)
save(fig, "FigS6_hoehn_yahr"); plt.show()
''')

md("## Graphical abstract")

code(r'''
apply_style()
X = load_npz("example_reading.npz"); PO = load("reading_pooled_EN_IT.csv").set_index("measure")
CL = load_npz("movement_clouds.npz"); grid = CL["grid"]; ctr = (grid[:-1] + grid[1:]) / 2
fig = plt.figure(figsize=(160 * MM, 72 * MM))
ax1 = fig.add_axes([0.02, 0.30, 0.25, 0.50]); show_spec(ax1, X["pd_spec"][:, :300], fmax=4000, ylabel=False)
ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_xlabel(""); ax1.set_title("Ordinary voice recording", fontsize=STYLE["size"], fontweight="bold")
fig.add_artist(FancyArrowPatch((0.285, 0.55), (0.35, 0.55), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, color=STYLE["ink2"], lw=1))
fig.text(0.318, 0.61, "deep\nlearning", ha="center", va="bottom", fontsize=STYLE["small"], color=STYLE["ink2"])
ax2 = fig.add_axes([0.35, 0.06, 0.33, 0.82]); draw_mouth(ax2, tongue=False, fill="#F2F4F7"); ax2.axis("off")
D, HS, vlim = density_difference(CL, ctr)
for a, (cx, cy) in ANCHOR.items():
    ax2.pcolormesh(cx + ctr * SCALE * 0.95, cy + ctr * SCALE * 0.95, D[a].T, cmap=DIV, vmin=-vlim, vmax=vlim, shading="gouraud", zorder=4,
                   rasterized=True)
fig.text(0.515, 0.95, "Where the mouth stops reaching", ha="center", va="top", fontsize=STYLE["size"], fontweight="bold")
fig.text(0.70, 0.88, "In Parkinson's disease", fontsize=STYLE["size"] + 0.5, fontweight="bold", va="top")
lines = [("Tongue & jaw move less", f"up to {abs(PO.loc['LI_speed', 'beta']):.1f} SD, 2 languages"), ("Tracks speech severity", "rho = -0.87 (UPDRS II)"),
         ("Jaw tremor at 3-7 Hz", "separate from voice tremor"), ("Validated on real sensors", "adds to classic acoustics")]
for k, (h, s) in enumerate(lines):
    fig.text(0.70, 0.76 - 0.17 * k, h, fontsize=STYLE["size"], fontweight="bold", va="top", color=STYLE["pd"] if k == 0 else STYLE["ink"])
    fig.text(0.70, 0.685 - 0.17 * k, s, fontsize=STYLE["small"], va="top", color=STYLE["ink2"])
save(fig, "Graphical_abstract"); plt.show()
''')

md("""
## Supplementary animation (optional)
Set `MAKE_ANIMATION = True` to make `SuppMovie_moving_mouth.gif` (adds about a minute): the healthy and the Parkinson's
example readers side by side - spectrogram with a moving cursor above, articulators moving on the mouth below.
""")

code(r'''
MAKE_ANIMATION = False
if MAKE_ANIMATION:
    from matplotlib.animation import FuncAnimation, PillowWriter
    apply_style()
    X = load_npz("example_reading.npz")
    fig = plt.figure(figsize=(150 * MM, 100 * MM), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 1], hspace=0.12, wspace=0.05, left=0.06, right=0.99, bottom=0.02, top=0.93)
    dots, curs, trails = {}, {}, {}
    for j, (tag, nm, col) in enumerate((("healthy", "Healthy", STYLE["healthy"]), ("pd", "Parkinson's", STYLE["pd"]))):
        a1 = fig.add_subplot(gs[0, j]); show_spec(a1, X[f"{tag}_spec"], fmax=4000, ylabel=(j == 0)); a1.set_xticks([])
        a1.set_title(nm, color=col, fontweight="bold", loc="left"); curs[tag] = a1.axvline(0, color="white", lw=1.2)
        a2 = fig.add_subplot(gs[1, j]); draw_mouth(a2)
        for a, (cx, cy) in ANCHOR.items():
            dots[(tag, a)] = a2.scatter([cx], [cy], s=30, color=ACOL[a], zorder=7, ec="white", lw=0.5)
            trails[(tag, a)], = a2.plot([], [], color=ACOL[a], lw=0.6, alpha=0.6, zorder=6)
    n = min(len(X["healthy_ema"]), len(X["pd_ema"]))
    def upd(i):
        out = []
        for tag in ("healthy", "pd"):
            Ex = X[f"{tag}_ema"]
            for a, (cx, cy) in ANCHOR.items():
                xy = Ex[:, [CI_[f"{a}X"], CI_[f"{a}Y"]]]; d = (xy - xy.mean(0)) * SCALE; lo = max(0, i - 12)
                dots[(tag, a)].set_offsets([[cx + d[i, 0], cy + d[i, 1]]]); trails[(tag, a)].set_data(cx + d[lo:i + 1, 0], cy + d[lo:i + 1, 1])
                out += [dots[(tag, a)], trails[(tag, a)]]
            curs[tag].set_xdata([i / FS_EMA, i / FS_EMA]); out.append(curs[tag])
        return out
    FuncAnimation(fig, upd, frames=range(0, n, 2), blit=True).save(FIG_DIR / "SuppMovie_moving_mouth.gif", writer=PillowWriter(fps=25))
    plt.close(fig); print("saved SuppMovie_moving_mouth.gif")
''')
