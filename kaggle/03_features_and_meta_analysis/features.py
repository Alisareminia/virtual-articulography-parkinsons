"""PD voice project - step 03: interpretable voice features, channel-robustness test, per-dataset effects, meta-analysis.

Runs on Kaggle, on the harmonised audio produced by step 02.

1. Features (identical code for every dataset), by task:
   - sustained vowels : F0, F0 variability, jitter, shimmer, HNR, CPPS, formants, spectral balance (<= 4 kHz)
   - connected speech : F0 variability/range (monopitch), intensity variability (monoloudness), speech and
                        articulation rate, pauses, voicing, HNR, CPPS, spectral balance incl. 4-8 kHz energy
   - syllable repetition (DDK): rate, rhythm regularity, intensity decay
   - IPVS 5-vowel articulation: vowel articulation index (VAI), triangular vowel space area (tVSA)
2. Per-dataset PD-vs-control effects: standardised mean difference adjusted for age and sex
   (Figshare: controls restricted to the PD age range; MDVR: sex estimated from F0, age not available).
3. Random-effects meta-analysis per feature for the tasks shared across datasets:
   sustained /a/ = IPVS + Figshare;  reading = IPVS + MDVR.
4. Channel-robustness test: add IPVS "PD-room" vs "control-room" background (at each room's typical SNR)
   to the same carrier recordings, re-run the whole pipeline, measure how far each feature moves.
   Natural experiment: the 3 IPVS patients recorded in both the 2016 and 2017 setups.
5. Candidate biomarkers: same direction in both datasets, significant in the bias-free dataset,
   pooled effect significant, and room effect small relative to the PD effect.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "praat-parselmouth"], check=False)

import io  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import warnings  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from math import gcd  # noqa: E402
from pathlib import Path  # noqa: E402

import librosa  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import parselmouth  # noqa: E402
import soundfile as sf  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from parselmouth.praat import call  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.signal import find_peaks, resample_poly  # noqa: E402
from scipy.stats import mannwhitneyu, norm  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
RNG = np.random.default_rng(2026)

C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", muted="#a3a29c",
         IPVS="#2a78d6", FIGSHARE="#eb6834", MDVR="#1baf7a")
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "text.color": C["ink"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False,
})

# ----------------------------------------------------------------------------- feature definitions
PHON = ["f0_mean_hz", "f0_sd_st", "jitter_local", "jitter_rap", "jitter_ppq5", "shimmer_local", "shimmer_apq3",
        "shimmer_apq11", "hnr_db", "cpps_db", "intensity_sd_db", "alpha_ratio_db", "hammarberg_db",
        "ltas_slope_db_khz", "hf_ratio_db", "centroid_hz", "bw95_hz", "f1_hz", "f2_hz"]
CONN = ["f0_mean_hz", "f0_sd_st", "f0_range_st", "intensity_sd_db", "voiced_frac", "speech_rate_syl_s",
        "artic_rate_syl_s", "pause_rate_min", "pause_mean_s", "pause_ratio", "hnr_db", "cpps_db",
        "alpha_ratio_db", "hammarberg_db", "ltas_slope_db_khz", "hf_ratio_db", "centroid_hz", "bw95_hz"]
DDK = ["ddk_rate_syl_s", "ddk_cv", "ddk_int_slope_db_s", "intensity_sd_db", "cpps_db"]
VOWEL_SPACE = ["vai", "tvsa_hz2"]
LOG_FEATS = {"jitter_local", "jitter_rap", "jitter_ppq5", "shimmer_local", "shimmer_apq3", "shimmer_apq11",
             "pause_mean_s", "ddk_cv", "f0_sd_st", "f0_range_st", "tvsa_hz2"}
LABELS = {
    "f0_mean_hz": "Mean pitch (F0)", "f0_sd_st": "Pitch variability (F0 SD, st)", "f0_range_st": "Pitch range (st)",
    "jitter_local": "Jitter (local)", "jitter_rap": "Jitter (RAP)", "jitter_ppq5": "Jitter (PPQ5)",
    "shimmer_local": "Shimmer (local)", "shimmer_apq3": "Shimmer (APQ3)", "shimmer_apq11": "Shimmer (APQ11)",
    "hnr_db": "Harmonics-to-noise ratio", "cpps_db": "Cepstral peak prominence (CPPS)",
    "intensity_sd_db": "Loudness variability", "alpha_ratio_db": "Alpha ratio (1-4k vs <1k)",
    "hammarberg_db": "Hammarberg index", "ltas_slope_db_khz": "Spectral slope 0-4 kHz",
    "hf_ratio_db": "High-freq energy 4-8 kHz", "centroid_hz": "Spectral centroid", "bw95_hz": "95% energy bandwidth",
    "f1_hz": "F1", "f2_hz": "F2", "voiced_frac": "Voiced fraction", "speech_rate_syl_s": "Speech rate",
    "artic_rate_syl_s": "Articulation rate", "pause_rate_min": "Pauses per minute", "pause_mean_s": "Mean pause length",
    "pause_ratio": "Pause time ratio", "ddk_rate_syl_s": "DDK rate", "ddk_cv": "DDK irregularity (CV)",
    "ddk_int_slope_db_s": "DDK loudness decay", "vai": "Vowel articulation index", "tvsa_hz2": "Vowel space area",
}


# ============================================================================= helpers
def find_root_with(fname, sibling_dir=None):
    for root, dirs, files in os.walk(INPUT):
        if fname in files and (sibling_dir is None or sibling_dir in dirs):
            return Path(root)
    return None


def find_dir(name):
    for root, dirs, _ in os.walk(INPUT):
        if name in dirs:
            return Path(root) / name
    return None


def section(t):
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


def frames_db(x, sr, frame_s=0.025, hop_s=0.010):
    n, h = int(round(frame_s * sr)), int(round(hop_s * sr))
    if len(x) < n:
        return np.array([-120.0]), h
    idx = np.arange(n)[None, :] + h * np.arange(1 + (len(x) - n) // h)[:, None]
    return 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12), h


def activity(e):
    """Speech-activity mask; the threshold rises above the noise floor in recordings that contain pauses."""
    p10, p90, p95 = np.percentile(e, [10, 90, 95])
    thr = p95 - 30
    if p90 - p10 > 15:
        thr = max(thr, p10 + 6)
    return e > thr


def runs(mask):
    """(start, length) of consecutive True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(s, e - s))


# ============================================================================= acoustic measures
def f0_range_for(snd):
    """Two-pass F0 range (Hirst): floor 0.75*Q1, ceiling 1.5*Q3 of a wide first pass."""
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=600)
    f = p.selected_array["frequency"]
    v = f[f > 0]
    if len(v) < 10:
        return 75.0, 500.0
    q1, q3 = np.percentile(v, [25, 75])
    return max(50.0, 0.75 * q1), min(800.0, max(1.5 * q3, 0.75 * q1 + 100))


def spectral_balance(x, sr, act):
    n, h = int(0.025 * sr), int(0.01 * sr)
    if len(x) < n:
        return {}
    P = np.abs(librosa.stft(x, n_fft=n, hop_length=h, win_length=n, center=False)) ** 2
    k = min(P.shape[1], len(act))
    if act[:k].sum() < 5:
        return {}
    p = P[:, :k][:, act[:k]].mean(axis=1)
    f = np.fft.rfftfreq(n, 1 / sr)

    def band(lo, hi):
        return p[(f >= lo) & (f < hi)].sum() + 1e-20

    pdb = 10 * np.log10(p + 1e-20)
    lo = (f >= 50) & (f < 4000)
    out = dict(
        alpha_ratio_db=10 * np.log10(band(1000, 4000) / band(50, 1000)),
        hammarberg_db=pdb[(f >= 50) & (f < 2000)].max() - pdb[(f >= 2000) & (f < 4000)].max(),
        ltas_slope_db_khz=float(np.polyfit(f[lo] / 1000, pdb[lo], 1)[0]),
    )
    if sr >= 16000:
        m = (f >= 50) & (f < 8000)
        c = np.cumsum(p[m]) / p[m].sum()
        out.update(hf_ratio_db=10 * np.log10(band(4000, 8000) / band(50, 4000)),
                   centroid_hz=float((f[m] * p[m]).sum() / p[m].sum()),
                   bw95_hz=float(f[m][min(np.searchsorted(c, 0.95), m.sum() - 1)]))
    return out


def cpps(x, sr, max_frames=3000):
    """Smoothed cepstral peak prominence (Hillenbrand-style), averaged over active frames. Same code for all data."""
    n, h = int(0.04 * sr), int(0.01 * sr)
    if len(x) < n * 2:
        return np.nan
    nfft = 1 << int(np.ceil(np.log2(n)) + 1)
    idx = np.arange(n)[None, :] + h * np.arange(1 + (len(x) - n) // h)[:, None]
    e = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    act = np.where(e > np.percentile(e, 95) - 30)[0]
    if len(act) < 5:
        return np.nan
    if len(act) > max_frames:  # long recordings: evenly spaced subset of active frames
        act = act[np.linspace(0, len(act) - 1, max_frames).astype(int)]
    fr = x[idx[act]] * np.hanning(n)[None, :]
    spec = 20 * np.log10(np.abs(np.fft.rfft(fr, nfft, axis=1)) + 1e-12)
    ceps = 20 * np.log10(np.abs(np.fft.irfft(spec, nfft, axis=1))[:, : nfft // 2] + 1e-12)
    ceps = uniform_filter1d(uniform_filter1d(ceps, 7, axis=0, mode="nearest"), 3, axis=1, mode="nearest")
    q = np.arange(nfft // 2) / sr
    pk = (q >= 1 / 330) & (q <= 1 / 60)
    fit = q >= 0.001
    qf = q[fit]
    A = np.vstack([qf, np.ones_like(qf)]).T
    coef, *_ = np.linalg.lstsq(A, ceps[:, fit].T, rcond=None)      # per-frame regression line
    i = np.argmax(np.where(pk[None, :], ceps, -np.inf), axis=1)
    peak = ceps[np.arange(len(ceps)), i]
    return float(np.mean(peak - (coef[0] * q[i] + coef[1])))


def voiced_stats(snd, lo, hi):
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    t, f = pitch.xs(), pitch.selected_array["frequency"]
    inten = snd.to_intensity(minimum_pitch=max(lo, 50), time_step=0.01)
    ti, vi = inten.xs(), inten.values[0]
    v = f > 0
    out = {}
    if v.sum() >= 10:
        fv = f[v]
        st = 12 * np.log2(fv / np.median(fv))
        out.update(f0_mean_hz=float(np.mean(fv)), f0_sd_st=float(np.std(st)),
                   f0_range_st=float(np.percentile(st, 95) - np.percentile(st, 5)),
                   intensity_sd_db=float(np.std(np.interp(t[v], ti, vi))))
    harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=lo, silence_threshold=0.1, periods_per_window=1.0)
    hv = harm.values[0]
    hv = hv[hv > -199]
    if len(hv) >= 5:
        out["hnr_db"] = float(np.mean(hv))
    return out, (t, f, ti, vi)


def phonation_features(x, sr, sex=None):
    """Sustained vowel: analyse the central 80% of the voiced stretch."""
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    lo, hi = f0_range_for(snd)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    t, f = p.xs(), p.selected_array["frequency"]
    if (f > 0).sum() < 20:
        return {}
    t0, t1 = t[f > 0][0], t[f > 0][-1]
    a, b = t0 + 0.1 * (t1 - t0), t1 - 0.1 * (t1 - t0)
    part = snd.extract_part(from_time=a, to_time=b, preserve_times=False)
    y = part.values[0]
    out, _ = voiced_stats(part, lo, hi)
    pp = call(part, "To PointProcess (periodic, cc)", lo, hi)
    for name, cmd in (("jitter_local", "Get jitter (local)"), ("jitter_rap", "Get jitter (rap)"),
                      ("jitter_ppq5", "Get jitter (ppq5)")):
        out[name] = call(pp, cmd, 0, 0, 0.0001, 0.02, 1.3)
    for name, cmd in (("shimmer_local", "Get shimmer (local)"), ("shimmer_apq3", "Get shimmer (apq3)"),
                      ("shimmer_apq11", "Get shimmer (apq11)")):
        out[name] = call([part, pp], cmd, 0, 0, 0.0001, 0.02, 1.3, 1.6)
    out["cpps_db"] = cpps(y, sr)
    e, _ = frames_db(y, sr)
    out.update(spectral_balance(y, sr, e > np.percentile(e, 95) - 30))
    fmax = 5000 if sex == "M" else 5500
    fm = part.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=min(fmax, sr / 2 - 100))
    tt = np.linspace(0.1 * part.duration, 0.9 * part.duration, 40)
    for k in (1, 2):
        vals = np.array([fm.get_value_at_time(k, ti) for ti in tt], float)
        out[f"f{k}_hz"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
    return {k: (float(v) if v is not None else np.nan) for k, v in out.items()}


def connected_features(x, sr):
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    lo, hi = f0_range_for(snd)
    out, (t, f, ti, vi) = voiced_stats(snd, lo, hi)
    dur = len(x) / sr
    e, h = frames_db(x, sr)
    act = activity(e)
    pauses = [(s, ln) for s, ln in runs(~act) if ln * h / sr >= 0.15 and s > 0 and s + ln < len(act)]
    pdur = np.array([ln * h / sr for _, ln in pauses])
    out.update(pause_rate_min=len(pauses) / (dur / 60), pause_mean_s=float(pdur.mean()) if len(pdur) else np.nan,
               pause_ratio=float(pdur.sum() / dur))
    phon_time = act.sum() * h / sr
    out["voiced_frac"] = float((f > 0).sum() * 0.01 / max(phon_time, 1e-6))
    # syllable nuclei (after de Jong & Wempe 2009): voiced intensity peaks with >= 2 dB prominence
    thr = np.percentile(vi, 99) - 25
    pk, _ = find_peaks(vi, prominence=2, distance=5)
    fv_at = np.interp(ti[pk], t, (f > 0).astype(float)) if len(t) else np.zeros(len(pk))
    nuclei = int(((vi[pk] > thr) & (fv_at > 0.5)).sum())
    out.update(speech_rate_syl_s=nuclei / dur, artic_rate_syl_s=nuclei / max(phon_time, 1e-6))
    out["cpps_db"] = cpps(x, sr)
    out.update(spectral_balance(x, sr, act))
    return out


def ddk_features(x, sr):
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    inten = snd.to_intensity(minimum_pitch=75, time_step=0.005)
    ti, vi = inten.xs(), inten.values[0]
    thr = np.percentile(vi, 99) - 25
    pk, _ = find_peaks(vi, prominence=3, distance=int(0.06 / 0.005))
    pk = pk[vi[pk] > thr]
    out = {}
    if len(pk) >= 4:
        tp = ti[pk]
        iv = np.diff(tp)
        out.update(ddk_rate_syl_s=(len(pk) - 1) / (tp[-1] - tp[0]), ddk_cv=float(iv.std() / iv.mean()),
                   ddk_int_slope_db_s=float(np.polyfit(tp, vi[pk], 1)[0]))
    lo, hi = f0_range_for(snd)
    vs, _ = voiced_stats(snd, lo, hi)
    out["intensity_sd_db"] = vs.get("intensity_sd_db", np.nan)
    out["cpps_db"] = cpps(x, sr)
    return out


def family_kind(dataset, task_family):
    if task_family.startswith("vowel"):
        return "phon"
    if task_family == "syllables":
        return "ddk"
    return "conn"


def extract(x, sr, kind, sex=None):
    try:
        if kind == "phon":
            return phonation_features(x, sr, sex)
        if kind == "ddk":
            return ddk_features(x, sr)
        return connected_features(x, sr)
    except Exception as ex:  # noqa: BLE001
        print(f"    feature error ({kind}): {ex}")
        return {}


# ============================================================================= audio pipeline (same as step 02)
TARGET_DBFS = -25.0


def trim_and_normalise(x, sr):
    e, h = frames_db(x, sr)
    n = int(round(0.025 * sr))
    act = e > (np.percentile(e, 95) - 30)
    first, last = int(np.argmax(act)), len(act) - 1 - int(np.argmax(act[::-1]))
    s, t = max(0, int(first * h - 0.05 * sr)), min(len(x), int(last * h + n + 0.05 * sr))
    y = x[s:t].copy()
    e2, _ = frames_db(y, sr)
    a2 = e2 > (np.percentile(e2, 95) - 30)
    rms = np.sqrt(np.mean(10 ** (e2[a2] / 10))) if a2.any() else np.sqrt(np.mean(y ** 2))
    y *= 10 ** (TARGET_DBFS / 20) / (rms + 1e-12)
    pk = np.max(np.abs(y)) if len(y) else 0
    return y * (0.99 / pk) if pk > 0.99 else y


# ============================================================================= statistics
def transform(v, feat):
    v = pd.to_numeric(v, errors="coerce").astype(float)
    if feat in LOG_FEATS:
        v = np.log(v.clip(lower=1e-6))
    med = v.median()
    mad = 1.4826 * (v - med).abs().median()
    if mad > 0:
        v = v.clip(med - 4 * mad, med + 4 * mad)   # tame extreme outliers before standardising
    return v


def dataset_effect(sub, feat, covars):
    d = sub[["label", feat] + covars].copy()
    d[feat] = transform(d[feat], feat)
    d = d.dropna()
    n1, n0 = int((d.label == 1).sum()), int((d.label == 0).sum())
    if n1 < 5 or n0 < 5 or d[feat].std() == 0:
        return None
    a, b = d.loc[d.label == 1, feat], d.loc[d.label == 0, feat]
    sp = np.sqrt(((n1 - 1) * a.var() + (n0 - 1) * b.var()) / (n1 + n0 - 2))
    if not sp > 0:
        return None
    J = 1 - 3 / (4 * (n1 + n0) - 9)
    g = J * (a.mean() - b.mean()) / sp
    X = pd.DataFrame({"const": 1.0, "pd": d.label.astype(float)}, index=d.index)
    for c in covars:
        if c == "sex":
            if d.sex.nunique() > 1:
                X["male"] = (d.sex == "M").astype(float)
        else:
            X[c] = (d[c] - d[c].mean()) / (d[c].std() or 1)
    fit = sm.OLS((d[feat] - d[feat].mean()) / sp, X).fit(cov_type="HC3")
    return dict(n_pd=n1, n_ctrl=n0, g_unadj=g, beta=float(fit.params["pd"]), se=float(fit.bse["pd"]),
                p=float(fit.pvalues["pd"]), p_mw=float(mannwhitneyu(a, b).pvalue),
                mean_pd=float(np.exp(a.mean()) if feat in LOG_FEATS else a.mean()),
                mean_ctrl=float(np.exp(b.mean()) if feat in LOG_FEATS else b.mean()))


def dl_meta(b, se):
    b, se = np.asarray(b, float), np.asarray(se, float)
    w = 1 / se ** 2
    fe = np.sum(w * b) / np.sum(w)
    q = np.sum(w * (b - fe) ** 2)
    k = len(b)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    w2 = 1 / (se ** 2 + tau2)
    re = np.sum(w2 * b) / np.sum(w2)
    se_re = np.sqrt(1 / np.sum(w2))
    return dict(beta_fe=fe, se_fe=np.sqrt(1 / np.sum(w)), beta_re=re, se_re=se_re,
                ci_lo=re - 1.96 * se_re, ci_hi=re + 1.96 * se_re, p_re=2 * norm.sf(abs(re / se_re)),
                i2=max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0, tau2=tau2)


def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    q = np.empty(len(pv))
    q[o] = np.minimum(r, 1)
    out[ok] = q
    return out


# ============================================================================= main
class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for s in self.s:
            s.write(x)

    def flush(self):
        for s in self.s:
            s.flush()


def main():
    buf = io.StringIO()
    with redirect_stdout(Tee(sys.stdout, buf)):
        run()
    (OUT / "report.txt").write_text(buf.getvalue())


def run():
    s02 = find_root_with("manifest.csv", "clean_audio")
    print(f"step-02 outputs: {s02}")
    man = pd.read_csv(s02 / "manifest.csv")
    man["kind"] = [family_kind(d, t) for d, t in zip(man.dataset, man.task_family)]

    # ------------------------------------------------------------------ 1. features
    section("1. FEATURE EXTRACTION (harmonised audio, identical code for all datasets)")
    rows = []
    for i, r in enumerate(man.itertuples()):
        x, sr = sf.read(str(s02 / "clean_audio" / r.clean_path), dtype="float64")
        feats = extract(x, sr, r.kind, r.sex if isinstance(r.sex, str) else None)
        rows.append(dict(file_id=r.file_id, **feats))
        if i % 100 == 0:
            print(f"  {i}/{len(man)}")
    F = man.merge(pd.DataFrame(rows), on="file_id", how="left")

    # MDVR has no sex metadata: estimate from median F0 of connected speech (<160 Hz -> male)
    md = F[F.dataset == "MDVR"].groupby("subject_id").f0_mean_hz.median()
    sex_est = (md < 160).map({True: "M", False: "F"})
    F.loc[F.dataset == "MDVR", "sex"] = F.loc[F.dataset == "MDVR", "subject_id"].map(sex_est)
    F["sex_source"] = np.where(F.dataset == "MDVR", "estimated_from_F0", "metadata")
    F.to_csv(OUT / "features_file.csv", index=False)
    print("\nfeature availability (non-missing share) by kind:")
    for kind, cols in (("phon", PHON), ("conn", CONN), ("ddk", DDK)):
        sub = F[F.kind == kind]
        print(f"  {kind}: files={len(sub)}  " + ", ".join(f"{c}={sub[c].notna().mean():.0%}" for c in cols if c in sub))
    print("\nMDVR estimated sex per group:")
    print(F[F.dataset == "MDVR"].drop_duplicates("subject_id").groupby(["group", "sex"]).size().to_string())

    # ------------------------------------------------------------------ 2. subject-level tables
    section("2. SUBJECT-LEVEL FEATURES (mean over repeated files of the same task)")
    meta_cols = ["dataset", "subject_id", "group", "label", "age", "sex"]
    fam_map = {"vowel_a": "vowel_a", "read_passage": "reading", "dialogue": "dialogue", "syllables": "ddk",
               "phrases": "phrases", "words": "words"}
    F["family"] = F.task_family.map(fam_map).fillna(F.task_family)
    allfeat = sorted(set(PHON + CONN + DDK))
    S = F.groupby(["family"] + meta_cols, dropna=False)[[c for c in allfeat if c in F]].mean().reset_index()
    # IPVS vowel articulation across the five vowels (per subject)
    vw = F[(F.dataset == "IPVS") & F.task_family.str.startswith("vowel")]
    fm = vw.groupby(["subject_id", "task_family"])[["f1_hz", "f2_hz"]].mean().unstack()
    va = pd.DataFrame(index=fm.index)
    try:
        f1 = {v: fm[("f1_hz", f"vowel_{v}")] for v in "aiu"}
        f2 = {v: fm[("f2_hz", f"vowel_{v}")] for v in "aiu"}
        va["vai"] = (f2["i"] + f1["a"]) / (f1["i"] + f1["u"] + f2["u"] + f2["a"])
        va["tvsa_hz2"] = 0.5 * (f1["i"] * (f2["a"] - f2["u"]) + f1["a"] * (f2["u"] - f2["i"])
                                + f1["u"] * (f2["i"] - f2["a"])).abs()
    except KeyError as ex:
        print(f"  vowel space: missing {ex}")
    base = F[F.dataset == "IPVS"].drop_duplicates("subject_id").set_index("subject_id")[meta_cols[:1] + meta_cols[2:]]
    va = va.join(base, how="inner").reset_index().assign(family="vowel_space")
    S = pd.concat([S, va], ignore_index=True)
    S.to_csv(OUT / "features_subject.csv", index=False)
    print(S.groupby(["family", "dataset", "group"]).size().unstack(fill_value=0).to_string())

    # ------------------------------------------------------------------ 3. per-dataset effects
    section("3. PER-DATASET PD EFFECTS (standardised, adjusted; + = higher in PD)")
    analyses = []  # (family, dataset, subset-name, frame, covariates, features)
    ip = S[(S.dataset == "IPVS") & (S.group != "YHC")]
    fs = S[S.dataset == "FIGSHARE"]
    pd_min_age = fs.loc[fs.label == 1, "age"].min()
    fs_matched = fs[(fs.label == 1) | (fs.age >= pd_min_age)]
    mdv = S[S.dataset == "MDVR"]
    no2016 = set(F.loc[(F.dataset == "IPVS") & (F.session_year == 2016), "subject_id"])
    for fam, feats in (("vowel_a", PHON), ("reading", CONN)):
        analyses.append((fam, "IPVS", "IPVS", ip[ip.family == fam], ["age", "sex"], feats))
        analyses.append((fam, "IPVS", "IPVS excl. 2016 patients", ip[(ip.family == fam) & ~ip.subject_id.isin(no2016)],
                         ["age", "sex"], feats))
    analyses.append(("vowel_a", "FIGSHARE", "FIGSHARE age-matched", fs_matched[fs_matched.family == "vowel_a"], ["age", "sex"], PHON))
    analyses.append(("vowel_a", "FIGSHARE", "FIGSHARE all (age-adjusted)", fs[fs.family == "vowel_a"], ["age", "sex"], PHON))
    analyses.append(("reading", "MDVR", "MDVR", mdv[mdv.family == "reading"], ["sex"], CONN))
    analyses.append(("dialogue", "MDVR", "MDVR", mdv[mdv.family == "dialogue"], ["sex"], CONN))
    for fam, feats in (("ddk", DDK), ("phrases", CONN), ("words", CONN), ("vowel_space", VOWEL_SPACE)):
        analyses.append((fam, "IPVS", "IPVS", ip[ip.family == fam], ["age", "sex"], feats))
    eff = []
    for fam, ds, name, frame, cov, feats in analyses:
        for ft in feats:
            if ft not in frame or frame[ft].notna().sum() < 10:
                continue
            r = dataset_effect(frame, ft, cov)
            if r:
                eff.append(dict(family=fam, dataset=ds, analysis=name, feature=ft, **r))
    E = pd.DataFrame(eff)
    E.to_csv(OUT / "effects_per_dataset.csv", index=False)
    show = E[E.analysis.isin(["IPVS", "FIGSHARE age-matched", "MDVR"])].copy()
    show["cell"] = show.apply(lambda r: f"{r.beta:+.2f}{'*' if r.p < 0.05 else ' '}", axis=1)
    print("adjusted effect (SD units), * p<0.05:")
    print(show.pivot_table(index=["family", "feature"], columns="analysis", values="cell", aggfunc="first").fillna("").to_string())

    # ------------------------------------------------------------------ 4. channel robustness
    section("4. CHANNEL-ROBUSTNESS TEST (IPVS PD-room vs control-room background added to the same carriers)")
    rob = channel_robustness(man, s02)
    rob.to_csv(OUT / "robustness.csv", index=False)
    print(rob.round(3).to_string(index=False))
    nat = natural_experiment(F, S)
    if len(nat):
        nat.to_csv(OUT / "natural_experiment_2016_vs_2017.csv", index=False)
        print("\nnatural experiment - 3 IPVS patients, 2016 vs 2017 setup (mean |shift| in between-subject SD):")
        print(nat.round(2).to_string(index=False))

    # ------------------------------------------------------------------ 5. meta-analysis + candidates
    section("5. META-ANALYSIS AND BIOMARKER CANDIDATES")
    pairs = {"vowel_a": ("IPVS", "FIGSHARE age-matched"), "reading": ("IPVS", "MDVR")}
    M = []
    for fam, (a_name, b_name) in pairs.items():
        for ft in sorted(E.loc[E.family == fam, "feature"].unique()):
            ea = E[(E.family == fam) & (E.analysis == a_name) & (E.feature == ft)]
            eb = E[(E.family == fam) & (E.analysis == b_name) & (E.feature == ft)]
            if not len(ea) or not len(eb):
                continue
            ea, eb = ea.iloc[0], eb.iloc[0]
            m = dl_meta([ea.beta, eb.beta], [ea.se, eb.se])
            rr = rob[(rob.family == fam) & (rob.feature == ft)]
            room = float(rr.room_effect_sd.iloc[0]) if len(rr) else np.nan
            M.append(dict(family=fam, feature=ft, label=LABELS.get(ft, ft),
                          ipvs_beta=ea.beta, ipvs_p=ea.p, other=b_name.split()[0], other_beta=eb.beta, other_p=eb.p,
                          other_mean_pd=eb.mean_pd, other_mean_ctrl=eb.mean_ctrl,
                          same_sign=bool(np.sign(ea.beta) == np.sign(eb.beta)), **m,
                          room_effect_sd=room,
                          room_aligned_with_ipvs=bool(np.sign(room) == np.sign(ea.beta)) if np.isfinite(room) else None,
                          robust=bool(abs(room) <= 0.5 * abs(m["beta_re"])) if np.isfinite(room) else False))
    M = pd.DataFrame(M)
    M["q_re"] = np.nan
    for fam in M.family.unique():
        k = M.family == fam
        M.loc[k, "q_re"] = bh(M.loc[k, "p_re"])

    def tier(r):
        base = r.same_sign and r.other_p < 0.05 and r.robust
        if base and r.q_re < 0.05:
            return "1 replicated (FDR<0.05)"
        if base and r.p_re < 0.05:
            return "2 replicated (nominal)"
        if r.other_p < 0.05 and r.robust:
            return "3 bias-free dataset only"
        return "-"
    M["tier"] = M.apply(tier, axis=1)
    M = M.sort_values(["family", "tier", "p_re"])
    M.to_csv(OUT / "meta_analysis.csv", index=False)
    cols = ["family", "label", "ipvs_beta", "other", "other_beta", "other_p", "beta_re", "ci_lo", "ci_hi", "p_re", "q_re",
            "i2", "room_effect_sd", "robust", "tier"]
    print(M[cols].round(3).to_string(index=False))
    print("\nTiers: 1/2 = same direction in both datasets, p<0.05 in the bias-free dataset, room effect <= half the "
          "pooled PD effect, pooled p (FDR / nominal) < 0.05. Tier 3 = bias-free dataset only.")

    section("6. SINGLE-DATASET EXPLORATORY RESULTS (no replication dataset available)")
    X = E[E.analysis.isin(["IPVS"]) & E.family.isin(["ddk", "phrases", "words", "vowel_space"]) |
          ((E.analysis == "MDVR") & (E.family == "dialogue"))].copy()
    X["q"] = np.nan
    for (fam, an), idx in X.groupby(["family", "analysis"]).groups.items():
        X.loc[idx, "q"] = bh(X.loc[idx, "p"])
    X["label"] = X.feature.map(LABELS)
    print(X.sort_values(["family", "p"])[["family", "analysis", "label", "beta", "p", "q", "mean_pd", "mean_ctrl"]]
          .round(3).to_string(index=False))
    print("\nNote: IPVS-only results cannot be separated from the IPVS recording-room difference (step 02).")
    X.to_csv(OUT / "effects_single_dataset.csv", index=False)

    figures(M)
    print("\nsaved: features_file.csv, features_subject.csv, effects_per_dataset.csv, robustness.csv, "
          "meta_analysis.csv, effects_single_dataset.csv, natural_experiment_2016_vs_2017.csv, fig4-5")


# ============================================================================= robustness
def noise_pool(group):
    """Concatenate non-speech stretches from raw IPVS 2017 connected-speech files of one group."""
    root = find_dir("28 People with Parkinsons disease").parent
    gdir = {"PD": "28 People with Parkinsons disease", "EHC": "22 Elderly Healthy Control"}[group]
    segs, snrs = [], []
    for p in sorted((root / gdir).rglob("*.wav")):
        stem = p.stem.rstrip(".")
        m = re.match(r"^(B1|B2|FB1|PR1)[A-Za-z]+\d{2}[MF](\d{12})$", stem)
        if not m or m.group(2)[4:8] != "2017":
            continue
        x, sr = sf.read(str(p), dtype="float64", always_2d=True)
        x = x.mean(axis=1)
        x -= x.mean()
        if sr != 16000:
            g = gcd(sr, 16000)
            x = resample_poly(x, 16000 // g, sr // g)
            sr = 16000
        e, h = frames_db(x, sr)
        # background = the quietest frames (pauses); works even when the room is loud relative to speech
        ns = e < np.percentile(e, 10) + 3
        sp = e > np.percentile(e, 50)
        if ns.sum() > 20 and sp.sum() > 20:
            snrs.append(10 * np.log10(np.mean(10 ** (e[sp] / 10)) / np.mean(10 ** (e[ns] / 10))))
        for s, ln in runs(ns):
            if ln >= 5:  # >= 50 ms of background
                seg = x[s * h: (s + ln) * h]
                if len(seg) > 400:
                    segs.append(seg * np.hanning(len(seg)) ** 0.1)
        if sum(map(len, segs)) > 16000 * 180:
            break
    if not segs or not snrs:
        raise RuntimeError(f"no background audio found for IPVS {group} - robustness test cannot run")
    return np.concatenate(segs), float(np.median(snrs))


def add_noise(y, pool, snr_db, rng):
    if len(pool) < len(y):
        pool = np.tile(pool, int(np.ceil(len(y) / len(pool))) + 1)
    s = rng.integers(0, len(pool) - len(y) + 1)
    nz = pool[s:s + len(y)]
    e, _ = frames_db(y, 16000)
    ps = np.mean(10 ** (e[e > np.percentile(e, 50)] / 10))
    pn = np.mean(nz ** 2) + 1e-20
    return y + nz * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def channel_robustness(man, s02):
    pools = {g: noise_pool(g) for g in ("PD", "EHC")}
    for g, (pool, snr) in pools.items():
        print(f"  noise pool {g}: {len(pool) / 16000:.0f} s of IPVS non-speech, typical speech-to-background SNR {snr:.1f} dB")
    carriers = {
        "vowel_a": ("phon", man[(man.dataset == "IPVS") & (man.group == "EHC") & (man.task_family == "vowel_a")
                                & (man.session_year == 2017)], PHON),
        "reading": ("conn", man[(man.dataset == "MDVR") & (man.task_family == "read_passage")], CONN),
        "ddk": ("ddk", man[(man.dataset == "IPVS") & (man.group == "EHC") & (man.task_family == "syllables")], DDK),
    }
    out = []
    for fam, (kind, rows, feats) in carriers.items():
        recs = []
        for r in rows.itertuples():
            x, sr = sf.read(str(s02 / "clean_audio" / r.clean_path), dtype="float64")
            rng = np.random.default_rng(int(str(r.file_id)[:8], 16))
            sex = r.sex if isinstance(r.sex, str) else None
            base = extract(x, sr, kind, sex)
            v_pd = extract(trim_and_normalise(add_noise(x, pools["PD"][0], pools["PD"][1], rng), sr), sr, kind, sex)
            v_hc = extract(trim_and_normalise(add_noise(x, pools["EHC"][0], pools["EHC"][1], rng), sr), sr, kind, sex)
            recs.append((base, v_pd, v_hc))
        print(f"  carriers for {fam}: {len(recs)} files")
        for ft in feats:
            b = transform(pd.Series([r[0].get(ft, np.nan) for r in recs]), ft)
            p = transform(pd.Series([r[1].get(ft, np.nan) for r in recs]), ft)
            h = transform(pd.Series([r[2].get(ft, np.nan) for r in recs]), ft)
            ok = b.notna() & p.notna() & h.notna()
            if ok.sum() < 8 or b[ok].std() == 0:
                continue
            sd = b[ok].std()
            out.append(dict(family=fam, feature=ft, n_carriers=int(ok.sum()),
                            room_effect_sd=float((p[ok] - h[ok]).mean() / sd),
                            shift_pd_room_sd=float((p[ok] - b[ok]).mean() / sd),
                            shift_ctrl_room_sd=float((h[ok] - b[ok]).mean() / sd),
                            test_retest_r=float(np.corrcoef(p[ok], h[ok])[0, 1])))
    return pd.DataFrame(out)


def natural_experiment(F, S):
    ip = F[(F.dataset == "IPVS") & (F.group == "PD")]
    multi = ip.groupby("subject_id").session_year.nunique()
    subj = multi[multi > 1].index
    out = []
    for fam, tf, feats in (("vowel_a", "vowel_a", PHON), ("reading", "read_passage", CONN)):
        ref = S[(S.dataset == "IPVS") & (S.group != "YHC") & (S.family == fam)]
        for ft in feats:
            if ft not in ip:
                continue
            sd = transform(ref[ft], ft).std()
            ds = []
            for s in subj:
                d = ip[(ip.subject_id == s) & (ip.task_family == tf)]
                a = transform(d.loc[d.session_year == 2016, ft], ft).mean()
                b = transform(d.loc[d.session_year == 2017, ft], ft).mean()
                if np.isfinite(a) and np.isfinite(b) and sd > 0:
                    ds.append((b - a) / sd)
            if ds:
                out.append(dict(family=fam, feature=ft, n_patients=len(ds), mean_abs_shift_sd=float(np.mean(np.abs(ds))),
                                shifts_sd=", ".join(f"{v:+.2f}" for v in ds)))
    return pd.DataFrame(out)


# ============================================================================= figures
XLIM = 2.5


def figures(M):
    other_color = {"vowel_a": C["FIGSHARE"], "reading": C["MDVR"]}
    other_name = {"vowel_a": "Figshare (age-matched)", "reading": "MDVR-KCL"}
    title = {"vowel_a": "Sustained /a/", "reading": "Reading passage"}
    fams = [f for f in ["vowel_a", "reading"] if f in set(M.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(6.2 * len(fams), 7.2))
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        d = M[M.family == fam].sort_values("beta_re")
        y = np.arange(len(d))
        ax.axvline(0, color=C["ink2"], linewidth=1)
        ax.errorbar(d.ipvs_beta, y + 0.22, fmt="o", ms=4.5, color=C["IPVS"], label="IPVS (Italian)",
                    markeredgecolor=C["surface"])
        ax.errorbar(d.other_beta, y, fmt="o", ms=4.5, color=other_color[fam], label=other_name[fam],
                    markeredgecolor=C["surface"])
        for yi, r in zip(y, d.itertuples()):
            ax.plot([r.ci_lo, r.ci_hi], [yi - 0.22] * 2, color=C["ink"], linewidth=1.5, solid_capstyle="round")
            ax.scatter(r.beta_re, yi - 0.22, marker="D", s=30, zorder=3,
                       color=C["ink"] if r.robust else C["surface"], edgecolors=C["ink"], linewidths=1.2)
        ax.set_xlim(-XLIM, XLIM)
        for yi, r in zip(y, d.itertuples()):
            for val, off in ((r.ipvs_beta, 0.22), (r.other_beta, 0.0), (r.beta_re, -0.22)):
                if abs(val) > XLIM:
                    ax.text(np.sign(val) * XLIM, yi + off, " >" if val > 0 else "< ", fontsize=7, color=C["ink2"],
                            ha="right" if val > 0 else "left", va="center")
        labels = [("★ " if t.startswith(("1", "2")) else "") + lab for lab, t in zip(d.label, d.tier)]
        ax.set_yticks(y, labels)
        ax.set_xlabel("PD - control (SD units, adjusted)")
        ax.set_title(title[fam], loc="left")
        ax.grid(axis="y", visible=False)
        ax.scatter([], [], marker="D", s=30, color=C["ink"], label="Pooled, 95% CI (filled = room-robust)")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-dataset and pooled PD effects (★ = replicated candidate)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig4_forest.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(fams), figsize=(5.4 * len(fams), 4.4))
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        d = M[(M.family == fam) & M.room_effect_sd.notna()]
        d = d.assign(bx=np.abs(d.beta_re).clip(upper=XLIM), by=np.abs(d.room_effect_sd).clip(upper=XLIM))
        xm = min(XLIM, max(0.5, float(d.bx.max()) * 1.1))
        xx = np.linspace(0, xm, 50)
        ax.fill_between(xx, 0, 0.5 * xx, color=C["grid"], alpha=0.6, linewidth=0)
        ax.text(xm * 0.97, 0.5 * xm * 0.97, "robust zone\n(room < half of PD effect)", ha="right", va="top",
                fontsize=7.5, color=C["ink2"])
        cand = d.tier.str.startswith(("1", "2"))
        ax.scatter(d.bx[~cand], d.by[~cand], s=22, color=C["muted"], linewidths=0)
        ax.scatter(d.bx[cand], d.by[cand], s=40, color=C["IPVS"],
                   edgecolors=C["surface"], linewidths=1.5)
        for r in d[(d.p_re < 0.05) | cand].itertuples():
            ax.annotate(r.label, (r.bx, r.by), xytext=(4, 3), textcoords="offset points",
                        fontsize=7.5, color=C["ink2"])
        ax.set_xlim(0, xm)
        ax.set_ylim(0, XLIM)
        ax.set_xlabel("|Pooled PD effect| (SD, capped at 2.5)")
        ax.set_title(title[fam], loc="left")
    axes[0].set_ylabel("|Room effect| (SD, capped at 2.5)")
    fig.suptitle("Is each feature's PD effect bigger than what the recording room alone can produce?",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig5_robustness.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
