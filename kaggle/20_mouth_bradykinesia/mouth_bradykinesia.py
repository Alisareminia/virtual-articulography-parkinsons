"""PD voice project - step 20: bradykinesia of the mouth - virtual articulography (deep acoustic-to-articulatory inversion) in Parkinson's disease across 3 datasets

Helpers are copied verbatim from the step-13 script by tools/build_step.py; inputs are the outputs of earlier steps.
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "praat-parselmouth"], check=False)
import io
import os
import re
import warnings
from contextlib import redirect_stdout
from math import gcd
from pathlib import Path
import librosa
import matplotlib
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
import statsmodels.api as sm
from parselmouth.praat import call
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, resample_poly
from scipy.stats import mannwhitneyu, norm
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)
INPUT = Path(os.environ.get("PDV_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("PDV_OUTPUT", "/kaggle/working"))
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


def find_root_with(fname, sibling_dir=None):
    for root, dirs, files in os.walk(INPUT):
        if fname in files and (sibling_dir is None or sibling_dir in dirs):
            return Path(root)
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


def transform(v, feat):
    v = pd.to_numeric(v, errors="coerce").astype(float)
    if feat in LOG_FEATS:
        v = np.log(v.clip(lower=1e-6))
    med = v.median()
    mad = 1.4826 * (v - med).abs().median()
    if mad > 0:
        v = v.clip(med - 4 * mad, med + 4 * mad)   # tame extreme outliers before standardising
    return v


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


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for s in self.s:
            s.write(x)

    def flush(self):
        for s in self.s:
            s.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.s[0], name)
import tempfile
import uuid
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
KNN_BACKEND = os.environ.get("PDV_KNN_BACKEND", "hub")
SEG_S = 12.0
SR16 = 16000
TMPW = Path(tempfile.mkdtemp(prefix="pdv08_"))


class KNNVC:
    """Thin wrapper around bshall/knn-vc (WavLM-Large layer 6 features + prematched HiFi-GAN vocoder)."""

    def __init__(self):
        self.stub = KNN_BACKEND == "stub"
        if self.stub:
            rng = np.random.default_rng(0)
            self.P = rng.standard_normal((40, 1024)) / 6
            return
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True,
                                device=self.dev)
        print(f"  kNN-VC loaded on {self.dev}", flush=True)

    def frames(self, x, layer=6):
        """(T, 1024) WavLM frame features (20 ms hop) for a 16 kHz signal."""
        if self.stub:
            m = librosa.feature.melspectrogram(y=np.asarray(x, np.float32), sr=SR16, n_mels=40, hop_length=320)
            return np.log(m + 1e-6).T @ self.P * (1.0 if layer == 6 else 0.7)
        torch = self.torch
        out, step = [], 20 * SR16  # 20 s chunks keep full-attention memory bounded on long recordings
        for s in range(0, len(x), step):
            seg = np.asarray(x[s:s + step], np.float32)
            if len(seg) < 400:
                continue
            with torch.inference_mode():
                feats = self.m.wavlm.extract_features(torch.tensor(seg)[None].to(self.dev), output_layer=layer,
                                                      ret_layer_results=False)[0]
            out.append(feats[0].float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def vocode(self, F):
        if self.stub:  # crude stand-in: pitch and level follow the features, so the pipeline can be exercised
            n = F.shape[0] * 320
            t = np.arange(n) / SR16
            f0 = 120 + 40 * np.tanh(np.repeat(F[:, 0], 320)[:n])
            amp = 0.1 * (1 + 0.5 * np.tanh(np.repeat(F[:, 1], 320)[:n]))
            return amp * np.sin(2 * np.pi * np.cumsum(f0) / SR16) + 1e-3 * np.random.default_rng(0).standard_normal(n)
        torch = self.torch
        c = torch.tensor(np.asarray(F, np.float32))[None].to(self.dev)
        with torch.inference_mode():
            y = self.m.vocode(c) if hasattr(self.m, "vocode") else self.m.hifigan(c)
        return y.squeeze().float().cpu().numpy().astype(np.float64)


def tx(v, feat):
    """Plain transform (log for skewed measures), no outlier clipping - used for synthetic speech."""
    v = pd.to_numeric(pd.Series(v), errors="coerce").astype(float)
    return np.log(v.clip(lower=1e-6)) if feat in LOG_FEATS else v


def segment(x, sr=SR16, seg_s=SEG_S):
    """A speech-rich excerpt: starts 25% into the recording (after trimming), length seg_s."""
    n = int(seg_s * sr)
    if len(x) <= n:
        return x
    s = min(int(0.25 * len(x)), len(x) - n)
    return x[s:s + n]
import hashlib


def pitch_contour(x):
    snd = parselmouth.Sound(np.asarray(x, np.float64), sampling_frequency=SR16)
    lo, hi = f0_range_for(snd)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    return p.xs(), p.selected_array["frequency"], lo, hi


def impose_contour(y, t, f, lo, hi, compress=1.0):
    """Replace y's pitch with the given contour (optionally compressed around its median, in semitones)."""
    v = f > 0
    if v.sum() < 10:
        return y
    med = float(np.median(f[v]))
    target = med * (f[v] / med) ** compress
    snd = parselmouth.Sound(np.asarray(y, np.float64), sampling_frequency=SR16)
    manip = call(snd, "To Manipulation", 0.01, lo, hi)
    tier = call("Create PitchTier", "target", snd.xmin, snd.xmax)
    for ti, fi in zip(t[v], target):
        if snd.xmin <= ti <= snd.xmax:
            call(tier, "Add point", float(ti), float(np.clip(fi, 40, 900)))
    call([tier, manip], "Replace pitch tier")
    return call(manip, "Get resynthesis (overlap-add)").values[0]


def scale_pauses(x, factor):
    """Scale every internal pause (>= 150 ms) by factor, keeping >= 50 ms; uses the pause's own background audio."""
    if abs(factor - 1) < 1e-3:
        return x
    e, h = frames_db(x, SR16)
    act = activity(e)
    out, last = [], 0
    for s, ln in runs(~act):
        if ln < 15 or s == 0 or s + ln >= len(act):
            continue
        a, b = s * h, (s + ln) * h
        seg = x[a:b]
        new_len = max(int(0.05 * SR16), int(len(seg) * factor))
        if new_len <= len(seg):
            c = (len(seg) - new_len) // 2
            seg2 = seg[c:c + new_len]
        else:
            reps = int(np.ceil(new_len / len(seg)))
            seg2 = np.concatenate([seg[::(-1) ** k] for k in range(reps)])[:new_len]  # mirrored tiling, no clicks
        out += [x[last:a], seg2]
        last = b
    out.append(x[last:])
    return np.concatenate(out)


def lengthen(x, factor, lo, hi):
    if abs(factor - 1) < 1e-3:
        return x
    snd = parselmouth.Sound(np.asarray(x, np.float64), sampling_frequency=SR16)
    return call(snd, "Lengthen (overlap-add)", lo, hi, float(factor)).values[0]
LOG_FEATS.update({"run_mean_s", "rate_cv"})


def dab_features(x, sr=SR16):
    """Acoustic correlates of the Mayo hypokinetic-dysarthria dimensions for one recording."""
    base = connected_features(x, sr)
    out = {k: base.get(k, np.nan) for k in ("f0_sd_st", "f0_range_st", "intensity_sd_db", "pause_ratio", "pause_mean_s",
                                            "hnr_db", "cpps_db", "f0_mean_hz")}
    snd = parselmouth.Sound(np.asarray(x, np.float64), sampling_frequency=sr)
    lo, hi = f0_range_for(snd)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    t, f = pitch.xs(), pitch.selected_array["frequency"]
    inten = snd.to_intensity(minimum_pitch=max(lo, 50), time_step=0.01)
    ti, vi = inten.xs(), inten.values[0]
    if len(t) < 10 or len(ti) < 10:
        return out

    def voiced_at(times):
        idx = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
        return f[idx]

    # syllable nuclei (voiced intensity peaks), as in step 03
    thr = np.percentile(vi, 99) - 25
    pk, _ = find_peaks(vi, prominence=2, distance=5)
    fpk = voiced_at(ti[pk])
    pk = pk[(vi[pk] > thr) & (fpk > 0)]
    if len(pk) >= 5:
        out["stress_int_sd_db"] = float(np.std(vi[pk]))
        fp = voiced_at(ti[pk])
        fp = fp[fp > 0]
        if len(fp) >= 5:
            out["stress_f0_sd_st"] = float(np.std(12 * np.log2(fp / np.median(fp))))

    # speech runs between pauses (pauses = internal inactive stretches >= 150 ms)
    e, h = frames_db(x, sr)
    act = activity(e)
    pauses = [(s, ln) for s, ln in runs(~act) if ln * h / sr >= 0.15 and s > 0 and s + ln < len(act)]
    first = int(np.argmax(act)) if act.any() else 0
    last = len(act) - 1 - int(np.argmax(act[::-1])) if act.any() else len(act) - 1
    bounds, cur = [], first
    for s, ln in pauses:
        bounds.append((cur, s))
        cur = s + ln
    bounds.append((cur, last + 1))
    durs, rates = [], []
    nuc_t = ti[pk]
    for a, b in bounds:
        da = (b - a) * h / sr
        if da < 0.1:
            continue
        durs.append(da)
        n_nuc = int(((nuc_t >= a * h / sr) & (nuc_t < b * h / sr)).sum())
        if da >= 0.3 and n_nuc >= 2:
            rates.append(n_nuc / da)
    if durs:
        out["run_mean_s"] = float(np.mean(durs))
    if len(rates) >= 3:
        out["rate_cv"] = float(np.std(rates) / np.mean(rates))

    # consonant precision: spectrum of active but unvoiced frames (fricatives, bursts)
    n = int(0.025 * sr)
    P = np.abs(librosa.stft(x, n_fft=n, hop_length=h, win_length=n, center=False)) ** 2
    k = min(P.shape[1], len(act))
    centers = (np.arange(k) * h + n / 2) / sr
    unv = act[:k] & (voiced_at(centers) <= 0)
    fr = np.fft.rfftfreq(n, 1 / sr)
    if unv.sum() >= 10:
        p = P[:, :k][:, unv].mean(axis=1)
        m = (fr >= 50) & (fr <= 8000)
        out["cons_centroid_hz"] = float((fr[m] * p[m]).sum() / (p[m].sum() + 1e-20))
        out["cons_hf_db"] = float(10 * np.log10(p[(fr >= 4000) & (fr <= 8000)].sum() / (p[(fr >= 50) & (fr < 4000)].sum() + 1e-20) + 1e-20))

    # breathiness: H1-H2 on voiced frames (uncorrected harmonic amplitudes from a 64 ms spectrum)
    nb = int(0.064 * sr)
    vt = t[f > 0]
    vf = f[f > 0]
    if len(vt) >= 20:
        sel = np.linspace(0, len(vt) - 1, min(200, len(vt))).astype(int)
        d = []
        frb = np.fft.rfftfreq(nb, 1 / sr)
        for i in sel:
            c = int(vt[i] * sr)
            seg = x[max(0, c - nb // 2): c + nb // 2]
            if len(seg) < nb:
                continue
            spec = 20 * np.log10(np.abs(np.fft.rfft(seg * np.hanning(nb))) + 1e-12)

            def amp(fq):
                band = (frb > fq * 0.8) & (frb < fq * 1.2)
                return spec[band].max() if band.any() else np.nan
            d.append(amp(vf[i]) - amp(2 * vf[i]))
        if d:
            out["h1h2_db"] = float(np.nanmedian(d))
    return out
from scipy.signal import butter as _butter, sosfiltfilt
from scipy.stats import spearmanr


def hedges_g(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    n1, n0 = len(a), len(b)
    if n1 < 3 or n0 < 3:
        return np.nan, np.nan
    sp = np.sqrt(((n1 - 1) * a.var() + (n0 - 1) * b.var()) / (n1 + n0 - 2))
    if not sp > 0:
        return np.nan, np.nan
    j = 1 - 3 / (4 * (n1 + n0) - 9)
    g = j * (a.mean() - b.mean()) / sp
    return g, np.sqrt((n1 + n0) / (n1 * n0) + g ** 2 / (2 * (n1 + n0)))


def resample(x, sr, target):
    if sr == target:
        return x
    g_ = gcd(int(sr), int(target))
    return resample_poly(x, target // g_, sr // g_)
LOG_FEATS.update({"tremor_f0_pct", "voice_break_frac", "f0_drift_st_s", "tvsa_hz2", "f0_sd_st_r"})


def _contour_bandpass(v, fs=100.0, lo=3.0, hi=8.0):
    sos = _butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, v)


def vowel_extras(x, sr):
    """Vocal tremor (3-8 Hz modulation of F0 and intensity), voice breaks, pitch drift, H1-H2 on a sustained vowel."""
    out = {}
    snd = parselmouth.Sound(np.asarray(x, np.float64), sampling_frequency=sr)
    lo, hi = f0_range_for(snd)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    t, f = p.xs(), p.selected_array["frequency"]
    if (f > 0).sum() < 30:
        return out
    i0, i1 = np.where(f > 0)[0][[0, -1]]
    a, b = i0 + int(0.1 * (i1 - i0)), i1 - int(0.1 * (i1 - i0))
    fc, tc = f[a:b + 1], t[a:b + 1]
    out["voice_break_frac"] = float(np.mean(fc <= 0))
    v = fc > 0
    if v.mean() >= 0.8 and len(fc) >= 150:
        st = 12 * np.log2(np.interp(tc, tc[v], fc[v]) / np.median(fc[v]))
        out["f0_drift_st_s"] = float(abs(np.polyfit(tc, st, 1)[0]))
        resid = st - np.polyval(np.polyfit(tc, st, 2), tc)
        bp = _contour_bandpass(resid)
        out["tremor_f0_pct"] = float((2 ** (np.std(bp) / 12) - 1) * 100)
        inten = snd.to_intensity(minimum_pitch=max(lo, 50), time_step=0.01)
        iv = np.interp(tc, inten.xs(), inten.values[0])
        ir = iv - np.polyval(np.polyfit(tc, iv, 2), tc)
        out["tremor_amp_db"] = float(np.std(_contour_bandpass(ir)))
    nb = int(0.064 * sr)
    frb = np.fft.rfftfreq(nb, 1 / sr)
    d = []
    for i in np.linspace(0, len(tc) - 1, min(100, len(tc))).astype(int):
        if fc[i] <= 0:
            continue
        c = int(tc[i] * sr)
        seg = x[max(0, c - nb // 2): c + nb // 2]
        if len(seg) < nb:
            continue
        spec = 20 * np.log10(np.abs(np.fft.rfft(seg * np.hanning(nb))) + 1e-12)
        h1 = spec[(frb > fc[i] * 0.8) & (frb < fc[i] * 1.2)]
        h2 = spec[(frb > 2 * fc[i] * 0.8) & (frb < 2 * fc[i] * 1.2)]
        if len(h1) and len(h2):
            d.append(h1.max() - h2.max())
    if d:
        out["h1h2_db"] = float(np.median(d))
    return out


def icc21(a, b):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    X = np.column_stack([a, b])
    X = X[np.isfinite(X).all(axis=1)]
    n, k = X.shape
    if n < 5:
        return np.nan
    gm = X.mean()
    msr = k * ((X.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))
from scipy.signal import butter, firwin2, sosfilt


def md_table(df):
    if df is None or len(df) == 0:
        return "-"
    df = df.copy()
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def all_features(y):
    global connected_features
    base = connected_features(y, SR16)
    _orig = connected_features
    connected_features = lambda x, sr: base  # noqa: E731  (dab_features reuses the same computation)
    try:
        f = dab_features(y)
    finally:
        connected_features = _orig
    for k in ("speech_rate_syl_s", "artic_rate_syl_s", "pause_rate_min", "alpha_ratio_db", "hammarberg_db",
              "hf_ratio_db", "centroid_hz"):
        f[k] = base.get(k, np.nan)
    return f


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z (rank residuals)."""
    r = pd.DataFrame({"x": x, "y": y, "z": z}).dropna().rank()
    if len(r) < 8:
        return np.nan, np.nan
    rx = r.x - np.polyval(np.polyfit(r.z, r.x, 1), r.z)
    ry = r.y - np.polyval(np.polyfit(r.z, r.y, 1), r.z)
    return spearmanr(rx, ry)
from sklearn.linear_model import Ridge as _Ridge


def vowel_feats(y):
    f = phonation_features(y, SR16)
    f.update(vowel_extras(y, SR16))
    return f
CH = ["TDX", "TDY", "TBX", "TBY", "TTX", "TTY", "LIX", "LIY", "ULX", "ULY", "LLX", "LLY"]
CHI = {c: i for i, c in enumerate(CH)}
FS_EMA = 50.0


def progress(msg):
    with open(OUT / "progress.txt", "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        fh.flush()
        os.fsync(fh.fileno())


class Inverter:
    """SPARC acoustic-to-articulatory inversion (or a crude stand-in for local dry runs)."""

    def __init__(self, name):
        self.name = name
        if KNN_BACKEND == "stub":
            rng = np.random.default_rng(len(name))
            self.P = rng.standard_normal((40, 12)) / 6
            return
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speech-articulatory-coding"], check=False)
        import torch
        from sparc import load_model
        self.coder = load_model(name, device="cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"  SPARC '{name}' loaded", flush=True)

    def ema(self, x):
        """(L, 12) EMA at 50 Hz for a 16 kHz signal."""
        if KNN_BACKEND == "stub":
            m = librosa.feature.melspectrogram(y=np.asarray(x, np.float32), sr=SR16, n_mels=40, hop_length=320)
            return np.log(m + 1e-6).T @ self.P
        out = self.coder.encode(np.asarray(x, np.float32))
        e = out["ema"]
        e = e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e)
        return e.reshape(-1, 12).astype(np.float64)


def phone_band(x):
    y = sosfilt(butter(4, [300, 3400], "bandpass", fs=SR16, output="sos"), x)
    return resample(resample(y, SR16, 8000), 8000, SR16)


def central(E, frac=0.6):
    n = len(E)
    a = int(n * (1 - frac) / 2)
    return E[a:n - a] if n - 2 * a >= 3 else E


def lip_aperture(E):
    return E[:, CHI["ULY"]] - E[:, CHI["LLY"]]


def tremor_power(sig, fs=FS_EMA):
    """Share of movement power in the 3-7 Hz tremor band (1-15 Hz reference), after removing slow drift."""
    sig = np.asarray(sig, float)
    if len(sig) < 50:
        return np.nan
    sig = sig - np.polyval(np.polyfit(np.arange(len(sig)), sig, 2), np.arange(len(sig)))
    f, p = welch(sig, fs=fs, nperseg=min(len(sig), 100))
    ref = (f >= 1) & (f <= 15)
    return float(p[(f >= 3) & (f <= 7)].sum() / (p[ref].sum() + 1e-12))


def adj_effect(d, feat):
    """PD - healthy on ORIGINAL recordings: Hedges g and sex(+age)-adjusted beta (SD units), HC3 p, AUC."""
    d = d[["label", "male", "age", feat]].replace([np.inf, -np.inf], np.nan).dropna(subset=["label", feat])
    a, b = d.loc[d.label == 1, feat], d.loc[d.label == 0, feat]
    if len(a) < 5 or len(b) < 5:
        return None
    g, _ = hedges_g(a, b)
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    X = pd.DataFrame({"const": 1.0, "pd": d.label.astype(float), "male": d.male}, index=d.index)
    if d.age.notna().all() and d.age.std() > 0:
        X["age"] = (d.age - d.age.mean()) / d.age.std()
    fit = sm.OLS((d[feat] - d[feat].mean()) / sp, X).fit(cov_type="HC3")
    from sklearn.metrics import roc_auc_score
    return dict(n_pd=len(a), n_hc=len(b), median_pd=a.median(), median_hc=b.median(), g=g, beta=float(fit.params["pd"]),
                p=float(fit.pvalues["pd"]), auc=float(roc_auc_score(d.label, d[feat])))


# =====================================================================================================================
# STEP 20 - Bradykinesia of the mouth: virtual articulography (deep acoustic-to-articulatory inversion, SPARC) in
# Parkinson's disease across 3 datasets
# (helpers above: steps 03-19 verbatim; SPARC validated in the step-19 pilot; audio from step 02, sex from step 03)
#
# A. "Finger-tapping test of the mouth" (IPVS: 'pa' 5 s - 20 s pause - 'ta' 5 s, two takes): each recording is split at
#    the long pause; lips in 'pa', tongue tip in 'ta': movement SIZE, SPEED, rhythm and FADING over repetitions
#    (the MDS-UPDRS bradykinesia triad), PD vs healthy, sex/age adjusted, test-retest.
# B. Connected speech (MDVR English reading, IPVS Italian reading): size and speed of each articulator (upper lip, lower
#    lip, jaw, tongue tip / body / back), lip opening and tongue working space; adjusted for speech rate as a check.
# C. Severity (MDVR patients): H&Y, UPDRS III-18, UPDRS II speech item.
# D. Where is the tremor? 3-7 Hz tremor of jaw / tongue / lips during sustained vowels (IPVS 10 vowels, Figshare /a/)
#    vs the voice-box loudness / pitch tremor; IPVS also degraded to phone band (is the Figshare result a quality issue?);
#    per-patient tremor map.
# =====================================================================================================================
import time  # noqa: E402

from scipy.signal import find_peaks as _fp, welch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ARTIC = ["UL", "LL", "LI", "TT", "TB", "TD"]
ARTIC_NAME = {"UL": "upper lip", "LL": "lower lip", "LI": "jaw", "TT": "tongue tip", "TB": "tongue body",
              "TD": "tongue back", "LA": "lip opening"}
CHUNK_S = 20.0


def ema_long(inv, x):
    """EMA for long recordings in 20-s chunks (bounded GPU memory), concatenated at 50 Hz."""
    n = int(CHUNK_S * SR16)
    parts = [inv.ema(x[s:s + n]) for s in range(0, len(x), n) if len(x[s:s + n]) >= 0.5 * SR16]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 12))


def frame_activity(x, L):
    e, _ = frames_db(x, SR16, 0.04, 0.02)
    act = activity(e)
    return np.concatenate([act, np.zeros(max(0, L - len(act)), bool)])[:L]


def cycles(sig, fs=50.0):
    """Repetition cycles of one articulator: amplitude, speed, rate, rhythm and fading over the repetitions."""
    sig = np.asarray(sig, float)
    if len(sig) < 40:
        return {}
    t = np.arange(len(sig))
    s = sig - np.polyval(np.polyfit(t, sig, 1), t)
    s = uniform_filter1d(s, 3)
    iqr = np.subtract(*np.percentile(s, [75, 25]))
    pk, _ = _fp(s, distance=4, prominence=0.3 * iqr if iqr > 0 else None)
    tr, _ = _fp(-s, distance=4, prominence=0.3 * iqr if iqr > 0 else None)
    if len(pk) < 5 or len(tr) < 4:
        return {}
    amps = []
    for p in pk:
        before, after = tr[tr < p], tr[tr > p]
        if len(before) and len(after):
            amps.append(s[p] - 0.5 * (s[before[-1]] + s[after[0]]))
    amps = np.array([a for a in amps if a > 0])
    if len(amps) < 4:
        return {}
    iv = np.diff(pk) / fs
    k = max(1, len(amps) // 3)
    v = np.abs(np.gradient(sig) * fs)
    first = amps[:k].mean()
    return dict(amp=float(np.median(amps)), speed=float(np.percentile(v, 95)), rate_hz=float(1 / np.median(iv)),
                rhythm_cv=float(np.std(iv) / np.mean(iv)), amp_cv=float(np.std(amps) / np.mean(amps)),
                fading=float(amps[-k:].mean() / first - 1),                       # < 0 = movements get smaller
                fading_slope=float(np.polyfit(np.arange(len(amps)), amps / first, 1)[0] * 100),   # % per cycle
                n_cycles=int(len(amps)), amp_series=(amps / first).tolist())


def artic_signal(E, a):
    if a == "LA":
        return lip_aperture(E)
    return E[:, CHI[f"{a}Y"]]


def connected_kinematics(E, act):
    Ea = E[act] if act.sum() >= 50 else E
    out = {}
    for a in ARTIC + ["LA"]:
        y = artic_signal(Ea, a)
        out[f"{a}_range"] = float(np.percentile(y, 95) - np.percentile(y, 5))
        if a == "LA":
            vel = np.abs(np.gradient(y)) * FS_EMA
        else:
            vel = np.hypot(np.gradient(Ea[:, CHI[f"{a}X"]]), np.gradient(Ea[:, CHI[f"{a}Y"]])) * FS_EMA
        out[f"{a}_speed"] = float(np.percentile(vel, 95))
        out[f"{a}_mean_speed"] = float(np.mean(vel))
    cov = np.cov(Ea[:, CHI["TBX"]], Ea[:, CHI["TBY"]])
    out["TB_space"] = float(np.sqrt(max(np.linalg.det(cov), 0)))
    return out


def step20():
    OUT.mkdir(parents=True, exist_ok=True)
    s02 = find_root_with("manifest.csv", "clean_audio")
    s03 = find_root_with("features_subject.csv")
    man = pd.read_csv(s02 / "manifest.csv")
    S3 = pd.read_csv(s03 / "features_subject.csv")
    sex_map = S3.drop_duplicates("subject_id").set_index("subject_id").sex
    man["sex"] = man.subject_id.map(sex_map).fillna(man.get("sex"))
    man = man[man.group.isin(["PD", "HC", "EHC"]) & man.sex.isin(["F", "M"])]
    man["age"] = pd.to_numeric(man.age, errors="coerce")

    def load(p, trim=True):
        x, sr = sf.read(str(s02 / "clean_audio" / p), dtype="float64")
        x = x.mean(axis=1) if x.ndim > 1 else x
        x = x if sr == SR16 else resample(x, sr, SR16)
        return trim_and_normalise(x, SR16) if trim else x

    inv = Inverter("multi")
    meta = lambda r: dict(sid=r.subject_id, label=int(r.label), sex=r.sex, male=float(r.sex == "M"), age=r.age,  # noqa: E731
                          hy=r.hoehn_yahr, updrs3=r.updrs_iii18, updrs2=r.updrs_ii5)

    # ------------------------------------------------------------------ A. pa / ta
    section("A. FINGER-TAPPING TEST OF THE MOUTH (IPVS: one 5-s 'pa' file and one 5-s 'ta' file per speaker)")
    DD, SERIES_RAW, EXAMPLES = [], [], {}
    ddk = man[(man.dataset == "IPVS") & (man.task_family == "syllables")]
    for r in ddk.itertuples():
        try:
            x = load(r.clean_path)
            E = ema_long(inv, x)
            act = frame_activity(x, len(E))
            idx = np.where(act)[0]
            a, b = (idx[0], idx[-1] + 1) if len(idx) >= 50 else (0, len(E))
            seg = E[a:b]
            la, tt = lip_aperture(seg), seg[:, CHI["TTY"]]
            row = dict(**meta(r), file=r.task, dur_s=(b - a) / 50,
                       lip_closure=float(np.percentile(la, 5)), tt_peak=float(np.percentile(tt, 95)))
            for art in ("LA", "TT", "LI", "TB"):
                for k, v in cycles(artic_signal(seg, art)).items():
                    if k == "amp_series":
                        SERIES_RAW.append(dict(sid=r.subject_id, label=int(r.label), file=r.task, art=art, series=v))
                    else:
                        row[f"{art}_{k}"] = v
            DD.append(row)
            EXAMPLES[(r.subject_id, r.task)] = (x, E, (a, b))
        except Exception as ex:  # noqa: BLE001
            print(f"    ddk failed {r.subject_id} {r.task}: {ex}")
    D = pd.DataFrame(DD)
    D.to_csv(OUT / "ddk_recordings.csv", index=False)
    progress("A done")
    # which file is 'pa'? In 'pa' the lips close fully each cycle; in 'ta' the tongue tip rises to the gum ridge.
    VAL, pa_file, valid = [], None, False
    files = sorted(D.file.unique()) if len(D) else []
    W = D.pivot_table(index=["sid", "label"], columns="file", values=["lip_closure", "tt_peak"]).dropna() if len(files) == 2 else None
    if W is not None and len(W) >= 6:
        f1, f2 = files
        dl = W["lip_closure"][f1] - W["lip_closure"][f2]      # < 0: lips close more in f1
        dt = W["tt_peak"][f1] - W["tt_peak"][f2]              # < 0: tongue tip higher in f2
        pa_file = f1 if dl.median() < 0 else f2
        labs = W.index.get_level_values("label")
        for grp, lab_ in ((0, "healthy"), (1, "PD"), (None, "all")):
            sel = np.ones(len(W), bool) if grp is None else labs == grp
            if sel.sum() >= 6:
                VAL.append(dict(group=lab_, n=int(sel.sum()), pa_file=pa_file,
                                lip_closure_f1_minus_f2=float(dl[sel].mean()), p_lips=float(wilcoxon(dl[sel]).pvalue),
                                tongue_tip_f1_minus_f2=float(dt[sel].mean()), p_tongue=float(wilcoxon(dt[sel]).pvalue),
                                consistent=bool(np.sign(dl[sel].median()) == np.sign(dt[sel].median()))))
        h = [v for v in VAL if v["group"] == "healthy"]
        valid = bool(h and h[0]["consistent"] and h[0]["p_lips"] < 0.05 and h[0]["p_tongue"] < 0.05)
    VAL = pd.DataFrame(VAL)
    VAL.to_csv(OUT / "ddk_validity.csv", index=False)
    print("  validity: lips close more in one file AND the tongue tip rises more in the other (healthy speakers decide):")
    print(VAL.round(4).to_string(index=False) if len(VAL) else "  -")
    print(f"  -> 'pa' file = {pa_file}; pa/ta validity in healthy speakers: {'PASSED' if valid else 'NOT PASSED'}")
    S = pd.DataFrame(columns=["sid", "label", "male", "age"])
    SERIES = []
    if pa_file is not None:
        D2 = D.assign(task=np.where(D.file == pa_file, "pa", "ta"))
        num = [c for c in D2.columns if c.startswith(("LA_", "TT_", "LI_", "TB_")) and D2[c].dtype != object]
        wide = D2.pivot_table(index="sid", columns="task", values=num)
        wide.columns = [f"{t}_{c}" for c, t in wide.columns]
        S = wide.join(D2.drop_duplicates("sid").set_index("sid")[["label", "male", "age"]]).reset_index()
        SERIES = [dict(sid=s_["sid"], label=s_["label"], task="pa" if s_["file"] == pa_file else "ta", series=s_["series"])
                  for s_ in SERIES_RAW if (s_["art"] == "LA" and s_["file"] == pa_file) or (s_["art"] == "TT" and s_["file"] != pa_file)]
    prim = [f"{part}_{art}_{m}" for part, art in (("pa", "LA"), ("ta", "TT"))
            for m in ("amp", "speed", "rate_hz", "rhythm_cv", "fading", "fading_slope")]
    sec_ = [f"{p}_{a}_{m}" for p in ("pa", "ta") for a in ("LI", "TB") for m in ("amp", "speed")]
    EA = []
    for col in prim + sec_:
        if col in S:
            e = adj_effect(S, col)
            if e:
                p_, a_, m_ = col.split("_", 2)
                EA.append(dict(section="A pa/ta" + ("" if valid else " (validity NOT passed)"), measure=col, task=p_,
                               articulator=ARTIC_NAME.get(a_, a_), quantity=m_, primary=col in prim, **e))
    EA = pd.DataFrame(EA)
    if len(EA):
        EA["q"] = np.nan
        EA.loc[EA.primary, "q"] = bh(EA.loc[EA.primary, "p"])
    print(EA.round(3).to_string(index=False) if len(EA) else "  -")
    REL = []   # no repeated pa / ta takes in IPVS: reliability comes from the two reading takes (section B)

    # ------------------------------------------------------------------ B. connected speech
    section("B. CONNECTED SPEECH: size and speed of each articulator (reading)")
    CS = []
    reading = man[((man.dataset == "MDVR") & (man.task_family == "read_passage")) |
                  ((man.dataset == "IPVS") & (man.task_family == "read_passage"))]
    for r in reading.itertuples():
        try:
            x = load(r.clean_path)
            E = ema_long(inv, x)
            act = frame_activity(x, len(E))
            k = connected_kinematics(E, act)
            rate = connected_features(x, SR16).get("speech_rate_syl_s", np.nan)
            CS.append(dict(**meta(r), cohort=f"{r.dataset} reading", take=r.task, speech_rate=rate, **k))
        except Exception as ex:  # noqa: BLE001
            print(f"    reading failed {r.subject_id}: {ex}")
    CSd = pd.DataFrame(CS)
    CSd.to_csv(OUT / "reading_recordings.csv", index=False)
    progress("B done")
    kin = [c for c in CSd.columns if c.endswith(("_range", "_speed", "_space")) and not c.endswith("mean_speed")]
    EB = []
    for coh, g in CSd.groupby("cohort"):
        Sg = g.groupby("sid").agg({**{c: "mean" for c in kin + ["speech_rate"]}, "label": "first", "male": "first",
                                    "age": "first", "hy": "first", "updrs3": "first", "updrs2": "first"}).reset_index()
        for col in kin:
            e = adj_effect(Sg, col)
            if e:
                a_, m_ = col.split("_", 1)
                Sr = Sg.dropna(subset=[col, "speech_rate"])
                X = pd.DataFrame({"const": 1.0, "pd": Sr.label, "male": Sr.male,
                                  "rate": (Sr.speech_rate - Sr.speech_rate.mean()) / (Sr.speech_rate.std() or 1)})
                sd_ = Sr[col].std() or 1
                f_ = sm.OLS((Sr[col] - Sr[col].mean()) / sd_, X).fit(cov_type="HC3")
                EB.append(dict(section=f"B {coh}", measure=col, articulator=ARTIC_NAME.get(a_, a_), quantity=m_, primary=True,
                               beta_rate_adjusted=float(f_.params["pd"]), p_rate_adjusted=float(f_.pvalues["pd"]), **e))
        if coh.startswith("IPVS"):
            for col in kin:
                w = g.pivot_table(index="sid", columns="take", values=col)
                if w.shape[1] >= 2:
                    w = w.iloc[:, :2].dropna()
                    if len(w) >= 10:
                        REL.append(dict(section="B IPVS reading", measure=col, n=len(w),
                                        icc=icc21(w.iloc[:, 0].to_numpy(), w.iloc[:, 1].to_numpy())))
    EB = pd.DataFrame(EB)
    if len(EB):
        EB["q"] = bh(EB.p)
    print(EB.round(3).to_string(index=False) if len(EB) else "  -")
    if len(EB):
        meta_rows = []
        for col, g in EB.groupby("measure"):
            if len(g) == 2:
                se = np.abs(g.beta) / np.maximum(norm.isf(g.p.clip(1e-12, 1) / 2), 1e-6)
                mm = dl_meta(g.beta, se)
                meta_rows.append(dict(measure=col, beta=mm["beta_re"], ci_lo=mm["ci_lo"], ci_hi=mm["ci_hi"], p=mm["p_re"],
                                      i2=mm["i2"], same_direction=bool(np.sign(g.beta).nunique() == 1)))
        MB = pd.DataFrame(meta_rows)
        if len(MB):
            MB["q"] = bh(MB.p)
            MB.to_csv(OUT / "reading_pooled_EN_IT.csv", index=False)
            print("\n  pooled English + Italian:")
            print(MB.sort_values("p").round(3).to_string(index=False))

    # ------------------------------------------------------------------ C. severity
    section("C. SEVERITY (MDVR patients, Spearman; 'adj' = partial rho controlling for speech rate)")
    SV = []
    g = CSd[(CSd.cohort == "MDVR reading") & (CSd.label == 1)]
    for col in kin:
        for sc in ("hy", "updrs3", "updrs2"):
            dd = g[[col, sc, "speech_rate"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(dd) >= 8 and dd[col].std() > 0:
                r_, p_ = spearmanr(dd[col], dd[sc])
                ra, pa = partial_spearman(dd[col], dd[sc], dd.speech_rate)
                SV.append(dict(measure=col, scale=sc, n=len(dd), rho=r_, p=p_, rho_adj_rate=ra, p_adj_rate=pa))
    SV = pd.DataFrame(SV)
    if len(SV):
        SV["q"] = bh(SV.p)
    print(SV.sort_values("p").round(3).head(20).to_string(index=False) if len(SV) else "  -")

    # ------------------------------------------------------------------ D. tremor map
    section("D. WHERE IS THE TREMOR? mouth (jaw / tongue / lips, 3-7 Hz) vs voice box (loudness / pitch), sustained vowels")
    TR = []
    vow = man[((man.dataset == "IPVS") & man.task_family.str.startswith("vowel")) |
              ((man.dataset == "FIGSHARE") & (man.task_family == "vowel_a"))]
    fs_ = vow[vow.dataset == "FIGSHARE"]
    amin = fs_.loc[fs_.label == 1, "age"].min()
    vow = vow[(vow.dataset != "FIGSHARE") | (vow.label == 1) | (vow.age >= amin)]
    for r in vow.itertuples():
        try:
            x = load(r.clean_path)
            versions = [("clean", x)] + ([("phone", phone_band(x))] if r.dataset == "IPVS" else [])
            for cond, xx in versions:
                C_ = central(inv.ema(xx))
                ve = vowel_extras(xx, SR16)
                TR.append(dict(**meta(r), cohort=f"{r.dataset} vowels", cond=cond, vowel=r.task_family[-1],
                               **{f"mouth_{a}": tremor_power(C_[:, CHI[f'{a}Y']]) for a in ("LI", "TT", "TB", "TD", "LL")},
                               voice_loudness=ve.get("tremor_amp_db", np.nan), voice_pitch=ve.get("tremor_f0_pct", np.nan)))
        except Exception as ex:  # noqa: BLE001
            print(f"    vowel failed {r.subject_id}: {ex}")
    T = pd.DataFrame(TR)
    T.to_csv(OUT / "tremor_recordings.csv", index=False)
    progress("D done")
    tcols = [c for c in T.columns if c.startswith(("mouth_", "voice_"))]
    for c in ("voice_loudness", "voice_pitch"):
        T[c] = np.log(T[c].clip(lower=1e-6))
    ED, MAP = [], []
    for (coh, cond), g in T.groupby(["cohort", "cond"]):
        Sg = g.groupby("sid").agg({**{c: "mean" for c in tcols}, "label": "first", "male": "first", "age": "first"}).reset_index()
        for col in tcols:
            e = adj_effect(Sg, col)
            if e:
                ED.append(dict(section=f"D {coh} ({cond})", measure=col, primary=cond == "clean", **e))
        if cond == "clean":
            Sg["mouth_tremor"] = Sg[["mouth_LI", "mouth_TB", "mouth_TD"]].mean(axis=1)
            for col in ("mouth_tremor", "voice_loudness"):
                flags = np.zeros(len(Sg), bool)
                for i in range(len(Sg)):
                    ref = Sg[(Sg.label == 0) & (Sg.index != i)][col].dropna()
                    flags[i] = bool(len(ref) >= 5 and Sg[col].iloc[i] > np.percentile(ref, 90))
                Sg[f"{col}_high"] = flags
            ct = pd.crosstab([Sg.label.map({0: "healthy", 1: "PD"})],
                             [Sg.mouth_tremor_high.map({True: "mouth+", False: "mouth-"}),
                              Sg.voice_loudness_high.map({True: "voice+", False: "voice-"})])
            ct.to_csv(OUT / f"tremor_map_{coh.split()[0]}.csv")
            print(f"\n  tremor map {coh} (above the healthy 90th percentile):")
            print(ct.to_string())
            rho, p = spearmanr(Sg.mouth_tremor, Sg.voice_loudness, nan_policy="omit")
            MAP.append(dict(cohort=coh, rho_mouth_vs_voice=rho, p=p))
            Sg.to_csv(OUT / f"tremor_speakers_{coh.split()[0]}.csv", index=False)
    ED = pd.DataFrame(ED)
    if len(ED):
        ED["q"] = np.nan
        ED.loc[ED.primary, "q"] = bh(ED.loc[ED.primary, "p"])
    print(ED.round(3).to_string(index=False) if len(ED) else "  -")
    print("\n  does mouth tremor go with voice tremor?")
    print(pd.DataFrame(MAP).round(3).to_string(index=False))

    # ------------------------------------------------------------------ outputs
    ALL = pd.concat([EA, EB, ED], ignore_index=True)
    ALL.to_csv(OUT / "mouth_bradykinesia_effects.csv", index=False)
    SV.to_csv(OUT / "mouth_severity.csv", index=False)
    pd.DataFrame(REL).to_csv(OUT / "mouth_reliability.csv", index=False)
    section("RELIABILITY (two takes, ICC 2,1)")
    print(pd.DataFrame(REL).round(3).to_string(index=False) if REL else "  -")
    section("SUMMARY: strongest effects (q < 0.10)")
    if len(ALL):
        print(ALL[ALL.q < 0.10].sort_values("p")[["section", "measure", "n_pd", "n_hc", "g", "beta", "p", "q", "auc"]]
              .round(3).to_string(index=False))
    figures20(SERIES, ALL, CSd, T)
    # example: typical PD and healthy speaker (same sex): 'pa' lip opening and 'ta' tongue tip, with audio
    try:
        Sx = S.dropna(subset=["pa_LA_amp"]) if "pa_LA_amp" in S else S.iloc[0:0]
        picks = []
        for lb in (0, 1):
            c = Sx[(Sx.label == lb) & (Sx.male == 1.0)]
            c = c if len(c) else Sx[Sx.label == lb]
            if len(c):
                picks.append(c.iloc[(c.pa_LA_amp - c.pa_LA_amp.median()).abs().argmin()].sid)
        if len(picks) == 2:
            ta_file = [f for f in files if f != pa_file][0]
            fig, axs = plt.subplots(2, 2, figsize=(10, 4.6), sharey="col")
            for row_, sid, lb in zip(axs, picks, ("healthy", "Parkinson's")):
                pieces = []
                for ax, fl, art, nm in ((row_[0], pa_file, "LA", "lip opening in 'pa'"), (row_[1], ta_file, "TT", "tongue tip in 'ta'")):
                    if (sid, fl) not in EXAMPLES:
                        continue
                    x, E, (a, b) = EXAMPLES[(sid, fl)]
                    ax.plot(np.arange(b - a) / 50, artic_signal(E[a:b], art), color=C["FIGSHARE"] if art == "LA" else C["IPVS"])
                    ax.set_title(f"{lb}: {nm}", loc="left", fontsize=9)
                    pieces += [x[int(a * 320):int(b * 320)], np.zeros(SR16 // 2)]
                if pieces:
                    seg = np.concatenate(pieces)
                    sf.write(OUT / f"example_{'healthy' if lb == 'healthy' else 'parkinsons'}__pa_then_ta.ogg",
                             (seg / (np.max(np.abs(seg)) + 1e-9) * 0.9).astype(np.float32), SR16, format="OGG", subtype="VORBIS")
            for ax in axs[1]:
                ax.set_xlabel("seconds")
            fig.tight_layout()
            fig.savefig(OUT / "fig50_example_mouth_movements.png", dpi=150)
            plt.close(fig)
    except Exception as ex:  # noqa: BLE001
        print(f"  example figure skipped: {ex}")
    L = ["# Bradykinesia of the mouth - virtual articulography in Parkinson's disease", "",
         "## pa/ta validity", "", md_table(VAL.round(4)) if len(VAL) else "-", "",
         "## All effects (PD - healthy, sex/age adjusted; beta in SD units)", "",
         md_table(ALL[["section", "measure", "n_pd", "n_hc", "g", "beta", "p", "q", "auc"]].round(3)) if len(ALL) else "-", "",
         "## Severity (MDVR)", "", md_table(SV.round(3)) if len(SV) else "-", "",
         "## Reliability", "", md_table(pd.DataFrame(REL).round(3)) if REL else "-"]
    (OUT / "report_mouth_bradykinesia.md").write_text("\n".join(L))
    print("\nsaved: mouth_bradykinesia_effects.csv, mouth_severity.csv, mouth_reliability.csv, ddk_*.csv, reading_*.csv, "
          "tremor_*.csv, report_mouth_bradykinesia.md, fig46-50, example clips")


def figures20(SERIES, ALL, CSd, T):
    if SERIES:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
        for ax, (task, lab_) in zip(axs, (("pa", "lips in 'pa'"), ("ta", "tongue tip in 'ta'"))):
            for lb, col in ((0, C["MDVR"]), (1, C["FIGSHARE"])):
                ser = [s["series"][:25] for s in SERIES if s["task"] == task and s["label"] == lb and len(s["series"]) >= 8]
                if not ser:
                    continue
                L = max(len(s) for s in ser)
                M = np.full((len(ser), L), np.nan)
                for i, s in enumerate(ser):
                    M[i, :len(s)] = s
                m = np.nanmean(M, axis=0)
                n = np.sum(np.isfinite(M), axis=0)
                keep = n >= 5
                ax.plot(np.arange(1, L + 1)[keep], m[keep], color=col, lw=2, label="PD" if lb else "healthy")
            ax.axhline(1, color=C["muted"], lw=0.8)
            ax.set_xlabel("repetition number")
            ax.set_title(f"Movement size over repetitions: {lab_}", loc="left", fontsize=9)
        axs[0].set_ylabel("amplitude (relative to first repetitions)")
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "fig46_mouth_finger_tapping.png", dpi=150)
        plt.close(fig)
    d = ALL[ALL.primary == True] if len(ALL) else ALL  # noqa: E712
    if len(d):
        d = d.sort_values(["section", "beta"]).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, 0.8 + 0.22 * len(d)))
        y = np.arange(len(d))[::-1]
        cols = [C["MDVR"] if "MDVR" in s else C["IPVS"] if "IPVS" in s else C["FIGSHARE"] for s in d.section]
        ax.scatter(d.beta, y, c=cols, s=14, zorder=5)
        ax.axvline(0, color=C["muted"], lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{s} | {m}" for s, m in zip(d.section, d.measure)], fontsize=6)
        ax.set_xlabel("PD - healthy (SD units, sex/age adjusted)")
        ax.set_title("Virtual articulography: Parkinson's vs healthy", loc="left")
        fig.tight_layout()
        fig.savefig(OUT / "fig47_mouth_effects.png", dpi=150)
        plt.close(fig)
    g = CSd[(CSd.cohort == "MDVR reading") & (CSd.label == 1)] if len(CSd) else CSd
    if len(g) >= 8:
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        for ax, col in zip(axs, ("LA_range", "TT_speed", "TB_space")):
            ax.scatter(pd.to_numeric(g.updrs3, errors="coerce"), g[col], color=C["FIGSHARE"], s=18)
            ax.set_xlabel("UPDRS III-18")
            ax.set_title(col, loc="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "fig48_mouth_severity.png", dpi=150)
        plt.close(fig)
    if len(T):
        t = T[T.cond == "clean"].copy()
        t["mouth_tremor"] = t[["mouth_LI", "mouth_TB", "mouth_TD"]].mean(axis=1)
        s = t.groupby(["cohort", "sid"]).agg(mouth=("mouth_tremor", "mean"), voice=("voice_loudness", "mean"),
                                              label=("label", "first")).reset_index()
        fig, axs = plt.subplots(1, s.cohort.nunique(), figsize=(9, 3.4))
        for ax, (coh, g_) in zip(np.atleast_1d(axs), s.groupby("cohort")):
            for lb, col in ((0, C["MDVR"]), (1, C["FIGSHARE"])):
                h = g_[g_.label == lb]
                ax.scatter(h.voice, h.mouth, color=col, s=16, label="PD" if lb else "healthy")
            ax.set_xlabel("voice-box loudness tremor (log)")
            ax.set_ylabel("mouth tremor share 3-7 Hz")
            ax.set_title(coh, loc="left")
        np.atleast_1d(axs)[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "fig49_tremor_map.png", dpi=150)
        plt.close(fig)


def main20():
    buf = io.StringIO()
    with redirect_stdout(Tee(sys.stdout, buf)):
        step20()
    (OUT / "report.txt").write_text(buf.getvalue())


if __name__ == "__main__":
    main20()
