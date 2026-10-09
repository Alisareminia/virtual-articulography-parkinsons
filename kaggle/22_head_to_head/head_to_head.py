"""PD voice project - step 22: head-to-head - does deep-learning articulography add anything beyond classic acoustics?

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


def progress(msg):
    with open(OUT / "progress.txt", "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        fh.flush()
        os.fsync(fh.fileno())


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
# STEP 22 - Head-to-head: does deep-learning articulography (step 20) add anything beyond CLASSIC acoustics?
# (helpers above: steps 03-20 verbatim; deep-learning mouth measures from step 20; audio from step 02, sex from step 03)
#
# Classic competitors computed on the SAME reading recordings (MDVR English, IPVS Italian):
#   timing/prosody : speech rate, pause time, pause length, pitch variability, loudness variability
#   voice quality  : HNR, CPPS, H1-H2, alpha ratio, Hammarberg
#   articulation   : consonant sharpness (centroid, 4-8 kHz), high-frequency energy, and a CLASSIC formant-based
#                    stand-in for mouth movement: F1 range / speed (jaw opening), F2 range / speed (tongue front-back),
#                    F1-F2 working space (formant "vowel space" of running speech)
# Deep learning (pre-specified from the step-21 sensor check): validated composite = tongue-tip range, tongue working
#   space, lower-lip range, lower-lip speed (higher = more hypokinetic); plus the full set of 15 measures.
# Tests: (1) cross-validated PD-vs-healthy AUC of each feature set and of classic + deep learning; (2) nested logistic
#   likelihood-ratio test of the composite over the best classic predictors; (3) severity (MDVR): composite vs classic,
#   partial correlations and bootstrap difference; (4) how much of the composite classic features can predict
#   (unique information); (5) does the formant-based stand-in reproduce the articulator-specific findings?
# =====================================================================================================================
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

N_JOBS = int(os.environ.get("PDV_JOBS", "4"))
N_REP = 50
DL_VALIDATED = ["TT_range", "TB_space", "LL_range", "LL_speed"]          # passed the step-21 sensor check
DL_ALL = ["UL_range", "UL_speed", "LL_range", "LL_speed", "LI_range", "LI_speed", "TT_range", "TT_speed", "TB_range",
          "TB_speed", "TD_range", "TD_speed", "LA_range", "LA_speed", "TB_space"]
SETS = {
    "classic timing/prosody": ["speech_rate_syl_s", "pause_ratio", "pause_mean_s", "f0_sd_st", "intensity_sd_db"],
    "classic voice quality": ["hnr_db", "cpps_db", "h1h2_db", "alpha_ratio_db", "hammarberg_db"],
    "classic articulation (incl. formant movement)": ["cons_centroid_hz", "cons_hf_db", "hf_ratio_db", "F1_range",
                                                      "F1_speed", "F2_range", "F2_speed", "F1F2_space"],
}
SETS["all classic"] = sum(SETS.values(), [])
SETS["deep learning (validated composite)"] = ["dl_composite"]
SETS["deep learning (all 15)"] = DL_ALL
SETS["all classic + DL composite"] = SETS["all classic"] + ["dl_composite"]
SETS["all classic + DL all"] = SETS["all classic"] + DL_ALL
FORMANT_PAIRS = [("F1_range", "LI_range", "jaw opening"), ("F1_speed", "LI_speed", "jaw speed"),
                 ("F2_range", "TB_range", "tongue front-back"), ("F2_speed", "TB_speed", "tongue speed"),
                 ("F1F2_space", "TB_space", "tongue working space")]


def formant_kinematics(x, male):
    """Classic formant-based stand-in for mouth movement in running speech (voiced frames only)."""
    snd = parselmouth.Sound(np.asarray(x, np.float64), sampling_frequency=SR16)
    lo, hi = f0_range_for(snd)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5000 if male else 5500)
    t = np.array(fm.xs())
    f1 = np.array([fm.get_value_at_time(1, ti) for ti in t], float)
    f2 = np.array([fm.get_value_at_time(2, ti) for ti in t], float)
    pt, pf = pitch.xs(), pitch.selected_array["frequency"]
    voiced = pf[np.clip(np.searchsorted(pt, t), 0, len(pt) - 1)] > 0
    ok = voiced & np.isfinite(f1) & np.isfinite(f2) & (f1 > 150) & (f2 > 500)
    if ok.sum() < 50:
        return {}
    out = {}
    for nm, f in (("F1", f1), ("F2", f2)):
        v = f[ok]
        out[f"{nm}_range"] = float(np.percentile(v, 95) - np.percentile(v, 5))
        d = np.abs(np.diff(f)) / 0.01
        dd = d[ok[1:] & ok[:-1]]
        out[f"{nm}_speed"] = float(np.percentile(dd, 95)) if len(dd) > 20 else np.nan
    out["F1F2_space"] = float(np.sqrt(max(np.linalg.det(np.cov(f1[ok], f2[ok])), 0)))
    return out


def classic_job(args):
    path, male = args
    try:
        x, sr = sf.read(path, dtype="float64")
        x = x.mean(axis=1) if x.ndim > 1 else x
        x = trim_and_normalise(x if sr == SR16 else resample(x, sr, SR16), SR16)
        f = all_features(x)
        f.update(formant_kinematics(x, male))
        return f
    except Exception as ex:  # noqa: BLE001
        return {"error": str(ex)[:150]}


def cv_auc(X, y, reps=N_REP):
    aucs = []
    for r in range(reps):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
    return np.array(aucs)


def step22():
    OUT.mkdir(parents=True, exist_ok=True)
    s02 = find_root_with("manifest.csv", "clean_audio")
    s20 = find_root_with("reading_recordings.csv")
    man = pd.read_csv(s02 / "manifest.csv")
    DL = pd.read_csv(s20 / "reading_recordings.csv").rename(columns={"take": "rec_take"})
    reading = man[man.task_family.eq("read_passage") & man.dataset.isin(["MDVR", "IPVS"])]
    reading = reading[reading.subject_id.isin(DL.sid)]
    sexmap = DL.drop_duplicates("sid").set_index("sid").male
    jobs = [(str(s02 / "clean_audio" / r.clean_path), bool(sexmap.get(r.subject_id, 0) == 1)) for r in reading.itertuples()]
    progress(f"classic features for {len(jobs)} recordings")
    try:
        with mp.get_context("spawn").Pool(N_JOBS) as pool:
            res = pool.map(classic_job, jobs, chunksize=2)
    except Exception as ex:  # noqa: BLE001
        print(f"  parallel unavailable ({ex}); serial")
        res = [classic_job(j) for j in jobs]
    CL = []
    for r, f in zip(reading.itertuples(), res):
        if "error" in f:
            print(f"    classic failed {r.subject_id} {r.task}: {f['error']}")
            continue
        CL.append(dict(sid=r.subject_id, rec_take=r.task, **f))
    CL = pd.DataFrame(CL)
    CL.to_csv(OUT / "classic_features_recordings.csv", index=False)
    progress("classic features done")

    M = DL.merge(CL, on=["sid", "rec_take"], how="inner")
    num = [c for c in M.columns if c not in ("sid", "rec_take", "sex", "cohort") and pd.api.types.is_numeric_dtype(M[c])]
    S = M.groupby(["cohort", "sid"])[num].mean().reset_index()
    for c in set(SETS["all classic"]) & set(LOG_FEATS):
        if c in S:
            S[c] = np.log(S[c].clip(lower=1e-6))
    # pre-specified composite: higher = smaller / slower movements; z within cohort and sex (healthy reference)
    S["dl_composite"] = np.nan
    for coh, g in S.groupby("cohort"):
        Z = []
        for c in DL_VALIDATED:
            z = pd.Series(np.nan, index=g.index)
            for mv in (0.0, 1.0):
                ref = g[(g.label == 0) & (g.male == mv)][c]
                if ref.notna().sum() < 5:
                    ref = g[g.label == 0][c]
                msk = g.male == mv
                z[msk] = (g.loc[msk, c] - ref.mean()) / (ref.std() or 1)
            Z.append(-z)
        S.loc[g.index, "dl_composite"] = pd.concat(Z, axis=1).mean(axis=1)
    S.to_csv(OUT / "speaker_table_dl_and_classic.csv", index=False)
    print(S.groupby(["cohort", "label"]).size().unstack().to_string())

    # ------------------------------------------------------------------ 1. discrimination
    section("1. PD vs HEALTHY: cross-validated AUC (50 x 5-fold; sex included in every model)")
    AU, DIFF = [], []
    for coh, g in S.groupby("cohort"):
        aucs = {}
        for name, feats in SETS.items():
            cols = [c for c in feats if c in g and g[c].notna().mean() > 0.8] + ["male"]
            d = g[cols + ["label"]].copy()
            d[cols] = d[cols].fillna(d[cols].median())
            a = cv_auc(d[cols].to_numpy(), d.label.to_numpy().astype(int))
            aucs[name] = a
            AU.append(dict(cohort=coh, feature_set=name, n_features=len(cols) - 1, auc=a.mean(),
                           auc_lo=np.percentile(a, 2.5), auc_hi=np.percentile(a, 97.5)))
        for base, plus in (("all classic", "all classic + DL composite"), ("all classic", "all classic + DL all"),
                           ("classic articulation (incl. formant movement)", "deep learning (all 15)")):
            dlt = aucs[plus] - aucs[base]
            DIFF.append(dict(cohort=coh, comparison=f"{plus}  vs  {base}", delta_auc=dlt.mean(),
                             share_repeats_better=float((dlt > 0).mean())))
    AU, DIFF = pd.DataFrame(AU), pd.DataFrame(DIFF)
    AU.to_csv(OUT / "h2h_auc.csv", index=False)
    DIFF.to_csv(OUT / "h2h_auc_differences.csv", index=False)
    print(AU.round(3).to_string(index=False))
    print()
    print(DIFF.round(3).to_string(index=False))

    # ------------------------------------------------------------------ 2. nested likelihood-ratio test
    section("2. DOES THE DL COMPOSITE ADD TO THE BEST CLASSIC PREDICTORS? (logistic, likelihood-ratio test)")
    LR = []
    for coh, g in S.groupby("cohort"):
        cand = [c for c in SETS["all classic"] if c in g and g[c].notna().mean() > 0.8]
        aucs1 = {c: abs(roc_auc_score(g.label, g[c].fillna(g[c].median())) - 0.5) for c in cand}
        best = sorted(aucs1, key=aucs1.get, reverse=True)[:3]
        d = g[["label", "male", "dl_composite"] + best].dropna()
        Xb = sm.add_constant((d[["male"] + best] - d[["male"] + best].mean()) / d[["male"] + best].std().replace(0, 1))
        Xf = Xb.assign(dl_composite=(d.dl_composite - d.dl_composite.mean()) / (d.dl_composite.std() or 1))
        try:
            f0 = sm.Logit(d.label, Xb).fit(disp=0, method="lbfgs", maxiter=500)
            f1 = sm.Logit(d.label, Xf).fit(disp=0, method="lbfgs", maxiter=500)
            chi = 2 * (f1.llf - f0.llf)
            LR.append(dict(cohort=coh, best_classic=", ".join(best), n=len(d), lr_chi2=chi, p=float(chi2_sf(chi, 1)),
                           or_per_sd=float(np.exp(f1.params["dl_composite"]))))
        except Exception as ex:  # noqa: BLE001
            LR.append(dict(cohort=coh, best_classic=", ".join(best), n=len(d), error=str(ex)[:80]))
    LR = pd.DataFrame(LR)
    LR.to_csv(OUT / "h2h_likelihood_ratio.csv", index=False)
    print(LR.round(4).to_string(index=False))

    # ------------------------------------------------------------------ 3. severity
    section("3. SEVERITY (MDVR patients): deep learning vs classic, and DL controlling for the best classic measure")
    g = S[(S.cohort == "MDVR reading") & (S.label == 1)].copy()
    SV = []
    rng = np.random.default_rng(22)
    comp = ["dl_composite", "TB_space", "TT_range", "speech_rate_syl_s", "pause_ratio", "f0_sd_st", "F1F2_space",
            "F2_speed", "cons_centroid_hz"]
    for sc in ("hy", "updrs3", "updrs2"):
        for c in comp:
            d = g[[c, sc]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(d) >= 8 and d[c].std() > 0:
                r_, p_ = spearmanr(d[c], d[sc])
                SV.append(dict(scale=sc, measure=c, kind="deep learning" if c in ("dl_composite", "TB_space", "TT_range")
                               else "classic", n=len(d), rho=r_, p=p_))
        for ctrl in ("speech_rate_syl_s", "F1F2_space"):
            d = g[["dl_composite", sc, ctrl]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(d) >= 8:
                ra, pa = partial_spearman(d.dl_composite, d[sc], d[ctrl])
                SV.append(dict(scale=sc, measure=f"dl_composite | controlling {ctrl}", kind="partial", n=len(d), rho=ra, p=pa))
        d = g[["dl_composite", "speech_rate_syl_s", sc]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) >= 8:
            diffs = []
            for _ in range(2000):
                i = rng.integers(0, len(d), len(d))
                b = d.iloc[i]
                if b[sc].nunique() < 2:
                    continue
                diffs.append(abs(spearmanr(b.dl_composite, b[sc])[0]) - abs(spearmanr(b.speech_rate_syl_s, b[sc])[0]))
            diffs = np.array([v for v in diffs if np.isfinite(v)])
            SV.append(dict(scale=sc, measure="|rho DL composite| - |rho speech rate|", kind="bootstrap difference",
                           n=len(d), rho=float(np.mean(diffs)), p=np.nan, ci_lo=np.percentile(diffs, 2.5),
                           ci_hi=np.percentile(diffs, 97.5)))
    SV = pd.DataFrame(SV)
    SV.to_csv(OUT / "h2h_severity.csv", index=False)
    print(SV.round(3).to_string(index=False))

    # ------------------------------------------------------------------ 4. unique information
    section("4. UNIQUE INFORMATION: how much of the DL composite can ALL classic features predict? (cross-validated R2)")
    UQ = []
    for coh, g in S.groupby("cohort"):
        cols = [c for c in SETS["all classic"] if c in g and g[c].notna().mean() > 0.8]
        d = g[cols + ["dl_composite", "male"]].dropna(subset=["dl_composite"])
        X = d[cols + ["male"]].fillna(d[cols + ["male"]].median()).to_numpy()
        y = d.dl_composite.to_numpy()
        pred = cross_val_predict(make_pipeline(StandardScaler(), sm_ridge()), X, y, cv=KFold(5, shuffle=True, random_state=0))
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        UQ.append(dict(cohort=coh, n=len(d), cv_r2_classic_predicts_dl=float(r2), unique_share=float(1 - max(r2, 0))))
    UQ = pd.DataFrame(UQ)
    UQ.to_csv(OUT / "h2h_unique_information.csv", index=False)
    print(UQ.round(3).to_string(index=False))

    # ------------------------------------------------------------------ 5. formant stand-in vs DL articulator measures
    section("5. CAN A CLASSIC FORMANT-BASED STAND-IN REPRODUCE THE ARTICULATOR FINDINGS? (PD - healthy, sex/age adjusted)")
    FP = []
    for coh, g in S.groupby("cohort"):
        for fc, dc, lab_ in FORMANT_PAIRS:
            ef, ed = adj_effect(g, fc), adj_effect(g, dc)
            rr = spearmanr(g[fc], g[dc], nan_policy="omit")[0] if fc in g and dc in g else np.nan
            FP.append(dict(cohort=coh, movement=lab_, classic_measure=fc, classic_beta=ef["beta"] if ef else np.nan,
                           classic_p=ef["p"] if ef else np.nan, dl_measure=dc, dl_beta=ed["beta"] if ed else np.nan,
                           dl_p=ed["p"] if ed else np.nan, rho_classic_vs_dl=rr))
    FP = pd.DataFrame(FP)
    FP.to_csv(OUT / "h2h_formant_vs_dl.csv", index=False)
    print(FP.round(3).to_string(index=False))

    # ------------------------------------------------------------------ verdict, report, figures
    section("VERDICT")
    V = []
    for coh in AU.cohort.unique():
        a = AU[AU.cohort == coh].set_index("feature_set").auc
        d1 = DIFF[(DIFF.cohort == coh) & DIFF.comparison.str.startswith("all classic + DL composite")]
        lr = LR[LR.cohort == coh]
        V.append(dict(cohort=coh, auc_all_classic=a.get("all classic"), auc_dl_composite=a.get("deep learning (validated composite)"),
                      auc_classic_plus_dl=a.get("all classic + DL composite"),
                      delta=float(d1.delta_auc.iloc[0]) if len(d1) else np.nan,
                      lr_p=float(lr.p.iloc[0]) if len(lr) and "p" in lr else np.nan,
                      unique_share=float(UQ[UQ.cohort == coh].unique_share.iloc[0]) if len(UQ[UQ.cohort == coh]) else np.nan))
    VD = pd.DataFrame(V)
    VD.to_csv(OUT / "h2h_verdict.csv", index=False)
    print(VD.round(3).to_string(index=False))
    L = ["# Head-to-head: deep-learning articulography vs classic acoustics", "", "## Verdict", "", md_table(VD.round(3)), "",
         "## Cross-validated AUC", "", md_table(AU.round(3)), "", md_table(DIFF.round(3)), "",
         "## Likelihood-ratio test", "", md_table(LR.round(4)), "", "## Severity (MDVR)", "", md_table(SV.round(3)), "",
         "## Unique information", "", md_table(UQ.round(3)), "", "## Formant stand-in vs DL", "", md_table(FP.round(3))]
    (OUT / "report_head_to_head.md").write_text("\n".join(L))
    if len(AU):
        fig, axs = plt.subplots(1, AU.cohort.nunique(), figsize=(11, 3.8), sharey=True)
        for ax, (coh, g) in zip(np.atleast_1d(axs), AU.groupby("cohort")):
            g = g.reset_index(drop=True)
            y = np.arange(len(g))[::-1]
            cols = [C["FIGSHARE"] if "deep" in n and "classic" not in n else C["MDVR"] if "+" in n else C["IPVS"]
                    for n in g.feature_set]
            ax.barh(y, g.auc - 0.5, left=0.5, color=cols)
            ax.errorbar(g.auc, y, xerr=[g.auc - g.auc_lo, g.auc_hi - g.auc], fmt="none", color=C["ink2"], lw=0.8)
            ax.axvline(0.5, color=C["muted"], lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(g.feature_set, fontsize=7)
            ax.set_xlabel("cross-validated AUC")
            ax.set_title(coh, loc="left")
        fig.suptitle("Does deep-learning articulography add to classic acoustics?", x=0.02, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / "fig52_head_to_head_auc.png", dpi=150)
        plt.close(fig)
    s = SV[SV.kind.isin(["deep learning", "classic"])] if len(SV) else SV
    if len(s):
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
        for ax, sc in zip(axs, ("hy", "updrs3", "updrs2")):
            g = s[s.scale == sc].sort_values("rho")
            ax.barh(g.measure, g.rho.abs(), color=[C["FIGSHARE"] if k == "deep learning" else C["IPVS"] for k in g.kind])
            ax.set_title({"hy": "Hoehn & Yahr", "updrs3": "UPDRS III-18", "updrs2": "UPDRS II speech"}[sc], loc="left")
            ax.set_xlabel("|Spearman rho|")
            ax.tick_params(axis="y", labelsize=7)
        fig.suptitle("Severity (MDVR patients): deep learning (orange) vs classic (blue)", x=0.02, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / "fig53_head_to_head_severity.png", dpi=150)
        plt.close(fig)
    print("\nsaved: h2h_*.csv, speaker_table_dl_and_classic.csv, classic_features_recordings.csv, report_head_to_head.md, fig52-53")


def chi2_sf(x, df):
    from scipy.stats import chi2
    return chi2.sf(x, df)


def sm_ridge():
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0)


def main22():
    buf = io.StringIO()
    with redirect_stdout(Tee(sys.stdout, buf)):
        step22()
    (OUT / "report.txt").write_text(buf.getvalue())


if __name__ == "__main__":
    main22()
