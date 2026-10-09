"""PD voice project - step 24: signal-level figure assets for the article (example traces, spectrograms, movement clouds, tremor spectra)

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
MOCHA_URL = "http://data.cstr.ed.ac.uk/mocha/"
MAP = {"TD": "td", "TB": "tb", "TT": "tt", "LI": "li", "UL": "ul", "LL": "ll"}


def read_est(path):
    """Edinburgh Speech Tools track file -> (channel names, frames x channels) ; binary or ascii."""
    raw = Path(path).read_bytes()
    end = raw.index(b"EST_Header_End") + len(b"EST_Header_End\n")
    head = raw[:end].decode("latin-1").splitlines()
    info, names = {}, {}
    for ln in head:
        p = ln.split()
        if len(p) >= 2:
            info[p[0]] = p[1]
            if p[0].startswith("Channel_"):
                names[int(p[0].split("_")[1])] = p[1]
    nch = int(info.get("NumChannels", len(names)))
    nfr = int(info.get("NumFrames", 0))
    if info.get("DataType", "binary").lower().startswith("ascii"):
        arr = np.loadtxt(raw[end:].decode("latin-1").splitlines())
    else:
        dt = "<f4" if info.get("ByteOrder", "01") == "01" else ">f4"
        arr = np.frombuffer(raw[end:], dtype=dt)
        per = 2 + nch
        arr = arr[: (len(arr) // per) * per].reshape(-1, per)
    data = arr[:, -nch:] if arr.shape[1] >= nch else arr
    if nfr and len(data) > nfr:
        data = data[:nfr]
    return [names.get(i, f"ch{i}") for i in range(nch)], data.astype(np.float64)


def read_nist(path):
    raw = Path(path).read_bytes()
    if raw[:4] == b"RIFF":
        x, sr = sf.read(path, dtype="float64")
        return x, sr
    head = raw[:1024].decode("latin-1")
    hsize = int(head.split("\n")[1].strip()) if head.startswith("NIST_1A") else 1024
    sr, order = 16000, "01"
    for ln in head.splitlines():
        p = ln.split()
        if p and p[0] == "sample_rate":
            sr = int(p[-1])
        if p and p[0] == "sample_byte_format":
            order = p[-1]
    x = np.frombuffer(raw[hsize:], dtype="<i2" if order == "01" else ">i2").astype(np.float64) / 32768.0
    return x, sr
MOCHA_COILS = ["ui", "li", "ul", "ll", "tt", "tb", "td", "v", "bn", "c10"]
MOCHA_DEFAULT = ([f"{c}_x" for c in MOCHA_COILS[:5]] + [f"{c}_y" for c in MOCHA_COILS[:5]]
                 + [f"{c}_x" for c in MOCHA_COILS[5:]] + [f"{c}_y" for c in MOCHA_COILS[5:]])


def real_ema_12(names, data, n50):
    """MOCHA channels -> SPARC channel order (TDX..LLY) at 50 Hz (block mean of 10 samples at 500 Hz)."""
    if not any("_x" in n.lower() for n in names):
        names = MOCHA_DEFAULT[:len(names)]
    idx = {n.lower(): i for i, n in enumerate(names)}
    cols = []
    for c in CH:
        art, ax = c[:2], c[2].lower()
        key = f"{MAP[art]}_{ax}"
        if key not in idx:
            return None
        cols.append(data[:, idx[key]])
    X = np.column_stack(cols)
    for j in range(X.shape[1]):                     # interpolate missing samples
        v = X[:, j]
        bad = ~np.isfinite(v)
        if bad.mean() > 0.2:
            return None
        if bad.any():
            X[bad, j] = np.interp(np.where(bad)[0], np.where(~bad)[0], v[~bad])
    m = len(X) // 10
    X50 = X[: m * 10].reshape(m, 10, 12).mean(axis=1)
    return X50[:n50] if len(X50) >= n50 else np.vstack([X50, np.repeat(X50[-1:], n50 - len(X50), axis=0)])


# =====================================================================================================================
# STEP 24 - Figure assets for the article (signal-level material the summary CSVs do not contain)
# (helpers above: steps 03-21 verbatim; audio from step 02, sex from step 03, speaker choices from steps 20 and 22)
#
# Saves small .npz files (a few MB in total) that the presentation notebook draws from:
#   example_reading.npz   typical healthy vs typical Parkinson's reader (MDVR, same sex): 6-s audio, spectrogram,
#                         pitch, formants F1-F3, and the 12 estimated articulator traces (50 Hz)
#   mocha_example.npz     one MOCHA-TIMIT sentence: spectrogram + REAL sensor traces + deep-learning estimates
#   movement_clouds.npz   where each articulator travels during reading (speaker-centred 2-D histograms per group and
#                         cohort) + per-speaker movement ellipses (covariance) with severity scores
#   tremor_spectra.npz    movement spectra of jaw / tongue (sustained /a/, IPVS + Figshare) and loudness-envelope
#                         spectra, per speaker; plus a typical patient / healthy example trace and its time-frequency map
# Model coordinates are per-channel normalised (not millimetres): +y = up, -x = front (checked on the vowels).
# =====================================================================================================================
import json  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402

from scipy.signal import stft, welch  # noqa: E402

GRID = np.linspace(-3, 3, 61)                      # movement-cloud histogram edges (model units, speaker-centred)
FREQ = np.linspace(0.5, 20, 79)                    # common frequency grid for spectra (Hz)
CLOUD_ART = ["TT", "TB", "TD", "LI", "LL", "UL"]
EXAMPLE_S = 6.0
MAX_FRAMES_PER_SPEAKER = 3000


def spectrogram_db(x):
    S = np.abs(librosa.stft(np.asarray(x, np.float32), n_fft=512, hop_length=160, win_length=400)) ** 2
    return (10 * np.log10(S + 1e-10)).astype(np.float16)


def formant_tracks(x, male):
    snd = parselmouth.Sound(np.asarray(x, np.float64), sampling_frequency=SR16)
    lo, hi = f0_range_for(snd)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5000 if male else 5500)
    t = np.array(fm.xs())
    pt, pf = pitch.xs(), pitch.selected_array["frequency"]
    voiced = pf[np.clip(np.searchsorted(pt, t), 0, len(pt) - 1)] > 0
    F = np.array([[fm.get_value_at_time(k, ti) for ti in t] for k in (1, 2, 3)], float)
    F[:, ~voiced] = np.nan
    f0 = np.interp(t, pt, np.where(pf > 0, pf, np.nan))
    return t.astype(np.float32), F.astype(np.float32), f0.astype(np.float32)


def best_window(x, seconds):
    """Start (samples) of the window with the most real voiced speech; penalises stationary tones (phone beeps)."""
    snd = parselmouth.Sound(np.asarray(x, np.float64), sampling_frequency=SR16)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=400)
    f = p.selected_array["frequency"]
    voiced = (f >= 75) & (f <= 400)
    S = np.abs(librosa.stft(np.asarray(x, np.float32), n_fft=512, hop_length=160)) ** 2
    peaky = (S.max(axis=0) / (S.sum(axis=0) + 1e-12)) > 0.35          # one dominant bin = tone, not speech
    n = min(len(voiced), len(peaky))
    score = voiced[:n].astype(float) - 2.0 * peaky[:n].astype(float)
    w = int(seconds * 100)
    if n <= w:
        return 0
    c = np.convolve(score, np.ones(w), mode="valid")
    start = int(np.argmax(c[int(0.05 * len(c)):]) + int(0.05 * len(c)))
    return start * 160


def abs_psd_db(sig, fs):
    sig = np.asarray(sig, float)
    if len(sig) < 1.5 * fs:
        return None
    n = np.arange(len(sig))
    sig = sig - np.polyval(np.polyfit(n, sig, 2), n)
    f, p = welch(sig, fs=fs, nperseg=min(len(sig), int(4 * fs)))
    return (10 * np.log10(np.interp(FREQ, f, p) + 1e-12)).astype(np.float32)


def rel_psd(sig, fs):
    sig = np.asarray(sig, float)
    if len(sig) < 1.5 * fs:
        return None
    n = np.arange(len(sig))
    sig = sig - np.polyval(np.polyfit(n, sig, 2), n)
    f, p = welch(sig, fs=fs, nperseg=min(len(sig), int(4 * fs)))
    g = np.interp(FREQ, f, p)
    ref = (FREQ >= 1) & (FREQ <= 15)
    return (g / (g[ref].sum() + 1e-12)).astype(np.float32)


def loudness_envelope(x):
    e, _ = frames_db(x, SR16, 0.025, 0.01)
    act = e > np.percentile(e, 95) - 30
    idx = np.where(act)[0]
    if len(idx) < 100:
        return None
    a, b = idx[0], idx[-1]
    k = int(0.1 * (b - a))
    return e[a + k:b - k]


def step24():
    OUT.mkdir(parents=True, exist_ok=True)
    s02 = find_root_with("manifest.csv", "clean_audio")
    s03 = find_root_with("features_subject.csv")
    s20 = find_root_with("tremor_speakers_IPVS.csv")
    s22 = find_root_with("speaker_table_dl_and_classic.csv")
    man = pd.read_csv(s02 / "manifest.csv")
    S3 = pd.read_csv(s03 / "features_subject.csv")
    sex_map = S3.drop_duplicates("subject_id").set_index("subject_id").sex
    man["sex"] = man.subject_id.map(sex_map).fillna(man.get("sex"))
    man = man[man.group.isin(["PD", "HC", "EHC"]) & man.sex.isin(["F", "M"])]
    SPK = pd.read_csv(s22 / "speaker_table_dl_and_classic.csv")
    inv = Inverter("multi")
    meta = {"coords": "per-channel normalised model units; +y up, -x front", "channels": CH, "ema_fs": FS_EMA}

    def load(p):
        x, sr = sf.read(str(s02 / "clean_audio" / p), dtype="float64")
        x = x.mean(axis=1) if x.ndim > 1 else x
        return trim_and_normalise(x if sr == SR16 else resample(x, sr, SR16), SR16)

    # ------------------------------------------------------------------ 1. example readers
    section("1. EXAMPLE READERS (MDVR, typical healthy vs typical Parkinson's, same sex)")
    m = SPK[SPK.cohort == "MDVR reading"]
    sx = 0.0 if ((m.male == 0) & (m.label == 1)).sum() >= 3 else 1.0
    ex = {}
    for lb, tag in ((0, "healthy"), (1, "pd")):
        g = m[(m.label == lb) & (m.male == sx)].dropna(subset=["TB_space"])
        r = g.iloc[(g.TB_space - g.TB_space.median()).abs().argmin()]
        row = man[(man.subject_id == r.sid) & (man.task_family == "read_passage")].iloc[0]
        x = load(row.clean_path)
        s = best_window(x, EXAMPLE_S)
        x = trim_and_normalise(x[s:s + int(EXAMPLE_S * SR16)], SR16)
        E = inv.ema(x)
        t, F, f0 = formant_tracks(x, sx == 1.0)
        ex.update({f"{tag}_audio": x.astype(np.float32), f"{tag}_spec": spectrogram_db(x), f"{tag}_ema": E.astype(np.float32),
                   f"{tag}_ftime": t, f"{tag}_formants": F, f"{tag}_f0": f0,
                   f"{tag}_info": json.dumps(dict(sid=r.sid, tb_space=float(r.TB_space), hy=r.hy, updrs3=r.updrs3,
                                                  updrs2=r.updrs2, male=bool(sx)))})
        print(f"  {tag}: {r.sid} TB_space {r.TB_space:.3f} (group median {g.TB_space.median():.3f})")
    np.savez_compressed(OUT / "example_reading.npz", **ex, meta=json.dumps(meta))
    progress("examples done")

    # ------------------------------------------------------------------ 2. MOCHA example sentence
    section("2. MOCHA-TIMIT EXAMPLE SENTENCE (real sensors vs estimate)")
    try:
        if KNN_BACKEND == "stub":
            raise RuntimeError("stub run: no download")
        root = TMPW / "mocha"
        root.mkdir(parents=True, exist_ok=True)
        tgz = root / "fsew0_v1.1.tar.gz"
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(MOCHA_URL + "fsew0_v1.1.tar.gz", tgz)
                break
            except Exception as ex_:  # noqa: BLE001
                print(f"  download attempt {attempt + 1} failed: {ex_}")
                time.sleep(10)
        with tarfile.open(tgz) as tf:
            mem = [mm for mm in tf.getmembers() if mm.name.endswith((".wav", ".ema")) and ".." not in mm.name]
            tf.extractall(root, members=mem)
        best = []
        for e in sorted(root.rglob("*.ema"))[:120]:
            w = e.with_suffix(".wav")
            if not w.exists():
                continue
            x, sr = read_nist(w)
            x = x if sr == SR16 else resample(x, sr, SR16)
            if not (2.5 <= len(x) / SR16 <= 4.5):
                continue
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
            est = inv.ema(x)
            names, data = read_est(e)
            real = real_ema_12(names, data, len(est))
            if real is None:
                continue
            rs = [np.corrcoef(est[:, CHI[c]], real[:, CHI[c]])[0, 1] for c in ("TTY", "LIY", "TBY", "LLY")]
            best.append((float(np.nanmean(rs)), e.stem, x, est, real))
        best.sort(key=lambda b: b[0])
        r_med, name, x, est, real = best[len(best) // 2]                  # the TYPICAL sentence, not the best one
        z = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-9)  # noqa: E731
        ez, rz = z(est), z(real)
        sign = np.sign([np.corrcoef(ez[:, j], rz[:, j])[0, 1] for j in range(12)])
        np.savez_compressed(OUT / "mocha_example.npz", audio=x.astype(np.float32), spec=spectrogram_db(x),
                            est=(ez * sign).astype(np.float32), real=rz.astype(np.float32),
                            r=np.array([np.corrcoef(ez[:, j], rz[:, j])[0, 1] for j in range(12)], np.float32),
                            info=json.dumps(dict(sentence=name, mean_r=r_med, n_candidates=len(best))), meta=json.dumps(meta))
        print(f"  sentence {name}: mean r {r_med:.2f} (median of {len(best)} candidates)")
    except Exception as ex_:  # noqa: BLE001
        print(f"  MOCHA example skipped: {ex_}")
    progress("MOCHA example done")

    # ------------------------------------------------------------------ 3. movement clouds + ellipses (reading)
    section("3. MOVEMENT CLOUDS (where each articulator travels during reading)")
    reading = man[man.task_family.eq("read_passage") & man.dataset.isin(["MDVR", "IPVS"])]
    reading = reading[(reading.dataset == "MDVR") | (reading.task == "B1")]
    H = {}
    ELL = []
    rng = np.random.default_rng(24)
    for r in reading.itertuples():
        try:
            x = load(r.clean_path)
            n = int(20 * SR16)
            E = np.concatenate([inv.ema(x[s:s + n]) for s in range(0, len(x), n) if len(x[s:s + n]) > SR16 // 2])
            e, _ = frames_db(x, SR16, 0.04, 0.02)
            act = activity(e)
            act = np.concatenate([act, np.zeros(max(0, len(E) - len(act)), bool)])[:len(E)]
            Ea = E[act] if act.sum() > 100 else E
            if len(Ea) > MAX_FRAMES_PER_SPEAKER:
                Ea = Ea[np.sort(rng.choice(len(Ea), MAX_FRAMES_PER_SPEAKER, replace=False))]
            row = dict(cohort=r.dataset, sid=r.subject_id, label=int(r.label), male=float(r.sex == "M"),
                       hy=r.hoehn_yahr, updrs3=r.updrs_iii18, updrs2=r.updrs_ii5)
            for a in CLOUD_ART:
                xy = Ea[:, [CHI[f"{a}X"], CHI[f"{a}Y"]]]
                d = xy - xy.mean(0)
                h, _, _ = np.histogram2d(d[:, 0], d[:, 1], bins=[GRID, GRID])
                key = f"{r.dataset}_{int(r.label)}_{a}"
                H[key] = H.get(key, 0) + h / max(h.sum(), 1)          # each speaker weighs the same
                cv = np.cov(d.T)
                row.update({f"{a}_cxx": cv[0, 0], f"{a}_cxy": cv[0, 1], f"{a}_cyy": cv[1, 1],
                            f"{a}_space": float(np.sqrt(max(np.linalg.det(cv), 0)))})
            ELL.append(row)
        except Exception as ex_:  # noqa: BLE001
            print(f"    reading failed {r.subject_id}: {ex_}")
    counts = pd.DataFrame(ELL).groupby(["cohort", "label"]).size().to_dict()
    np.savez_compressed(OUT / "movement_clouds.npz", grid=GRID, **{k: v.astype(np.float32) for k, v in H.items()},
                        counts=json.dumps({f"{a}_{b}": int(c) for (a, b), c in counts.items()}), meta=json.dumps(meta))
    pd.DataFrame(ELL).to_csv(OUT / "movement_ellipses.csv", index=False)
    print(f"  speakers per cohort/group: {counts}")
    progress("clouds done")

    # ------------------------------------------------------------------ 4. tremor spectra
    section("4. TREMOR SPECTRA (sustained /a/): jaw / tongue movement and loudness envelope")
    TI = pd.read_csv(s20 / "tremor_speakers_IPVS.csv")
    TF = pd.read_csv(s20 / "tremor_speakers_FIGSHARE.csv")
    vow = man[((man.dataset == "IPVS") & man.task.isin(["VA1", "VA2"])) |
              ((man.dataset == "FIGSHARE") & (man.task_family == "vowel_a"))]
    keep = set(TI.sid) | set(TF.sid)
    vow = vow[vow.subject_id.isin(keep)]
    rows, psd_j, psd_t, psd_l = [], [], [], []
    abs_j, abs_t, abs_l = [], [], []
    examples = {}
    pick = {}
    for T, ds, col in ((TI, "IPVS", "mouth_LI"), (TF, "FIGSHARE", "voice_loudness")):
        for lb in (0, 1):
            g = T[T.label == lb].dropna(subset=[col])
            pick[(ds, lb)] = g.iloc[(g[col] - g[col].median()).abs().argmin()].sid
    for r in vow.itertuples():
        try:
            x = load(r.clean_path)
            E = inv.ema(x)
            C_ = E[int(0.1 * len(E)): len(E) - int(0.1 * len(E))]
            pj, pt_ = rel_psd(C_[:, CHI["LIY"]], FS_EMA), rel_psd(C_[:, CHI["TBY"]], FS_EMA)
            env = loudness_envelope(x)
            pl = rel_psd(env, 100.0) if env is not None else None
            if pj is None or pl is None:
                continue
            rows.append(dict(cohort=r.dataset, sid=r.subject_id, label=int(r.label), task=r.task))
            psd_j.append(pj)
            psd_t.append(pt_)
            psd_l.append(pl)
            abs_j.append(abs_psd_db(C_[:, CHI["LIY"]], FS_EMA))
            abs_t.append(abs_psd_db(C_[:, CHI["TBY"]], FS_EMA))
            abs_l.append(abs_psd_db(env, 100.0))
            for lb in (0, 1):
                if r.subject_id == pick.get((r.dataset, lb)) and (r.dataset, lb) not in examples:
                    sig = C_[:, CHI["LIY"]] if r.dataset == "IPVS" else env
                    fs = FS_EMA if r.dataset == "IPVS" else 100.0
                    sig = sig - np.polyval(np.polyfit(np.arange(len(sig)), sig, 2), np.arange(len(sig)))
                    f, tt, Z = stft(sig, fs=fs, nperseg=int(1.28 * fs), noverlap=int(1.18 * fs))
                    examples[(r.dataset, lb)] = dict(sig=sig.astype(np.float32), fs=fs, f=f.astype(np.float32),
                                                     t=tt.astype(np.float32), tf=np.abs(Z).astype(np.float32))
        except Exception as ex_:  # noqa: BLE001
            print(f"    vowel failed {r.subject_id}: {ex_}")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "tremor_spectra_index.csv", index=False)
    ex_arrays = {}
    for (ds, lb), d in examples.items():
        for k, v in d.items():
            ex_arrays[f"ex_{ds}_{lb}_{k}"] = np.asarray(v)
    np.savez_compressed(OUT / "tremor_spectra.npz", freq=FREQ.astype(np.float32), jaw=np.array(psd_j),
                        tongue=np.array(psd_t), loudness=np.array(psd_l), jaw_db=np.array(abs_j),
                        tongue_db=np.array(abs_t), loudness_db=np.array(abs_l), **ex_arrays, meta=json.dumps(meta))
    print(f"  spectra: {R.groupby(['cohort', 'label']).size().to_dict()}; examples: {sorted(examples)}")
    progress("tremor done")
    print("\nsaved: example_reading.npz, mocha_example.npz, movement_clouds.npz, movement_ellipses.csv, "
          "tremor_spectra.npz, tremor_spectra_index.csv")


def main24():
    buf = io.StringIO()
    with redirect_stdout(Tee(sys.stdout, buf)):
        step24()
    (OUT / "report.txt").write_text(buf.getvalue())


if __name__ == "__main__":
    main24()
