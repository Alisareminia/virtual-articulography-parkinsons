"""PD voice project - step 19 (pilot): virtual articulography - is deep acoustic-to-articulatory inversion (SPARC) valid on the 3 datasets?

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


# =====================================================================================================================
# STEP 19 (PILOT) - Virtual articulography: does a deep acoustic-to-articulatory inversion model (SPARC, Berkeley) give
# valid tongue / lip / jaw movements on OUR recordings (Italian, Parkinson's patients, 8 kHz phone audio)?
# (helpers above: steps 03-13 verbatim; audio from step 02)
#
# SPARC estimates 12 EMA channels at 50 Hz: tongue dorsum / body / tip, lower incisor (= jaw), upper / lower lip (x, y).
# Pre-specified validity checks (no ground-truth EMA needed):
#   P1 vowel identity: the 5 Italian vowels (a e i o u) must be recognisable from the estimated mouth posture
#      (leave-one-speaker-out, chance = 20%) - in healthy speakers AND in patients.
#   P2 vowel physiology: /a/ = open jaw and lips; /i/, /u/ = high tongue; /u/, /o/ = rounded lips (smaller lip opening).
#   P3 syllable physiology: "pa" repetitions move mainly the LIPS, "ta" repetitions mainly the TONGUE TIP; and the
#      repetition rate seen in the movements equals the acoustic syllable rate.
#   P4 phone audio: Figshare /a/ (8 kHz) still looks like /a/; IPVS vowels degraded to phone band still recognisable.
#   P5 test-retest: vowel postures agree between the two IPVS takes.
# Preview (exploratory, not a result yet): PD vs healthy lip / tongue movement in pa/ta, and 3-7 Hz jaw / tongue tremor
# during sustained /a/.
# =====================================================================================================================
import time  # noqa: E402

from scipy.signal import welch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

KNN_BACKEND = os.environ.get("PDV_KNN_BACKEND", "hub")
CH = ["TDX", "TDY", "TBX", "TBY", "TTX", "TTY", "LIX", "LIY", "ULX", "ULY", "LLX", "LLY"]
CHI = {c: i for i, c in enumerate(CH)}
VOWELS = ["a", "e", "i", "o", "u"]
MODELS = ["multi", "en"]
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


def movement_stats(sig, fs=FS_EMA):
    """Range (p95-p5), peak speed (p95 |velocity|), dominant repetition rate (1.5-10 Hz) and amplitude decrement."""
    sig = np.asarray(sig, float)
    if len(sig) < 25:
        return {}
    v = np.gradient(sig) * fs
    f, p = welch(sig - sig.mean(), fs=fs, nperseg=min(len(sig), 128))
    band = (f >= 1.5) & (f <= 10)
    half = len(sig) // 2
    r1 = np.percentile(sig[:half], 95) - np.percentile(sig[:half], 5)
    r2 = np.percentile(sig[half:], 95) - np.percentile(sig[half:], 5)
    return dict(range=float(np.percentile(sig, 95) - np.percentile(sig, 5)),
                speed=float(np.percentile(np.abs(v), 95)),
                rate_hz=float(f[band][np.argmax(p[band])]) if band.any() else np.nan,
                decrement=float((r2 - r1) / (r1 + 1e-9)))


def tremor_power(sig, fs=FS_EMA):
    """Share of movement power in the 3-7 Hz tremor band (1-15 Hz reference), after removing slow drift."""
    sig = np.asarray(sig, float)
    if len(sig) < 50:
        return np.nan
    sig = sig - np.polyval(np.polyfit(np.arange(len(sig)), sig, 2), np.arange(len(sig)))
    f, p = welch(sig, fs=fs, nperseg=min(len(sig), 100))
    ref = (f >= 1) & (f <= 15)
    return float(p[(f >= 3) & (f <= 7)].sum() / (p[ref].sum() + 1e-12))


def loso_accuracy(X, y, groups):
    pred = np.empty(len(y), dtype=object)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2:
            continue
        m = LinearDiscriminantAnalysis().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def step19():
    OUT.mkdir(parents=True, exist_ok=True)
    s02 = find_root_with("manifest.csv", "clean_audio")
    s03 = find_root_with("features_subject.csv")
    man = pd.read_csv(s02 / "manifest.csv")
    S3 = pd.read_csv(s03 / "features_subject.csv")
    sex_map = S3.drop_duplicates("subject_id").set_index("subject_id").sex
    man["sex"] = man.subject_id.map(sex_map).fillna(man.get("sex"))
    man = man[man.group.isin(["PD", "HC", "EHC"])]

    def load(p):
        x, sr = sf.read(str(s02 / "clean_audio" / p), dtype="float64")
        x = x.mean(axis=1) if x.ndim > 1 else x
        return trim_and_normalise(x if sr == SR16 else resample(x, sr, SR16), SR16)

    vow = man[(man.dataset == "IPVS") & man.task_family.str.startswith("vowel")].copy()
    vow["vowel"] = vow.task.str[1].str.lower()
    vow["take"] = vow.task.str[-1]
    ddk = man[(man.dataset == "IPVS") & (man.task_family == "syllables")].copy()
    fig = man[(man.dataset == "FIGSHARE") & (man.task_family == "vowel_a")].copy()
    amin = pd.to_numeric(fig.loc[fig.label == 1, "age"], errors="coerce").min()
    fig = fig[(fig.label == 1) | (pd.to_numeric(fig.age, errors="coerce") >= amin)]
    print(f"IPVS vowel recordings {len(vow)} ({vow.subject_id.nunique()} speakers); DDK {len(ddk)}; Figshare {len(fig)}")

    REC, VERD = [], []
    for mname in MODELS:
        section(f"MODEL '{mname}'")
        try:
            inv = Inverter(mname)
        except Exception as ex:  # noqa: BLE001
            print(f"  could not load SPARC '{mname}': {ex}")
            continue
        progress(f"model {mname} loaded")
        for r in vow.itertuples():
            try:
                x = load(r.clean_path)
                for cond, xx in (("clean", x), ("phone", phone_band(x))):
                    E = inv.ema(xx)
                    C_ = central(E)
                    REC.append(dict(model=mname, kind="vowel", cond=cond, sid=r.subject_id, label=int(r.label), sex=r.sex,
                                    vowel=r.vowel, take=r.take, n=len(E),
                                    **{c: float(np.mean(C_[:, i])) for i, c in enumerate(CH)},
                                    lip_ap=float(np.mean(lip_aperture(C_))),
                                    trem_LIY=tremor_power(C_[:, CHI["LIY"]]), trem_TBY=tremor_power(C_[:, CHI["TBY"]]),
                                    trem_TDY=tremor_power(C_[:, CHI["TDY"]]), trem_LLY=tremor_power(C_[:, CHI["LLY"]])))
            except Exception as ex:  # noqa: BLE001
                print(f"    vowel failed {r.subject_id} {r.task}: {ex}")
        progress(f"{mname}: IPVS vowels done")
        for r in fig.itertuples():
            try:
                E = inv.ema(load(r.clean_path))
                C_ = central(E)
                REC.append(dict(model=mname, kind="figshare", cond="phone_real", sid=r.subject_id, label=int(r.label),
                                sex=r.sex, vowel="a", take="1", n=len(E),
                                **{c: float(np.mean(C_[:, i])) for i, c in enumerate(CH)},
                                lip_ap=float(np.mean(lip_aperture(C_))),
                                trem_LIY=tremor_power(C_[:, CHI["LIY"]]), trem_TBY=tremor_power(C_[:, CHI["TBY"]]),
                                trem_TDY=tremor_power(C_[:, CHI["TDY"]]), trem_LLY=tremor_power(C_[:, CHI["LLY"]])))
            except Exception as ex:  # noqa: BLE001
                print(f"    figshare failed {r.subject_id}: {ex}")
        progress(f"{mname}: Figshare done")
        for r in ddk.itertuples():
            try:
                x = load(r.clean_path)
                E = inv.ema(x)
                loud = librosa.feature.rms(y=np.asarray(x, np.float32), frame_length=640, hop_length=320)[0]
                loud = np.concatenate([loud, np.repeat(loud[-1:], max(0, len(E) - len(loud)))])[:len(E)]
                act = loud > np.percentile(loud, 30)
                Ea = E[act] if act.sum() >= 25 else E
                la, tt = movement_stats(lip_aperture(Ea)), movement_stats(Ea[:, CHI["TTY"]])
                jw = movement_stats(Ea[:, CHI["LIY"]])
                ac = ddk_features(x, SR16)
                REC.append(dict(model=mname, kind="ddk", cond="clean", sid=r.subject_id, label=int(r.label), sex=r.sex,
                                task=r.task, n=len(E), acoustic_rate=ac.get("ddk_rate_syl_s", np.nan),
                                **{f"lip_{k}": v for k, v in la.items()}, **{f"tt_{k}": v for k, v in tt.items()},
                                **{f"jaw_{k}": v for k, v in jw.items()}))
            except Exception as ex:  # noqa: BLE001
                print(f"    ddk failed {r.subject_id} {r.task}: {ex}")
        progress(f"{mname}: DDK done")
        del inv
    R = pd.DataFrame(REC)
    R.to_csv(OUT / "pilot_ema_recordings.csv", index=False)
    if not len(R):
        print("no SPARC output - pilot cannot be evaluated")
        return

    for mname in R.model.unique():
        section(f"VALIDITY CHECKS - model '{mname}'")
        V = R[(R.model == mname) & (R.kind == "vowel")].copy()
        # speaker-centred posture: each recording minus that speaker's mean posture over all their vowels
        feats = CH + ["lip_ap"]
        for cond in ("clean", "phone"):
            v = V[V.cond == cond].copy()
            v[feats] = v[feats] - v.groupby("sid")[feats].transform("mean")
            V.loc[v.index, [f"{c}_c" for c in feats]] = v[feats].to_numpy()
        cf = [f"{c}_c" for c in CH]

        # P1 vowel identity
        res = {}
        for cond in ("clean", "phone"):
            v = V[V.cond == cond].dropna(subset=cf)
            pred = loso_accuracy(v[cf].to_numpy(), v.vowel.to_numpy(), v.sid.to_numpy())
            v = v.assign(pred=pred)
            res[cond] = v
            acc = (v.pred == v.vowel).mean()
            acc_hc = (v[v.label == 0].pred == v[v.label == 0].vowel).mean()
            acc_pd = (v[v.label == 1].pred == v[v.label == 1].vowel).mean()
            print(f"  P1 vowel identity ({cond}): accuracy {acc:.2f} (healthy {acc_hc:.2f}, PD {acc_pd:.2f}); chance 0.20")
            VERD.append(dict(model=mname, check=f"P1 vowel identity ({cond})", value=acc, healthy=acc_hc, pd=acc_pd,
                             passed=bool(acc_hc >= 0.6 and acc_pd >= 0.6)))
        cm = pd.crosstab(res["clean"].vowel, res["clean"].pred, normalize="index")
        print("  confusion (clean, rows = true vowel):")
        print(cm.round(2).to_string())
        cm.to_csv(OUT / f"pilot_vowel_confusion_{mname}.csv")

        # P2 vowel physiology (speaker-centred means, paired over speakers)
        v = V[V.cond == "clean"]
        M = v.groupby(["sid", "vowel"])[[f"{c}_c" for c in ("LIY", "TBY", "TDY", "lip_ap", "TBX")]].mean()
        means = M.groupby("vowel").mean()
        print("  speaker-centred mean posture by vowel:")
        print(means.round(3).to_string())
        means.to_csv(OUT / f"pilot_vowel_postures_{mname}.csv")

        def paired(col, v1, v2, expect):
            a = M.xs(v1, level="vowel")[col]
            b = M.xs(v2, level="vowel")[col]
            idx = a.index.intersection(b.index)
            d = (a[idx] - b[idx]).dropna()
            if len(d) < 8:
                return np.nan, np.nan, False
            p = wilcoxon(d).pvalue
            return float(d.mean()), float(p), bool(np.sign(d.mean()) == expect and p < 0.01)

        # y axis = up; the jaw (LIY) is LOWER for /a/ than /i/; the tongue is HIGHER for /i/, /u/ than /a/;
        # lips are more OPEN for /a/ than /u/
        tests = [("jaw lower for /a/ than /i/", "LIY_c", "a", "i", -1),
                 ("tongue body higher for /i/ than /a/", "TBY_c", "i", "a", 1),
                 ("tongue back higher for /u/ than /a/", "TDY_c", "u", "a", 1),
                 ("lips more open for /a/ than /u/", "lip_ap_c", "a", "u", 1),
                 ("lips more open for /a/ than /o/", "lip_ap_c", "a", "o", 1)]
        okc = 0
        for lab_, col, v1, v2, exp in tests:
            dm, p, ok = paired(col, v1, v2, exp)
            okc += ok
            print(f"  P2 {lab_:38s} diff {dm:+.3f}  p={p:.2g}  {'OK' if ok else 'NOT CONFIRMED'}")
            VERD.append(dict(model=mname, check=f"P2 {lab_}", value=dm, p=p, passed=ok))
        dfx, pfx, _ = paired("TBX_c", "i", "u", 1)
        print(f"  (front-back: tongue body x for /i/ minus /u/ = {dfx:+.3f}, p={pfx:.2g}; sign shows the x-axis direction)")

        # P3 pa vs ta
        d = R[(R.model == mname) & (R.kind == "ddk")].copy()
        lipdom = "D1"
        if len(d):
            w = d.pivot_table(index="sid", columns="task", values=["lip_range", "tt_range", "lip_rate_hz", "acoustic_rate"])
            if {"D1", "D2"} <= set(w.columns.get_level_values(1)):
                ratio1 = np.log(w["lip_range"]["D1"] / w["tt_range"]["D1"])
                ratio2 = np.log(w["lip_range"]["D2"] / w["tt_range"]["D2"])
                dd = (ratio1 - ratio2).dropna()
                p = wilcoxon(dd).pvalue if len(dd) >= 8 else np.nan
                lipdom = "D1" if dd.mean() > 0 else "D2"
                print(f"  P3 lip-vs-tongue dominance, D1 minus D2 (log ratio): {dd.mean():+.3f}, p={p:.2g} -> lips dominate in {lipdom}")
                ok = p < 0.01
                VERD.append(dict(model=mname, check="P3 pa (lips) vs ta (tongue tip) separated", value=float(dd.mean()), p=p,
                                 passed=bool(ok), note=f"lip-dominant task = {lipdom}"))
                for grp, lab_ in ((0, "healthy"), (1, "PD")):
                    ids = d[d.label == grp].sid.unique()
                    ddg = dd[dd.index.isin(ids)]
                    if len(ddg) >= 6:
                        print(f"     {lab_}: {ddg.mean():+.3f}, p={wilcoxon(ddg).pvalue:.2g}, n={len(ddg)}")
            rr = d.dropna(subset=["lip_rate_hz", "acoustic_rate"])
            rr = rr.assign(ema_rate=np.where(rr.task == lipdom, rr.lip_rate_hz, rr.tt_rate_hz))
            if len(rr) >= 10:
                rho, p = spearmanr(rr.ema_rate, rr.acoustic_rate)
                print(f"  P3b repetition rate from movements vs acoustic syllable rate: rho={rho:.2f} (p={p:.2g}), "
                      f"median {rr.ema_rate.median():.2f} vs {rr.acoustic_rate.median():.2f} Hz")
                VERD.append(dict(model=mname, check="P3b movement rate = acoustic syllable rate", value=rho, p=p,
                                 passed=bool(rho >= 0.5)))

        # P4 phone audio
        if "phone" in res:
            acc_c = (res["clean"].pred == res["clean"].vowel).mean()
            acc_p = (res["phone"].pred == res["phone"].vowel).mean()
            VERD.append(dict(model=mname, check="P4a IPVS vowels in phone band still recognisable", value=acc_p,
                             passed=bool(acc_p >= acc_c - 0.2 and acc_p >= 0.5)))
            print(f"  P4a phone-band vowel accuracy {acc_p:.2f} (clean {acc_c:.2f})")
        F = R[(R.model == mname) & (R.kind == "figshare")].copy()
        if len(F):
            v = V[V.cond == "phone"].dropna(subset=CH)
            F_c = F[CH] - v[CH].mean()          # Figshare speakers have only /a/: place them in the IPVS phone-band space
            Vc = v[CH] - v[CH].mean()
            lda = LinearDiscriminantAnalysis().fit(Vc.to_numpy(), v.vowel.to_numpy())
            pa = (lda.predict(F_c.to_numpy()) == "a").mean()
            raw = LinearDiscriminantAnalysis().fit(v[CH].to_numpy(), v.vowel.to_numpy()).predict(F[CH].to_numpy())
            print(f"  P4b Figshare /a/ recognised as /a/: {pa:.2f} (uncentred model: {(raw == 'a').mean():.2f}); chance 0.20")
            VERD.append(dict(model=mname, check="P4b Figshare phone /a/ looks like /a/", value=max(pa, (raw == 'a').mean()),
                             passed=bool(max(pa, (raw == "a").mean()) >= 0.5)))

        # P5 test-retest
        v = V[V.cond == "clean"]
        w = v.pivot_table(index=["sid", "vowel"], columns="take", values=[f"{c}_c" for c in CH])
        iccs = []
        for c in CH:
            try:
                ab = w[f"{c}_c"].dropna()
                if ab.shape[1] >= 2 and len(ab) >= 10:
                    iccs.append(icc21(ab.iloc[:, 0].to_numpy(), ab.iloc[:, 1].to_numpy()))
            except Exception:  # noqa: BLE001
                pass
        if iccs:
            print(f"  P5 test-retest ICC of vowel postures: median {np.median(iccs):.2f} (range {min(iccs):.2f}-{max(iccs):.2f})")
            VERD.append(dict(model=mname, check="P5 test-retest of postures (median ICC)", value=float(np.median(iccs)),
                             passed=bool(np.median(iccs) >= 0.6)))

        # preview (exploratory)
        section(f"PREVIEW (exploratory) - model '{mname}'")
        PV = []
        if len(d):
            for col in ("lip_range", "lip_speed", "lip_decrement", "tt_range", "tt_speed", "tt_decrement", "jaw_range"):
                for task in sorted(d.task.unique()):
                    g = d[d.task == task]
                    a_, b_ = g[g.label == 1][col].dropna(), g[g.label == 0][col].dropna()
                    if len(a_) >= 5 and len(b_) >= 5:
                        gg, _ = hedges_g(a_, b_)
                        PV.append(dict(source=f"DDK {task}", measure=col, g=gg, p=mannwhitneyu(a_, b_).pvalue,
                                       auc=roc_auc_score(np.r_[np.ones(len(a_)), np.zeros(len(b_))], np.r_[a_, b_])))
        for src, sub in (("IPVS /a/", V[(V.cond == "clean") & (V.vowel == "a")]), ("Figshare /a/", F)):
            for col in ("trem_LIY", "trem_TBY", "trem_TDY", "trem_LLY"):
                if len(sub) and col in sub:
                    g_ = sub.groupby("sid").agg({col: "mean", "label": "first"})
                    a_, b_ = g_[g_.label == 1][col].dropna(), g_[g_.label == 0][col].dropna()
                    if len(a_) >= 5 and len(b_) >= 5:
                        gg, _ = hedges_g(a_, b_)
                        PV.append(dict(source=src, measure=col.replace("trem_", "tremor 3-7 Hz "), g=gg,
                                       p=mannwhitneyu(a_, b_).pvalue,
                                       auc=roc_auc_score(np.r_[np.ones(len(a_)), np.zeros(len(b_))], np.r_[a_, b_])))
        PV = pd.DataFrame(PV)
        PV.to_csv(OUT / f"pilot_preview_{mname}.csv", index=False)
        print(PV.round(3).to_string(index=False) if len(PV) else "  -")

    VD = pd.DataFrame(VERD)
    VD.to_csv(OUT / "pilot_verdict.csv", index=False)
    section("PILOT VERDICT (GO if P1-P3 pass for at least one model; P4 decides whether Figshare can be used)")
    print(VD.round(3).to_string(index=False))
    L = ["# Pilot: virtual articulography (SPARC) on the 3 datasets", "", md_table(VD.round(3))]
    (OUT / "report_pilot.md").write_text("\n".join(L))

    # figure: vowel postures (speaker-centred) in tongue-height x lip-opening space, healthy vs PD
    m0 = VD.model.iloc[0] if len(VD) else None
    if m0 is not None:
        V = R[(R.model == m0) & (R.kind == "vowel") & (R.cond == "clean")].copy()
        V[["TBY", "lip_ap", "LIY"]] = V[["TBY", "lip_ap", "LIY"]] - V.groupby("sid")[["TBY", "lip_ap", "LIY"]].transform("mean")
        fig_, ax = plt.subplots(figsize=(5.2, 4.2))
        cols = dict(zip(VOWELS, [C["FIGSHARE"], C["IPVS"], C["MDVR"], C["ink2"], C["muted"]]))
        for vw in VOWELS:
            for lb, mk in ((0, "o"), (1, "x")):
                g = V[(V.vowel == vw) & (V.label == lb)]
                ax.scatter(g.lip_ap, g.TBY, s=10, color=cols[vw], marker=mk, alpha=0.6,
                           label=f"/{vw}/ {'PD' if lb else 'healthy'}" if lb == 0 or vw == "a" else None)
        ax.set_xlabel("lip opening (speaker-centred)")
        ax.set_ylabel("tongue-body height (speaker-centred)")
        ax.set_title(f"Estimated mouth posture of Italian vowels ({m0})", loc="left")
        ax.legend(fontsize=6, ncol=2)
        fig_.tight_layout()
        fig_.savefig(OUT / "fig45_vowel_postures.png", dpi=150)
        plt.close(fig_)


def main19():
    buf = io.StringIO()
    with redirect_stdout(Tee(sys.stdout, buf)):
        step19()
    (OUT / "report.txt").write_text(buf.getvalue())


if __name__ == "__main__":
    main19()
