"""STEP 27 - Does the inversion degrade on Parkinsonian speech? A label-free check by resynthesis.

No articulograph data exist for the patients, so the inversion cannot be compared with sensors in PD. Instead, each
reading recording is passed through the SPARC analysis model (audio -> articulator trajectories + source features) and
back through its synthesis model (-> audio). If the inversion were systematically worse for dysarthric speech, the
resynthesised audio would match the original less well in patients, and the mismatch would grow with severity
(resynthesis-based evaluation as proposed by Wu et al., ICASSP 2023).

Fidelity per recording (first 20 s of speech, both signals at 16 kHz, frame-aligned):
  * mel-cepstral distortion (MCD, dB; coefficients 1-12, speech frames only; lower = more faithful)
  * log-mel spectral correlation (higher = more faithful)
Analyses: PD vs healthy per dataset (sex-adjusted, SD units); association with UPDRS II-5, III-18 and H&Y (MDVR).
Outputs: resynthesis_fidelity.csv, resynthesis_group.csv, resynthesis_severity.csv, report.md
"""
import os
import subprocess
import sys
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

INPUT = Path(os.environ.get("PDV_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("PDV_OUTPUT", "/kaggle/working"))
SR16 = 16000
TARGET_DBFS = -25.0
EXCERPT_S = 20.0


def find_root_with(fname, sibling_dir=None):
    for root, dirs, files in os.walk(INPUT):
        if fname in files and (sibling_dir is None or sibling_dir in dirs):
            return Path(root)
    return None


def resample(x, sr, target):
    from scipy.signal import resample_poly
    if sr == target:
        return x
    g_ = gcd(int(sr), int(target))
    return resample_poly(x, target // g_, sr // g_)


def frames_db(x, sr, frame_s=0.025, hop_s=0.010):
    n, h = int(round(frame_s * sr)), int(round(hop_s * sr))
    if len(x) < n:
        return np.array([-120.0]), h
    idx = np.arange(n)[None, :] + h * np.arange(1 + (len(x) - n) // h)[:, None]
    return 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12), h


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


def fidelity(x, y):
    import librosa
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    kw = dict(sr=SR16, n_fft=512, hop_length=160, n_mels=40)
    mx = librosa.feature.melspectrogram(y=x, **kw); my = librosa.feature.melspectrogram(y=y, **kw)
    lx, ly = np.log(mx + 1e-8), np.log(my + 1e-8)
    e = 10 * np.log10(mx.sum(0) + 1e-12); speech = e > np.percentile(e, 95) - 30
    cx = librosa.feature.mfcc(S=lx, n_mfcc=13)[1:, speech]; cy = librosa.feature.mfcc(S=ly, n_mfcc=13)[1:, speech]
    mcd = float(np.mean(10 / np.log(10) * np.sqrt(2 * np.sum((cx - cy) ** 2, axis=0))))
    r = float(np.corrcoef(lx[:, speech].ravel(), ly[:, speech].ravel())[0, 1])
    return mcd, r


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speech-articulatory-coding"], check=False)
    import statsmodels.api as sm
    import torch
    from scipy.stats import spearmanr
    from sparc import load_model
    coder = load_model("multi", device="cuda:0" if torch.cuda.is_available() else "cpu")
    s02, s03 = find_root_with("manifest.csv", "clean_audio"), find_root_with("features_subject.csv")
    man = pd.read_csv(s02 / "manifest.csv")
    S3 = pd.read_csv(s03 / "features_subject.csv")
    man["sex"] = man.subject_id.map(S3.drop_duplicates("subject_id").set_index("subject_id").sex).fillna(man.get("sex"))
    rd = man[man.task_family.eq("read_passage") & man.dataset.isin(["MDVR", "IPVS"]) & man.group.isin(["PD", "HC", "EHC"])]
    rd = rd.sort_values("task").drop_duplicates("subject_id")          # one reading per speaker
    rows = []
    for r in rd.itertuples():
        try:
            x, sr = sf.read(str(s02 / "clean_audio" / r.clean_path), dtype="float64")
            x = x.mean(axis=1) if x.ndim > 1 else x
            x = trim_and_normalise(x if sr == SR16 else resample(x, sr, SR16), SR16)[: int(EXCERPT_S * SR16)]
            code = coder.encode(np.asarray(x, np.float32))
            try:
                y = coder.decode(**code)
            except TypeError:
                y = coder.decode(**{k: code[k] for k in ("ema", "pitch", "loudness", "spk_emb") if k in code})
            y = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
            y = resample(np.asarray(y, float).ravel(), int(getattr(coder, "sr", SR16)), SR16)
            mcd, rr = fidelity(x, y)
            rows.append(dict(sid=r.subject_id, dataset=r.dataset, label=int(r.label), male=float(r.sex == "M"),
                             hy=r.hoehn_yahr, updrs2=r.updrs_ii5, updrs3=r.updrs_iii18, mcd_db=mcd, mel_r=rr))
            print(f"  {r.dataset} {r.subject_id[-6:]} label={r.label} MCD={mcd:.2f} r={rr:.3f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"  failed {r.subject_id}: {ex}", flush=True)
    F = pd.DataFrame(rows); F.to_csv(OUT / "resynthesis_fidelity.csv", index=False)
    G = []
    for ds, g in F.groupby("dataset"):
        for m in ("mcd_db", "mel_r"):
            d = g.dropna(subset=[m]); sd = d[m].std()
            f = sm.OLS((d[m] - d[m].mean()) / sd, sm.add_constant(d[["label", "male"]])).fit(cov_type="HC3")
            G.append(dict(dataset=ds, metric=m, mean_pd=d[d.label == 1][m].mean(), mean_hc=d[d.label == 0][m].mean(),
                          beta=float(f.params["label"]), ci_lo=float(f.conf_int().loc["label", 0]),
                          ci_hi=float(f.conf_int().loc["label", 1]), p=float(f.pvalues["label"])))
    G = pd.DataFrame(G); G.to_csv(OUT / "resynthesis_group.csv", index=False)
    V = []
    pdm = F[(F.dataset == "MDVR") & (F.label == 1)]
    for m in ("mcd_db", "mel_r"):
        for sc in ("updrs2", "updrs3", "hy"):
            d = pdm[[m, sc]].apply(pd.to_numeric, errors="coerce").dropna()
            rho, p = spearmanr(d[m], d[sc]); V.append(dict(metric=m, scale=sc, n=len(d), rho=rho, p=p))
    V = pd.DataFrame(V); V.to_csv(OUT / "resynthesis_severity.csv", index=False)
    rep = ["# Resynthesis check", "", f"recordings: {len(F)}", "", "## PD vs healthy (sex-adjusted, SD units)", "",
           G.round(3).to_string(index=False), "", "## Association with severity (MDVR patients)", "",
           V.round(3).to_string(index=False)]
    (OUT / "report.md").write_text("\n".join(rep)); print("\n".join(rep))


if __name__ == "__main__":
    main()
