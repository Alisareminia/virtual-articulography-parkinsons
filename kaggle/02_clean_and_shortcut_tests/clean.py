"""PD voice project - step 02: fixed manifest, harmonised audio, shortcut (confound) tests.

Runs on Kaggle. Builds on the step-01 audit and fixes every issue it found:
  * IPVS speaker identity: union of folder + identity code (merges the 3 PD people recorded in
    2016 and 2017, and the birth-year typo); young controls identified by folder (their file
    names reuse a template code). Age = session year - birth year from the code, cross-checked
    against the metadata spreadsheets (names are used for matching only, never written out).
  * Exact duplicate audio dropped (one PD speaker's B2 is a byte copy of B1).
  * Figshare demographics joined on the full sample ID; MDVR 'ID22hc' name parsed.
  * Every file: mono, DC removed, resampled to 16 kHz (Figshare stays at its native 8 kHz),
    identical energy-based trimming of leading/trailing silence, identical loudness normalisation.
  * Shortcut tests: can PD be predicted from recording artefacts alone (onset level, noise floor,
    non-speech spectrum), before vs after cleaning? Subject-grouped CV, subject-level AUC,
    label-permutation p-values.

Outputs in /kaggle/working:
  clean_audio/...              harmonised wavs (stay on Kaggle; used by later kernels)
  manifest.csv                 one row per kept file (pseudonymised)
  subjects.csv                 one row per unique speaker
  file_features.csv            recording-artefact + reference features, raw and clean
  shortcut_results.csv         AUCs for every subset x feature set
  report.txt, fig*.png
"""
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import warnings
import zipfile
from contextlib import redirect_stdout
from math import gcd
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
CLEAN = OUT / "clean_audio"
TMP = Path("/tmp/pdvoice")
TMP.mkdir(parents=True, exist_ok=True)
MDVR_URL = "https://zenodo.org/records/2867216/files/26_29_09_2017_KCL.zip?download=1"

TARGET_SR = 16000
FRAME_S, HOP_S = 0.025, 0.010
VAD_REL_DB = 30.0   # a frame is "active" if within 30 dB of the file's 95th-percentile frame energy
PAD_S = 0.05        # padding kept around the first/last active frame
TARGET_DBFS = -25.0  # RMS of active frames after normalisation
N_MELS = 32
MIN_NS_FRAMES = 10   # non-speech spectrum needs at least this many non-speech frames

# chart tokens (reference palette, light mode); colour follows the group, never its rank
C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
         HC="#2a78d6", EHC="#2a78d6", PD="#eb6834", YHC="#1baf7a")
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "text.color": C["ink"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False,
})


# ============================================================================= helpers
def find_dir(name):
    for root, dirs, _ in os.walk(INPUT):
        if name in dirs:
            return Path(root) / name
    return None


def find_file(name):
    for root, _, files in os.walk(INPUT):
        if name in files:
            return Path(root) / name
    return None


def pseudo(*parts, n=8):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:n]


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# ============================================================================= signal processing
def load_mono(path):
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    ch = x.shape[1]
    x = x.mean(axis=1)
    return x - x.mean(), sr, ch


def resample(x, sr, target):
    if sr == target:
        return x
    g = gcd(int(sr), int(target))
    return resample_poly(x, target // g, sr // g)


def frame_params(sr):
    return int(round(FRAME_S * sr)), int(round(HOP_S * sr))


def frame_db(x, sr):
    n, h = frame_params(sr)
    if len(x) < n:
        return np.array([-120.0])
    idx = np.arange(n)[None, :] + h * np.arange(1 + (len(x) - n) // h)[:, None]
    return 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)


def active_mask(e):
    return e > (np.percentile(e, 95) - VAD_REL_DB)


def db(x):
    return float(10 * np.log10(np.mean(x ** 2) + 1e-12)) if len(x) else np.nan


def trim_and_normalise(x, sr):
    """Identical rule for every file: cut leading/trailing non-active audio, normalise active RMS."""
    e = frame_db(x, sr)
    act = active_mask(e)
    n, h = frame_params(sr)
    first = int(np.argmax(act))
    last = len(act) - 1 - int(np.argmax(act[::-1]))
    start = max(0, int(first * h - PAD_S * sr))
    end = min(len(x), int(last * h + n + PAD_S * sr))
    y = x[start:end].copy()
    e2 = frame_db(y, sr)
    a2 = active_mask(e2)
    rms_act = np.sqrt(np.mean(10 ** (e2[a2] / 10))) if a2.any() else np.sqrt(np.mean(y ** 2))
    gain = 10 ** (TARGET_DBFS / 20) / (rms_act + 1e-12)
    y *= gain
    peak = np.max(np.abs(y)) if len(y) else 0
    if peak > 0.99:
        y *= 0.99 / peak
        gain *= 0.99 / peak
    return y, dict(trim_lead_s=start / sr, trim_tail_s=(len(x) - end) / sr, gain_db=20 * np.log10(gain + 1e-12))


def bw_at(p, freqs, q):
    c = np.cumsum(p) / (np.sum(p) + 1e-20)
    return float(freqs[min(np.searchsorted(c, q), len(freqs) - 1)])


def features(x, sr, with_mfcc=False):
    """Recording-artefact descriptors (+ optional MFCC reference) for one signal."""
    n, h = frame_params(sr)
    e = frame_db(x, sr)
    act = active_mask(e)
    f = dict(
        duration_s=len(x) / sr,
        lead_200ms_db=db(x[: int(0.2 * sr)]),
        tail_200ms_db=db(x[-int(0.2 * sr):]),
        noise_floor_db=float(np.percentile(e, 10)),
        active_frac=float(act.mean()),
        ns_frames=int((~act).sum()),
        ns_level_db=float(np.mean(e[~act])) if (~act).sum() >= MIN_NS_FRAMES else np.nan,
    )
    if len(x) < n:
        return f
    P = np.abs(librosa.stft(x, n_fft=n, hop_length=h, win_length=n, center=False)) ** 2  # (freq, frames)
    k = min(P.shape[1], len(act))
    P, act = P[:, :k], act[:k]
    freqs = np.fft.rfftfreq(n, 1 / sr)
    if act.any():
        f["sp_bw95_hz"] = bw_at(P[:, act].mean(axis=1), freqs, 0.95)
    fmax = min(sr / 2, 8000)
    M = librosa.filters.mel(sr=sr, n_fft=n, n_mels=N_MELS, fmin=50, fmax=fmax)
    if (~act).sum() >= MIN_NS_FRAMES:
        pn = P[:, ~act].mean(axis=1)
        f["ns_bw95_hz"] = bw_at(pn, freqs, 0.95)
        mel = 10 * np.log10(M @ pn + 1e-12)
        for i, v in enumerate(mel):
            f[f"ns_mel_{i:02d}"] = float(v)
    if with_mfcc and act.sum() >= 5:
        S = librosa.power_to_db(M @ P[:, act])
        mf = librosa.feature.mfcc(S=S, n_mfcc=13)
        for i in range(13):
            f[f"mfcc{i:02d}_mean"] = float(mf[i].mean())
            f[f"mfcc{i:02d}_std"] = float(mf[i].std())
    return f


# ============================================================================= manifests (fixed parsing)
IPVS_GROUPS = {"15 Young Healthy Control": "YHC", "22 Elderly Healthy Control": "EHC",
               "28 People with Parkinsons disease": "PD"}
IPVS_RE = re.compile(r"^([A-Z]+\d)([A-Za-z]+?)(\d{2})([MF])(\d+)$")
IPVS_TASK_RE = re.compile(r"^(VA|VE|VI|VO|VU|B|D|FB|PR)(\d)")


def ipvs_task_family(t):
    base = t.rstrip("0123456789")
    if base.startswith("V") and len(base) == 2:
        return "vowel_" + base[1].lower()
    return {"B": "read_passage", "D": "syllables", "FB": "phrases", "PR": "words"}.get(base, "other")


def read_ipvs_sheet(path):
    """Return {folder-style key: (sex, age)} from an IPVS metadata sheet (header row contains 'name')."""
    raw = pd.read_excel(path, header=None)
    hdr = None
    for i in range(min(6, len(raw))):
        if any(str(v).strip().lower() == "name" for v in raw.iloc[i]):
            hdr = i
            break
    if hdr is None:
        return {}
    cols = [str(v).strip().lower() for v in raw.iloc[hdr]]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols
    out = {}
    for _, r in df.iterrows():
        name, sur = str(r.get("name", "")).strip(), str(r.get("surname", "")).strip()
        if name in ("", "nan") or sur in ("", "nan"):
            continue
        key = f"{name.lower()} {sur[0].lower()}"
        age = pd.to_numeric(r.get("age"), errors="coerce")
        out[key] = (str(r.get("sex", "")).strip().upper()[:1] or None, age)
    return out


def manifest_ipvs():
    root = find_dir("28 People with Parkinsons disease").parent
    rows = []
    for gdir, g in IPVS_GROUPS.items():
        for p in sorted((root / gdir).rglob("*.wav")):
            stem = p.stem.rstrip(".")
            folder = str(p.relative_to(root / gdir).parent)
            r = dict(dataset="IPVS", group=g, label=int(g == "PD"), path=str(p), orig_name=p.name, folder=folder)
            m = IPVS_RE.match(stem)
            if m:
                task, code, yy, sex, stamp = m.groups()
                r.update(task=task, code_key="".join(sorted(code.upper())) + sex, birth_yy=int(yy), sex=sex)
                if len(stamp) == 12:
                    r["session_date"] = f"{stamp[4:8]}-{stamp[2:4]}-{stamp[0:2]}"
                elif len(stamp) == 10:
                    r["session_date"] = f"20{stamp[4:6]}-{stamp[2:4]}-{stamp[0:2]}"
            else:
                tm = IPVS_TASK_RE.match(stem.upper())
                r["task"] = tm.group(0) if tm else "?"
            r["task_family"] = ipvs_task_family(r["task"])
            rows.append(r)
    df = pd.DataFrame(rows)

    # --- identity: union of folder and (letters + sex) code; YHC by folder only (template codes)
    uf = UnionFind()
    for _, r in df.iterrows():
        fnode = f"F|{r.group}|{r.folder}"
        uf.find(fnode)
        if r.group != "YHC" and isinstance(r.get("code_key"), str):
            uf.union(fnode, f"K|{r.group}|{r.code_key}")
    df["subject_id"] = [f"IPVS_{r.group}_{pseudo(uf.find(f'F|{r.group}|{r.folder}'))}" for _, r in df.iterrows()]

    # --- session date: fill from the same folder when the file name lacked one
    df["session_date"] = df.groupby(["group", "folder"])["session_date"].transform(lambda s: s.fillna(s.dropna().iloc[0]) if s.notna().any() else s)
    df["session_year"] = pd.to_numeric(df["session_date"].str[:4], errors="coerce")

    # --- age: birth year in code (not YHC), cross-checked against spreadsheets
    df["age_code"] = np.where(df.group != "YHC", df.session_year - (1900 + df.birth_yy), np.nan)
    sheets = {}
    for p in root.rglob("*.xlsx"):
        if "FILE CODES" in p.name.upper():
            continue
        g = next((v for k, v in IPVS_GROUPS.items() if k in str(p)), None)
        for k, v in read_ipvs_sheet(p).items():
            sheets[(g, k)] = v
    fkey = df.folder.map(lambda f: f.split("/")[-1].strip().lower())
    sheet_vals = [sheets.get((g, k)) for g, k in zip(df.group, fkey)]
    df["age_sheet"] = [v[1] if v else np.nan for v in sheet_vals]
    df["sex_sheet"] = [v[0] if v else None for v in sheet_vals]
    df["age"] = df["age_code"].fillna(df["age_sheet"])
    df["sex"] = df["sex"].fillna(df["sex_sheet"])
    # per-subject consistency: one sex, age at first session
    df["sex"] = df.groupby("subject_id")["sex"].transform(lambda s: s.mode().iloc[0] if s.notna().any() else None)
    df["age"] = df.groupby("subject_id")["age"].transform("min")
    return df


def manifest_figshare():
    rows = []
    for gdir, g in (("HC_AH", "HC"), ("PD_AH", "PD")):
        for p in sorted(find_dir(gdir).rglob("*.wav")):
            rows.append(dict(dataset="FIGSHARE", group=g, label=int(g == "PD"), path=str(p), orig_name=p.name,
                             sample_id=p.stem, task="A", task_family="vowel_a"))
    df = pd.DataFrame(rows)
    demo = pd.read_excel(find_file("Demographics_age_sex.xlsx"))
    demo.columns = [c.strip() for c in demo.columns]
    demo["Sample ID"] = demo["Sample ID"].astype(str).str.strip()
    df = df.merge(demo.rename(columns={"Sample ID": "sample_id", "Label": "label_demo", "Age": "age", "Sex": "sex"}),
                  on="sample_id", how="left")
    df["subject_id"] = "FIGSHARE_" + df.group + "_" + df.sample_id.map(pseudo)
    return df


MDVR_RE = re.compile(r"^(ID\d+)_?(hc|pd)_(\d+)_(\d+)_(\d+)$", re.I)


def manifest_mdvr():
    z = TMP / "mdvr.zip"
    if not z.exists():
        subprocess.run(["wget", "-q", "-O", str(z), MDVR_URL], check=True)
    ext = TMP / "mdvr"
    if not ext.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ext)
    rows = []
    for p in sorted(ext.rglob("*.wav")):
        m = MDVR_RE.match(p.stem)
        task = "read_passage" if "read" in str(p).lower() else "dialogue" if "dialog" in str(p).lower() else "?"
        if not m:
            print(f"  MDVR: skipping unparseable file name {p.name}")
            continue
        sid, g, hy, u2, u3 = m.groups()
        rows.append(dict(dataset="MDVR", group=g.upper(), label=int(g.lower() == "pd"), path=str(p), orig_name=p.name,
                         task=task, task_family=task, subject_id=f"MDVR_{g.upper()}_{pseudo(sid.upper())}",
                         hoehn_yahr=int(hy), updrs_ii5=int(u2), updrs_iii18=int(u3)))
    return pd.DataFrame(rows)


# ============================================================================= shortcut tests
def make_model(kind):
    if kind == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                                n_jobs=-1, random_state=0))


def subject_auc(X, y, g, kind, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, g):
        m = make_model(kind).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    s = pd.DataFrame(dict(g=g, y=y, p=p)).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    return roc_auc_score(s.y, s.p)


def shortcut_test(df, cols, name, subset, repeats=10, n_perm=100):
    cols = [c for c in cols if c in df and df[c].notna().mean() >= 0.5]
    if not cols or df.label.nunique() < 2:
        return dict(subset=subset, feature_set=name, n_features=0)
    X = df[cols].to_numpy(float)
    y = df.label.to_numpy()
    g = df.subject_id.to_numpy()
    lr = [subject_auc(X, y, g, "lr", s) for s in range(repeats)]
    rf = [subject_auc(X, y, g, "rf", s) for s in range(max(3, repeats // 2))]
    # permutation null: shuffle labels at SUBJECT level, keep grouping
    rng = np.random.default_rng(0)
    subj = pd.Series(y, index=g).groupby(level=0).first()
    null = []
    for _ in range(n_perm):
        perm = pd.Series(rng.permutation(subj.values), index=subj.index)
        null.append(subject_auc(X, perm.loc[g].to_numpy(), g, "lr", 0))
    obs = float(np.mean(lr))
    return dict(subset=subset, feature_set=name, n_features=len(cols), n_files=len(df),
                n_subj_pd=df.loc[df.label == 1, "subject_id"].nunique(),
                n_subj_ctrl=df.loc[df.label == 0, "subject_id"].nunique(),
                auc_lr=obs, auc_lr_lo=float(np.percentile(lr, 2.5)), auc_lr_hi=float(np.percentile(lr, 97.5)),
                auc_rf=float(np.mean(rf)), auc_rf_lo=float(np.min(rf)), auc_rf_hi=float(np.max(rf)),
                p_perm=(1 + np.sum(np.array(null) >= obs)) / (1 + n_perm))


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
    section("1. MANIFESTS (fixed parsing)")
    ipvs, fig, mdvr = manifest_ipvs(), manifest_figshare(), manifest_mdvr()
    man = pd.concat([ipvs, fig, mdvr], ignore_index=True)
    print(man.groupby(["dataset", "group"]).agg(files=("path", "size"), subjects=("subject_id", "nunique")).to_string())

    section("1a. IPVS identity resolution")
    per = ipvs.groupby("subject_id").agg(group=("group", "first"), folders=("folder", "nunique"),
                                         sessions=("session_date", lambda s: sorted(set(s.dropna()))))
    print(per.groupby("group").agg(subjects=("folders", "size"), merged_multi_folder=("folders", lambda s: (s > 1).sum())).to_string())
    print("\nsubjects with >1 folder (merged):")
    print(per[per.folders > 1].to_string())
    print(f"\nIPVS files still without a parsed code: {ipvs.code_key.isna().sum()} (identity from folder)")

    section("1b. IPVS age check: code-derived age vs spreadsheet age (per subject)")
    s = ipvs.drop_duplicates("subject_id")
    chk = ipvs.groupby("subject_id").agg(group=("group", "first"), age_code=("age_code", "min"), age_sheet=("age_sheet", "first"))
    both = chk.dropna()
    print(f"spreadsheet matched for {chk.age_sheet.notna().sum()}/{len(chk)} subjects;"
          f" both ages available for {len(both)}")
    if len(both):
        d = both.age_sheet - both.age_code
        print(f"sheet - code age difference: median {d.median():.1f}, |diff|<=1 for {(d.abs() <= 1).mean():.0%}")
    print("\nfinal demographics per group (one row per subject):")
    print(s.groupby("group").agg(n=("subject_id", "size"), age_mean=("age", "mean"), age_sd=("age", "std"),
                                 age_min=("age", "min"), age_max=("age", "max"), age_missing=("age", lambda a: a.isna().sum()),
                                 male=("sex", lambda x: (x == "M").sum()), female=("sex", lambda x: (x == "F").sum())).round(1).to_string())

    section("1c. Figshare demographics (joined on full sample ID)")
    print(f"matched: {fig.age.notna().sum()}/{len(fig)}; folder label vs sheet label agree: "
          f"{((fig.group == 'PD') == (fig.label_demo.astype(str).str.upper().str.contains('PD'))).mean():.0%}")
    print(fig.groupby("group").agg(n=("subject_id", "size"), age_mean=("age", "mean"), age_sd=("age", "std"),
                                   age_min=("age", "min"), age_max=("age", "max"),
                                   male=("sex", lambda x: (x == "M").sum()), female=("sex", lambda x: (x == "F").sum())).round(1).to_string())
    print(f"controls younger than the youngest PD ({fig.loc[fig.group == 'PD', 'age'].min():.0f} y): "
          f"{(fig.loc[fig.group == 'HC', 'age'] < fig.loc[fig.group == 'PD', 'age'].min()).sum()}")

    section("1d. MDVR-KCL")
    print(mdvr.groupby(["task", "group"]).agg(files=("path", "size"), subjects=("subject_id", "nunique")).to_string())

    # ------------------------------------------------------------------ 2. process every file
    section("2. HARMONISATION (mono, DC removed, 16 kHz [Figshare 8 kHz], identical trim + loudness)")
    feats, md5s = [], []
    for i, r in man.iterrows():
        x, sr, ch = load_mono(r.path)
        md5s.append(hashlib.md5(np.round(x, 6).tobytes()).hexdigest())
        fr = {f"raw_{k}": v for k, v in features(x, sr).items()}
        tsr = 8000 if r.dataset == "FIGSHARE" else TARGET_SR
        y = resample(x, sr, tsr)
        y, tinfo = trim_and_normalise(y, tsr)
        fc = {f"clean_{k}": v for k, v in features(y, tsr, with_mfcc=True).items()}
        file_id = pseudo(r.dataset, r.path, n=12)
        rel = Path(r.dataset) / r.subject_id / f"{r.task}_{file_id}.wav"
        (CLEAN / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(CLEAN / rel, y.astype(np.float32), tsr, subtype="PCM_16")
        feats.append(dict(idx=i, file_id=file_id, clean_path=str(rel), orig_sr=sr, orig_channels=ch,
                          clean_sr=tsr, **tinfo, **fr, **fc))
        if i % 100 == 0:
            print(f"  processed {i}/{len(man)}")
    F = pd.DataFrame(feats).set_index("idx")
    man = man.join(F)
    man["audio_md5"] = md5s

    dup = man[man.duplicated("audio_md5", keep="first")]
    print(f"\nexact duplicate audio dropped: {len(dup)} file(s) -> {dup[['dataset', 'group', 'task']].values.tolist()}")
    for p in dup.clean_path:
        (CLEAN / p).unlink(missing_ok=True)
    man = man.drop(dup.index)
    man["is_44k"] = (man.orig_sr == 44100).astype(int)

    print("\nonset level (first 200 ms, dB) - median by dataset/group, raw vs clean:")
    print(man.groupby(["dataset", "group"])[["raw_lead_200ms_db", "clean_lead_200ms_db",
                                             "raw_noise_floor_db", "clean_noise_floor_db"]].median().round(1).to_string())

    # ------------------------------------------------------------------ 3. shortcut tests
    section("3. SHORTCUT TESTS - can diagnosis be predicted from recording artefacts alone?")
    print("Subject-grouped 5-fold CV (10 repeats LR, 5 RF), predictions averaged per subject, subject-level AUC.\n"
          "p_perm: subject-level label permutation (100x). AUC ~0.5 = no shortcut; high AUC = recording bias.\n"
          "clean_speech_mfcc is a REFERENCE (voice + any residual channel), not a shortcut test.")
    ns_raw = [f"raw_ns_mel_{i:02d}" for i in range(N_MELS)]
    ns_cln = [f"clean_ns_mel_{i:02d}" for i in range(N_MELS)]
    rec_raw = ["is_44k", "raw_lead_200ms_db", "raw_tail_200ms_db", "raw_noise_floor_db", "raw_ns_level_db", "raw_ns_bw95_hz"]
    rec_cln = ["clean_lead_200ms_db", "clean_tail_200ms_db", "clean_noise_floor_db", "clean_ns_level_db", "clean_ns_bw95_hz"]
    mfcc = [c for c in man.columns if c.startswith("clean_mfcc")]
    fsets = {"raw: recording-level": rec_raw, "raw: non-speech spectrum": ns_raw,
             "clean: recording-level": rec_cln, "clean: non-speech spectrum": ns_cln,
             "clean: speech MFCC (reference)": mfcc}
    ip = man[(man.dataset == "IPVS") & (man.group != "YHC")]
    subsets = {
        "IPVS PD vs EHC - all": ip,
        "IPVS PD vs EHC - 2017 only": ip[ip.session_year == 2017],
        "IPVS 2017 - vowels": ip[(ip.session_year == 2017) & ip.task_family.str.startswith("vowel")],
        "IPVS 2017 - reading": ip[(ip.session_year == 2017) & (ip.task_family == "read_passage")],
        "MDVR - reading": man[(man.dataset == "MDVR") & (man.task == "read_passage")],
        "MDVR - dialogue": man[(man.dataset == "MDVR") & (man.task == "dialogue")],
        "FIGSHARE - /a/": man[man.dataset == "FIGSHARE"],
    }
    res = []
    for sn, sd in subsets.items():
        for fn, cols in fsets.items():
            r = shortcut_test(sd, cols, fn, sn)
            res.append(r)
            if r.get("n_features"):
                print(f"  {sn:28s} | {fn:32s} | AUC LR {r['auc_lr']:.2f} [{r['auc_lr_lo']:.2f}-{r['auc_lr_hi']:.2f}]"
                      f"  RF {r['auc_rf']:.2f} | p={r['p_perm']:.3f} | subj {r['n_subj_pd']}/{r['n_subj_ctrl']}")
            else:
                print(f"  {sn:28s} | {fn:32s} | not enough data (e.g. no non-speech frames after trimming)")
    R = pd.DataFrame(res)
    R.to_csv(OUT / "shortcut_results.csv", index=False)

    # ------------------------------------------------------------------ 4. MDVR bandwidth check
    section("4. MDVR bandwidth gap: voice or recording? (per-subject medians, raw audio)")
    md = man[man.dataset == "MDVR"]
    for task in ["read_passage", "dialogue"]:
        t = md[md.task == task].groupby("subject_id").agg(g=("group", "first"), sp=("raw_sp_bw95_hz", "median"),
                                                          ns=("raw_ns_bw95_hz", "median"), nsl=("raw_ns_level_db", "median"))
        for col, lab in (("sp", "speech-frame bw95"), ("ns", "non-speech bw95"), ("nsl", "non-speech level dB")):
            a, b = t.loc[t.g == "HC", col].dropna(), t.loc[t.g == "PD", col].dropna()
            if len(a) > 2 and len(b) > 2:
                print(f"  {task:13s} {lab:20s} HC median {a.median():8.1f}  PD median {b.median():8.1f}  "
                      f"Mann-Whitney p={mannwhitneyu(a, b).pvalue:.4f}")
    print("  -> a gap in NON-SPEECH frames points to room/device/session; a gap only in speech frames points to voice.")

    # ------------------------------------------------------------------ 5. figures
    figures(man, R)

    # ------------------------------------------------------------------ 6. save
    subj_cols = ["dataset", "group", "label", "age", "sex", "hoehn_yahr", "updrs_ii5", "updrs_iii18"]
    subj = man.groupby("subject_id").agg(**{c: (c, "first") for c in subj_cols if c in man},
                                         n_files=("file_id", "size"),
                                         sessions=("session_date", lambda s: s.nunique()),
                                         has_2016=("session_year", lambda s: int((s == 2016).any()))).reset_index()
    subj.to_csv(OUT / "subjects.csv", index=False)
    mcols = ["file_id", "dataset", "subject_id", "group", "label", "task", "task_family", "session_date", "session_year",
             "age", "sex", "hoehn_yahr", "updrs_ii5", "updrs_iii18", "orig_sr", "orig_channels", "is_44k",
             "clean_sr", "clean_path", "trim_lead_s", "trim_tail_s", "gain_db"]
    man[[c for c in mcols if c in man]].to_csv(OUT / "manifest.csv", index=False)
    fcols = ["file_id"] + [c for c in man.columns if c.startswith(("raw_", "clean_")) and c != "clean_path" and c != "clean_sr"]
    man[fcols].to_csv(OUT / "file_features.csv", index=False)
    print(f"\nsaved manifest.csv ({len(man)} files), subjects.csv ({len(subj)} subjects), "
          f"file_features.csv, shortcut_results.csv, clean_audio/ ({sum(1 for _ in CLEAN.rglob('*.wav'))} wavs)")
    shutil.rmtree(TMP, ignore_errors=True)


def figures(man, R):
    rng = np.random.default_rng(1)
    # fig 1: IPVS onset level raw vs clean (the shortcut, before/after)
    ip = man[man.dataset == "IPVS"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, col, title in zip(axes, ["raw_lead_200ms_db", "clean_lead_200ms_db"], ["Raw files", "After harmonisation"]):
        for i, g in enumerate(["EHC", "PD", "YHC"]):
            v = ip.loc[ip.group == g, col].dropna()
            ax.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v, s=9, color=C[g], alpha=0.55, linewidths=0)
            ax.hlines(v.median(), i - 0.34, i + 0.34, color=C["ink"], linewidth=2)
            ax.text(i + 0.36, v.median(), f"{v.median():.0f}", va="center", fontsize=8, color=C["ink2"])
        ax.set_xticks(range(3), ["Elderly ctrl", "PD", "Young ctrl"])
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Level of first 200 ms (dBFS)")
    fig.suptitle("IPVS: how each file begins, by group (bar = median)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_ipvs_onset_level.png", dpi=150)
    plt.close(fig)

    # fig 2: mean non-speech spectrum by group, raw (small multiples)
    panels = [("IPVS 2017", (man.dataset == "IPVS") & (man.session_year == 2017) & (man.group != "YHC")),
              ("MDVR reading", (man.dataset == "MDVR") & (man.task == "read_passage")),
              ("Figshare /a/", man.dataset == "FIGSHARE")]
    cols = [f"raw_ns_mel_{i:02d}" for i in range(N_MELS)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (title, mask) in zip(axes, panels):
        d = man[mask]
        for g in [g for g in ["HC", "EHC", "PD"] if g in set(d.group)]:
            m = d.loc[d.group == g, cols].mean()
            if m.notna().sum() == 0:
                continue
            ax.plot(range(N_MELS), m.values, color=C[g], linewidth=2)
            ax.text(N_MELS - 0.5, m.values[-1], " PD" if g == "PD" else " Control", color=C["ink2"], fontsize=8, va="center")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Mel band (low -> high frequency)")
        ax.set_xlim(0, N_MELS + 4)
    axes[0].set_ylabel("Mean non-speech power (dB)")
    fig.suptitle("Background (non-speech) spectrum by group, raw recordings", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_nonspeech_spectrum.png", dpi=150)
    plt.close(fig)

    # fig 3: shortcut AUCs, raw vs clean panels; series = feature family (max 3 per panel)
    R = R[R.n_features > 0].copy()
    subsets = list(dict.fromkeys(R.subset))
    series = [("recording-level", "#2a78d6"), ("non-speech spectrum", "#eb6834"), ("speech MFCC (reference)", "#1baf7a")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, stage in zip(axes, ["raw", "clean"]):
        for j, (fam, colr) in enumerate(series):
            d = R[R.feature_set == f"{stage}: {fam}"].set_index("subset").reindex(subsets)
            if d.auc_lr.notna().sum() == 0:
                continue
            yy = np.arange(len(subsets)) + (j - 1) * 0.24
            ax.errorbar(d.auc_lr, yy, xerr=[d.auc_lr - d.auc_lr_lo, d.auc_lr_hi - d.auc_lr], fmt="o", ms=5,
                        color=colr, ecolor=colr, elinewidth=1.5, capsize=0, label=fam, markeredgecolor=C["surface"])
        ax.axvline(0.5, color=C["ink2"], linewidth=1, linestyle="--")
        ax.text(0.5, -0.9, " chance", color=C["ink2"], fontsize=8)
        ax.set_xlim(0.2, 1.02)
        ax.set_title("Raw recordings" if stage == "raw" else "After harmonisation", loc="left")
        ax.set_xlabel("Subject-level AUC (logistic regression, 95% range over CV repeats)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(subsets)), subsets)
    axes[0].invert_yaxis()
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, fontsize=8)
    fig.suptitle("Shortcut test: predicting PD from recording artefacts only", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig3_shortcut_auc.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
