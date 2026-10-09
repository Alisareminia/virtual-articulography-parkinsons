"""PD voice project - step 01: data audit.

Runs on Kaggle. Audits three open raw-audio Parkinson's corpora before any modelling:
  - IPVS      Italian Parkinson's Voice and Speech (Dimauro & Girardi)  [Kaggle mirror]
  - FIGSHARE  Voice Samples for PD and HC, sustained /a/ (Iyer et al.)  [Kaggle mirror]
  - MDVR      Mobile Device Voice Recordings at KCL (Jaeger et al.)     [Zenodo 2867216]

Outputs (small, in /kaggle/working):
  audit_files.csv     one row per audio file (pseudonymised subject IDs, no names)
  audit_subjects.csv  one row per unique speaker
  audit_report.txt    human-readable summary of all checks
"""
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
TMP = Path("/tmp/pdvoice")  # not /kaggle/working, so raw audio never becomes notebook output
TMP.mkdir(parents=True, exist_ok=True)
MDVR_URL = "https://zenodo.org/records/2867216/files/26_29_09_2017_KCL.zip?download=1"


def find_dir(name):
    """Kaggle's mount layout varies; locate a directory by name anywhere under /kaggle/input."""
    for root, dirs, _ in os.walk(INPUT):
        if name in dirs:
            return Path(root) / name
    return None


def find_file(name):
    for root, _, files in os.walk(INPUT):
        if name in files:
            return Path(root) / name
    return None


# ----------------------------------------------------------------------------- signal checks
def signal_stats(path):
    info = sf.info(str(path))
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    raw_md5 = hashlib.md5(x.tobytes()).hexdigest()
    x = x.mean(axis=1)
    n = len(x)
    row = dict(
        sr=sr,
        channels=info.channels,
        subtype=info.subtype,
        duration_s=n / sr if sr else np.nan,
        audio_md5=raw_md5,
    )
    if n < sr * 0.1:
        return row
    peak = float(np.max(np.abs(x)))
    row["peak_dbfs"] = 20 * np.log10(peak + 1e-12)
    row["rms_dbfs"] = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
    row["clip_frac"] = float(np.mean(np.abs(x) >= 0.999))
    row["dc_offset"] = float(np.mean(x))
    # frame energies (25 ms) -> noise floor and a crude SNR
    hop = int(0.025 * sr)
    frames = x[: (n // hop) * hop].reshape(-1, hop)
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    row["noise_floor_db"] = float(np.percentile(e, 10))
    row["speech_level_db"] = float(np.percentile(e, 90))
    row["snr_est_db"] = row["speech_level_db"] - row["noise_floor_db"]
    row["lead_200ms_db"] = float(10 * np.log10(np.mean(x[: int(0.2 * sr)] ** 2) + 1e-12))
    # effective bandwidth: frequency below which 99% of spectral energy lies.
    # Reveals telephone-band or upsampled audio that nominal sample rate hides.
    f, p = welch(x, fs=sr, nperseg=min(4096, n))
    c = np.cumsum(p) / (np.sum(p) + 1e-20)
    row["bw99_hz"] = float(f[np.searchsorted(c, 0.99)])
    row["bw95_hz"] = float(f[np.searchsorted(c, 0.95)])
    return row


# ----------------------------------------------------------------------------- IPVS
IPVS_GROUPS = {
    "15 Young Healthy Control": "YHC",
    "22 Elderly Healthy Control": "EHC",
    "28 People with Parkinsons disease": "PD",
}
# IPVS file names: <TASK><identity letters><age><sex><date+time>
#   2017 sessions: VA1VSIPTIOZ46M240120171924  (DDMMYYYY + HHMM)
#   2016 sessions: VA1vsiptioz46M1606161705    (DDMMYY + HHMM)
IPVS_RE = re.compile(r"^([A-Z]+\d)([A-Za-z]+?)(\d{2})([MF])(\d+)$")


def ipvs_task_family(t):
    if t.startswith("V"):
        return "vowel_" + t[1].lower()
    return {"B": "read_passage", "D": "syllables", "FB": "phrases", "PR": "words"}.get(t.rstrip("0123456789"), "other")


def audit_ipvs():
    root = find_dir("28 People with Parkinsons disease")
    if root is None:
        print("!! IPVS not found under /kaggle/input")
        return pd.DataFrame(), {}
    root = root.parent
    print(f"IPVS root: {root}")
    rows, meta = [], {}
    for gdir, g in IPVS_GROUPS.items():
        for p in sorted((root / gdir).rglob("*")):
            if p.suffix.lower() in (".xlsx", ".xls"):
                meta[str(p.relative_to(root))] = p
            if p.suffix.lower() != ".wav":
                continue
            stem = p.stem.rstrip(".")
            m = IPVS_RE.match(stem)
            rel = p.relative_to(root / gdir)
            folder = str(rel.parent)  # e.g. "17-28/Vito S" or "Alberto R"
            r = dict(dataset="IPVS", group=g, label=int(g == "PD"), folder=folder, file=p.name)
            if m:
                task, code, age, sex, stamp = m.groups()
                r.update(task=task, code=code, age=int(age), sex=sex, stamp=stamp)
                r["id_key"] = "".join(sorted(code.upper())) + age + sex  # robust to letter permutations
                r["code_case"] = "lower" if code.islower() else "upper" if code.isupper() else "mixed"
                if len(stamp) == 12:      # DDMMYYYYHHMM
                    r["session_date"] = f"{stamp[4:8]}-{stamp[2:4]}-{stamp[0:2]}"
                elif len(stamp) == 10:    # DDMMYYHHMM
                    r["session_date"] = f"20{stamp[4:6]}-{stamp[2:4]}-{stamp[0:2]}"
            else:
                r.update(task=re.match(r"^[A-Z]+\d?", stem).group(0) if re.match(r"^[A-Z]+", stem) else "?",
                         id_key="UNPARSED:" + g + ":" + folder)
            r["task_family"] = ipvs_task_family(r["task"])
            r.update(signal_stats(p))
            rows.append(r)
    return pd.DataFrame(rows), meta


# ----------------------------------------------------------------------------- Figshare (Iyer et al.)
def audit_figshare():
    rows = []
    for gdir, g in (("HC_AH", "HC"), ("PD_AH", "PD")):
        d = find_dir(gdir)
        if d is None:
            print(f"!! Figshare {gdir} not found")
            continue
        for p in sorted(d.rglob("*.wav")):
            parts = p.stem.split("_")
            rows.append(dict(dataset="FIGSHARE", group=g, label=int(g == "PD"), file=p.name,
                             id_key=parts[1] if len(parts) > 1 else p.stem,
                             task="A", task_family="vowel_a", **signal_stats(p)))
    demo = find_file("Demographics_age_sex.xlsx")
    return pd.DataFrame(rows), demo


# ----------------------------------------------------------------------------- MDVR-KCL
MDVR_RE = re.compile(r"^(ID\d+)_(hc|pd)_(\d+)_(\d+)_(\d+)$", re.I)


def audit_mdvr():
    z = TMP / "mdvr.zip"
    if not z.exists():
        print("Downloading MDVR-KCL from Zenodo (on the Kaggle machine) ...")
        subprocess.run(["wget", "-q", "-O", str(z), MDVR_URL], check=True)
    ext = TMP / "mdvr"
    if not ext.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ext)
    rows = []
    for p in sorted(ext.rglob("*.wav")):
        m = MDVR_RE.match(p.stem)
        task = "read_passage" if "read" in str(p).lower() else "dialogue" if "dialog" in str(p).lower() else "?"
        r = dict(dataset="MDVR", file=p.name, task=task, task_family=task)
        if m:
            sid, g, hy, u2, u3 = m.groups()
            r.update(group=g.upper(), label=int(g.lower() == "pd"), id_key=sid.upper(),
                     hoehn_yahr=int(hy), updrs_ii5=int(u2), updrs_iii18=int(u3))
        else:
            r.update(group="?", id_key="UNPARSED:" + p.stem)
        r.update(signal_stats(p))
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report
def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def describe_xlsx(path, label):
    """Print structure of a metadata sheet without printing personal names."""
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
    except Exception as e:  # noqa: BLE001
        print(f"  {label}: unreadable ({e})")
        return
    for sn, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        print(f"  [{label} :: sheet '{sn}'] shape={df.shape}")
        # show first rows, masking cells that look like personal names
        head = df.head(4).copy().astype(str)
        head = head.map(lambda v: "<name?>" if re.fullmatch(r"[A-Z][a-zà-ù]+( [A-Z][a-zà-ù]*\.?)+", v) else v[:18])
        print(head.to_string(index=False, header=False))


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
    (OUT / "audit_report.txt").write_text(buf.getvalue())


def run():
    section("INPUTS")
    for p in sorted(INPUT.iterdir()) if INPUT.exists() else []:
        print(" ", p)

    ipvs, ipvs_meta = audit_ipvs()
    fig, fig_demo = audit_figshare()
    try:
        mdvr = audit_mdvr()
    except Exception as e:  # noqa: BLE001
        print(f"!! MDVR-KCL failed: {e}")
        mdvr = pd.DataFrame()

    # ---------------------------------------------------------------- IPVS identity audit
    section("IPVS - speaker identity audit (same person under several folders?)")
    if len(ipvs):
        print(f"wav files: {len(ipvs)}   unparsed names: {ipvs['id_key'].str.startswith('UNPARSED').sum()}")
        fk = ipvs.groupby(["group", "folder"])["id_key"].agg(lambda s: sorted(set(s)))
        multi_key_folders = fk[fk.map(len) > 1]
        print(f"folders whose files carry >1 identity code: {len(multi_key_folders)}")
        for (g, f), keys in multi_key_folders.items():
            print(f"   {g} | folder#{hashlib.sha1(f.encode()).hexdigest()[:6]} | keys={keys}")
        kf = ipvs.groupby("id_key").agg(groups=("group", lambda s: sorted(set(s))),
                                        n_folders=("folder", "nunique"),
                                        sessions=("session_date", lambda s: sorted(set(s.dropna()))),
                                        n_files=("file", "size"))
        dup = kf[kf.n_folders > 1]
        print(f"\nidentity codes spread over >1 folder (same person, multiple sessions): {len(dup)}")
        print(dup.to_string())
        cross = kf[kf.groups.map(len) > 1]
        print(f"\nidentity codes appearing in >1 GROUP (would be a labelling error): {len(cross)}")
        print(cross.to_string() if len(cross) else "   none")
        print("\nfolders vs unique speakers per group:")
        print(ipvs.groupby("group").agg(folders=("folder", "nunique"), unique_speakers=("id_key", "nunique"),
                                        files=("file", "size")).to_string())

        section("IPVS - recording-condition confound check (does session/format track diagnosis?)")
        ipvs["year"] = ipvs["session_date"].str[:4]
        print("sessions by year x group (files):")
        print(pd.crosstab(ipvs["year"].fillna("?"), ipvs["group"]).to_string())
        print("\nsample rate x group (files):")
        print(pd.crosstab(ipvs["sr"], ipvs["group"]).to_string())
        print("\nsubtype x group:")
        print(pd.crosstab(ipvs["subtype"], ipvs["group"]).to_string())
        print("\nidentity-code case (lower=2016 naming) x group:")
        print(pd.crosstab(ipvs["code_case"].fillna("?"), ipvs["group"]).to_string())
        num = ["duration_s", "rms_dbfs", "noise_floor_db", "lead_200ms_db", "snr_est_db", "bw95_hz", "bw99_hz", "clip_frac"]
        print("\nsignal properties by group (median) - large gaps here = recording bias risk:")
        print(ipvs.groupby("group")[num].median().round(2).to_string())
        print("\nsame, by year:")
        print(ipvs.groupby("year")[num].median().round(2).to_string())

        section("IPVS - task inventory")
        print(pd.crosstab(ipvs["task"], ipvs["group"]).to_string())
        print("\nduration (s) by task family:")
        print(ipvs.groupby("task_family")["duration_s"].describe()[["count", "min", "50%", "max"]].round(2).to_string())

        section("IPVS - demographics from file codes (one row per unique speaker)")
        spk = ipvs.drop_duplicates("id_key")
        print(spk.groupby("group").agg(n=("id_key", "size"), age_mean=("age", "mean"), age_sd=("age", "std"),
                                       age_min=("age", "min"), age_max=("age", "max"),
                                       male=("sex", lambda s: (s == "M").sum()),
                                       female=("sex", lambda s: (s == "F").sum())).round(1).to_string())

        section("IPVS - metadata spreadsheets (structure only; names masked)")
        for k, p in ipvs_meta.items():
            describe_xlsx(p, k)

    # ---------------------------------------------------------------- Figshare
    section("FIGSHARE (Iyer et al.) - sustained /a/, phone recordings")
    if len(fig):
        print(fig.groupby("group").agg(files=("file", "size"), speakers=("id_key", "nunique")).to_string())
        num = ["duration_s", "sr", "rms_dbfs", "noise_floor_db", "snr_est_db", "bw95_hz", "bw99_hz", "clip_frac"]
        print("\nsignal properties by group (median):")
        print(fig.groupby("group")[num].median().round(2).to_string())
        print("\nsample rate x group:")
        print(pd.crosstab(fig["sr"], fig["group"]).to_string())
        if fig_demo is not None:
            try:
                demo = pd.read_excel(fig_demo)
                print(f"\ndemographics sheet: shape={demo.shape} columns={list(demo.columns)}")
                print(demo.head(3).to_string(index=False))
                idcol = demo.columns[0]
                ids = set(demo[idcol].astype(str).str.strip())
                print(f"audio ids matched in demographics: {fig['id_key'].isin(ids).sum()}/{len(fig)}")
                print("\ndemographics by label column:")
                print(demo.groupby(demo.columns[-1]).describe(include="all").T.head(30).to_string())
            except Exception as e:  # noqa: BLE001
                print(f"demographics unreadable: {e}")

    # ---------------------------------------------------------------- MDVR
    section("MDVR-KCL - smartphone reading + dialogue")
    if len(mdvr):
        print(pd.crosstab(mdvr["task"], mdvr["group"]).to_string())
        print("\nunique speakers by group:")
        print(mdvr.groupby("group")["id_key"].nunique().to_string())
        spk = mdvr.drop_duplicates("id_key")
        print("\nclinical scores (per speaker) by group:")
        print(spk.groupby("group")[["hoehn_yahr", "updrs_ii5", "updrs_iii18"]].describe().round(2).T.to_string())
        num = ["duration_s", "sr", "rms_dbfs", "noise_floor_db", "snr_est_db", "bw95_hz", "bw99_hz", "clip_frac"]
        print("\nsignal properties by task x group (median):")
        print(mdvr.groupby(["task", "group"])[num].median().round(2).to_string())
        unp = mdvr["id_key"].str.startswith("UNPARSED").sum()
        print(f"\nunparsed file names: {unp}")
        if unp:
            print(mdvr.loc[mdvr.id_key.str.startswith("UNPARSED"), "file"].head(10).to_string())

    # ---------------------------------------------------------------- cross-dataset
    allf = pd.concat([ipvs, fig, mdvr], ignore_index=True)
    section("EXACT DUPLICATE AUDIO (identical samples) - within and across datasets")
    d = allf[allf.duplicated("audio_md5", keep=False)].sort_values("audio_md5")
    print(f"files involved: {len(d)}")
    if len(d):
        print(d[["audio_md5", "dataset", "group", "task", "file"]].to_string(index=False))

    section("OVERALL")
    print(allf.groupby(["dataset", "group"]).agg(files=("file", "size"), speakers=("id_key", "nunique"),
                                                 median_dur=("duration_s", "median"),
                                                 sr=("sr", lambda s: sorted(set(s)))).to_string())

    # ---------------------------------------------------------------- save (pseudonymised)
    allf["subject_id"] = allf["dataset"] + "_" + allf["group"].astype(str) + "_" + \
        allf["id_key"].map(lambda k: hashlib.sha1(str(k).encode()).hexdigest()[:8])
    keep = [c for c in allf.columns if c not in ("folder", "code", "id_key")]
    allf[keep].to_csv(OUT / "audit_files.csv", index=False)
    subj_cols = [c for c in ["dataset", "group", "label", "age", "sex", "hoehn_yahr", "updrs_ii5", "updrs_iii18"] if c in allf]
    agg = {c: (c, "first") for c in subj_cols}
    agg["n_files"] = ("file", "size")
    if "session_date" in allf:
        agg["n_sessions"] = ("session_date", "nunique")
    subj = allf.groupby("subject_id").agg(**agg).reset_index()
    subj.to_csv(OUT / "audit_subjects.csv", index=False)
    print(f"\nsaved: audit_files.csv ({len(allf)} rows), audit_subjects.csv ({len(subj)} rows)")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
