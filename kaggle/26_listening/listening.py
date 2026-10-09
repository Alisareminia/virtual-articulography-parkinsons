"""STEP 26 - Listening companion for "Bradykinesia of the mouth" (virtual articulography in Parkinson's disease).

Sonification of the MEASURED mouth movements (nothing is synthesised from a model of Parkinson's):
  1. Typical speakers, chosen by rule: per language, a sex-matched healthy and Parkinson's reader closest to their group's
     median on the sensor-validated movement measures (step 20 table), and, for tremor, the Italian /a/ of the healthy and
     Parkinson's speaker at their group's median jaw tremor (+ one clearly labelled 'clear example' from the top of the
     Parkinson's tremor distribution).
  2. Their audio goes through the same audio-only inversion network (SPARC, as in steps 19-22) -> 12 articulator channels.
  3. Movement -> sound with ONE fixed scale shared by both speakers of a pair:
       * tongue tip / jaw height -> pitch (a wide sweep = a big movement; a narrow wobble = a small movement);
       * jaw tremor (2-12 Hz band) -> tremolo of a steady tone at its real rate (a rhythmic pulsing = tremor).
Outputs: listening/*.ogg, listening/README.md, listening/clip_index.csv
"""
import os
import subprocess
import sys
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, resample_poly, sosfiltfilt

INPUT = Path(os.environ.get("PDV_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("PDV_OUTPUT", "/kaggle/working")) / "listening"
SR16, SR_OUT, FS_EMA = 16000, 22050, 50.0
TARGET_DBFS = -23.0
EXCERPT_S = 20.0
CHN = ["TDX", "TDY", "TBX", "TBY", "TTX", "TTY", "LIX", "LIY", "ULX", "ULY", "LLX", "LLY"]
CHI = {c: i for i, c in enumerate(CHN)}
VALID = ["TT_range", "TB_space", "LL_range", "LL_speed"]          # sensor-validated (step 21) and used in step 20


# ------------------------------------------------------------------ helpers (as in steps 13-20)
def find_root_with(fname, sibling_dir=None):
    for root, dirs, files in os.walk(INPUT):
        if fname in files and (sibling_dir is None or sibling_dir in dirs):
            return Path(root)
    return None


def find_file(fname):
    for root, _, files in os.walk(INPUT):
        if fname in files:
            return Path(root) / fname
    return None


def resample(x, sr, target):
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


def activity(e):
    p10, p90, p95 = np.percentile(e, [10, 90, 95])
    thr = p95 - 30
    if p90 - p10 > 15:
        thr = max(thr, p10 + 6)
    return e > thr


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


class Inverter:
    def __init__(self, name="multi"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speech-articulatory-coding"], check=False)
        import torch
        from sparc import load_model
        self.coder = load_model(name, device="cuda:0" if torch.cuda.is_available() else "cpu")

    def ema(self, x):
        e = self.coder.encode(np.asarray(x, np.float32))["ema"]
        e = e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e)
        return e.reshape(-1, 12).astype(np.float64)


# ------------------------------------------------------------------ sonification
def gate_50hz(x, L):
    """Speech-activity gate at the EMA frame rate (50 Hz), softened so tones fade in and out."""
    e, _ = frames_db(x, SR16, 0.04, 0.02)
    g = activity(e).astype(float)
    g = np.concatenate([g, np.zeros(max(0, L - len(g)))])[:L]
    return np.clip(uniform_filter1d(g, 7), 0, 1)


def up(v, n_out):
    t_in = np.arange(len(v)) / FS_EMA
    return np.interp(np.arange(n_out) / SR_OUT, t_in, v)


def tone(freq, amp):
    ph = 2 * np.pi * np.cumsum(freq) / SR_OUT
    y = (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)) * amp
    return y / 1.47


def speech_width(y, gate):
    """p5-p95 height during speech frames only (pauses excluded, as in the step-20 measures)."""
    a = y[gate > 0.5]
    return float(np.subtract(*np.percentile(a if len(a) > 20 else y, [95, 5])))


def movement_to_pitch(y, gate, base_hz, octaves_per_unit):
    """Height of an articulator (centred on the speaker's own speech-frame median) -> pitch; fixed scale -> comparable."""
    a = y[gate > 0.5]
    yc = y - np.median(a if len(a) > 20 else y)
    n = int(len(y) / FS_EMA * SR_OUT)
    f = base_hz * 2 ** (octaves_per_unit * up(yc, n))
    return tone(f, up(gate, n))


def tremor_to_tremolo(jaw, gain, base_hz=196.0):
    """Jaw movement in the 2-12 Hz band drives the loudness (and a little the pitch) of a steady tone, at its real rate."""
    bp = sosfiltfilt(butter(4, [2, 12], "bandpass", fs=FS_EMA, output="sos"), jaw - np.mean(jaw))
    n = int(len(jaw) / FS_EMA * SR_OUT)
    m = np.clip(gain * up(bp, n), -0.9, 0.9)
    fade = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.15 * SR_OUT))
    return tone(base_hz * (1 + 0.04 * m), (0.55 + 0.45 * m) * fade), bp


def write(name, y, sr=SR_OUT):
    y = np.asarray(y, float)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    sf.write(OUT / f"{name}.ogg", y.astype(np.float32), sr, format="OGG", subtype="VORBIS")
    return f"{name}.ogg"


def chime(sec=0.9):
    t = np.arange(int(sec * SR_OUT)) / SR_OUT
    return 0.25 * np.sin(2 * np.pi * 880 * t) * np.exp(-6 * t)


def to_out(x):
    return resample(x, SR16, SR_OUT)


# ------------------------------------------------------------------ choose typical speakers
def pick_typical(df, value, label_col="label"):
    out = {}
    for lb in (0, 1):
        g = df[df[label_col] == lb]
        med = g[value].median()
        out[lb] = g.iloc[(g[value] - med).abs().argsort().iloc[0]]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s02 = find_root_with("manifest.csv", "clean_audio")
    s03 = find_root_with("features_subject.csv")
    man = pd.read_csv(s02 / "manifest.csv")
    S3 = pd.read_csv(s03 / "features_subject.csv")
    man["sex"] = man.subject_id.map(S3.drop_duplicates("subject_id").set_index("subject_id").sex).fillna(man.get("sex"))
    RR = pd.read_csv(find_file("reading_recordings.csv"))
    TI = pd.read_csv(find_file("tremor_speakers_IPVS.csv"))

    def load(p):
        x, sr = sf.read(str(s02 / "clean_audio" / p), dtype="float64")
        x = x.mean(axis=1) if x.ndim > 1 else x
        return trim_and_normalise(x if sr == SR16 else resample(x, sr, SR16), SR16)

    inv = Inverter("multi")
    rows = []

    # ---------------- hypokinesia: reading
    for coh, ds, sex, lang in (("MDVR reading", "MDVR", "F", "English"), ("IPVS reading", "IPVS", "M", "Italian")):
        sp = RR[RR.cohort == coh].groupby("sid").agg({**{v: "mean" for v in VALID}, "label": "first", "sex": "first"}).reset_index()
        ref = sp[(sp.label == 0) & (sp.sex == sex)]
        sp["composite_z"] = np.mean([(sp[v] - ref[v].mean()) / ref[v].std() for v in VALID], axis=0)
        typ = pick_typical(sp[sp.sex == sex], "composite_z")
        EM, AU, GT = {}, {}, {}
        for lb, r in typ.items():
            files = man[(man.dataset == ds) & (man.task_family == "read_passage") & (man.subject_id == r.sid)].sort_values("task")
            x = load(files.clean_path.iloc[0])[: int(EXCERPT_S * SR16)]
            E = inv.ema(x)
            EM[lb], AU[lb], GT[lb] = E, x, gate_50hz(x, len(E))
            pct = (sp[(sp.sex == sex) & (sp.label == lb)].composite_z < r.composite_z).mean()
            rows.append(dict(part="movement", language=lang, group="healthy" if lb == 0 else "Parkinson's", sex=sex,
                             speaker=str(r.sid)[-6:], composite_z=round(float(r.composite_z), 2),
                             percentile_in_group=round(100 * float(pct)), **{v: round(float(r[v]), 3) for v in VALID}))
        tag = lang[:2].upper()
        for art, ch, base, nm in (("tongue_tip", "TTY", 330.0, "tongue tip"), ("jaw", "LIY", 220.0, "jaw")):
            h_rng = speech_width(EM[0][:, CHI[ch]], GT[0])
            k = 1.5 / max(h_rng, 1e-6)                                       # healthy p5-p95 = 1.5 octaves; same k for PD
            clips = {}
            for lb in (0, 1):
                y = movement_to_pitch(EM[lb][:, CHI[ch]], GT[lb], base, k)
                g_ = "healthy" if lb == 0 else "parkinsons"
                clips[lb] = y
                rows.append(dict(part="movement", language=lang, group=g_, clip=write(f"{tag}_{art}_as_pitch__{g_}", y),
                                 what=f"{nm} height as pitch (shared scale)",
                                 octaves_p5_p95=round(k * speech_width(EM[lb][:, CHI[ch]], GT[lb]), 2)))
            both = np.concatenate([clips[0], chime(), clips[1]])
            rows.append(dict(part="movement", language=lang, group="healthy then Parkinson's",
                             clip=write(f"{tag}_{art}_as_pitch__COMPARE_healthy_then_parkinsons", both),
                             what=f"{nm}: typical healthy reader, chime, typical Parkinson's reader"))
        for lb in (0, 1):
            g_ = "healthy" if lb == 0 else "parkinsons"
            sp_ = to_out(AU[lb])
            tt = movement_to_pitch(EM[lb][:, CHI["TTY"]], GT[lb], 330.0,
                                   1.5 / max(speech_width(EM[0][:, CHI["TTY"]], GT[0]), 1e-6))
            n = min(len(sp_), len(tt))
            rows.append(dict(part="movement", language=lang, group=g_, clip=write(f"{tag}_reading__{g_}__original", sp_),
                             what="original recording (excerpt)"))
            rows.append(dict(part="movement", language=lang, group=g_,
                             clip=write(f"{tag}_reading__{g_}__voice_plus_tongue_tip", 0.45 * sp_[:n] / (np.max(np.abs(sp_)) + 1e-9)
                                        + 0.55 * tt[:n] / (np.max(np.abs(tt)) + 1e-9)),
                             what="the voice with its tongue-tip movement playing along as pitch"))

    # ---------------- tremor: Italian sustained /a/
    ti = TI[TI.male == 1.0].dropna(subset=["mouth_LI"])
    typ = pick_typical(ti, "mouth_LI")
    clear = ti[ti.label == 1].sort_values("mouth_LI").iloc[int(0.9 * (ti.label == 1).sum())]
    picks = [("healthy", typ[0]), ("parkinsons", typ[1]), ("parkinsons_clear_example", clear)]
    J, AUD = {}, {}
    for nm, r in picks:
        f_ = man[(man.dataset == "IPVS") & man.task_family.astype(str).str.startswith("vowel") & (man.subject_id == r.sid)]
        f_ = f_[f_.task_family.astype(str).str[-1] == "a"].sort_values("task")
        x = load(f_.clean_path.iloc[0])
        E = inv.ema(x)
        a, b = int(0.2 * len(E)), int(0.8 * len(E))                         # central part, as in step 20
        J[nm], AUD[nm] = E[a:b, CHI["LIY"]], x[int(a / FS_EMA * SR16):int(b / FS_EMA * SR16)]
        grp = ti[ti.label == (0 if nm == "healthy" else 1)].mouth_LI
        rows.append(dict(part="tremor", language="Italian", group=nm, sex="M", speaker=str(r.sid)[-6:],
                         jaw_tremor_share_3_7Hz=round(float(r.mouth_LI), 3),
                         percentile_in_group=round(100 * float((grp < r.mouth_LI).mean()))))
    bps = {nm: sosfiltfilt(butter(4, [2, 12], "bandpass", fs=FS_EMA, output="sos"), j - j.mean()) for nm, j in J.items()}
    gain = 0.9 / max(np.percentile(np.abs(np.concatenate([bps["healthy"], bps["parkinsons"]])), 99), 1e-9)   # shared scale
    seq = []
    for nm in ("healthy", "parkinsons", "parkinsons_clear_example"):
        y, _ = tremor_to_tremolo(J[nm], gain)
        seq.append(y)
        rows.append(dict(part="tremor", language="Italian", group=nm, clip=write(f"IT_vowel_a__jaw_tremor_as_tremolo__{nm}", y),
                         what="jaw movement (2-12 Hz) pulsing a steady tone at its real rate (shared scale)"))
        rows.append(dict(part="tremor", language="Italian", group=nm, clip=write(f"IT_vowel_a__{nm}__original", to_out(AUD[nm])),
                         what="original sustained /a/ (central part)"))
    rows.append(dict(part="tremor", language="Italian", group="healthy then Parkinson's",
                     clip=write("IT_vowel_a__jaw_tremor__COMPARE_healthy_then_parkinsons", np.concatenate([seq[0], chime(), seq[1]])),
                     what="typical healthy speaker, chime, typical Parkinson's speaker"))
    idx = pd.DataFrame(rows)
    idx.to_csv(OUT / "clip_index.csv", index=False)
    sel = idx[idx["clip"].isna()].drop(columns=["clip", "what"], errors="ignore").dropna(axis=1, how="all")
    (OUT / "README.md").write_text(f"""# Listening companion - bradykinesia of the mouth

Everything here is the MEASURED movement of real speakers' mouths, estimated from their voices by the same audio-only
network used in the paper, turned into sound with one fixed scale shared by both speakers of each pair.
Nothing is a simulation of Parkinson's disease.

## Who you are hearing (chosen by rule, not by ear)
Typical = the speaker closest to their group's median (reading: mean z of the four sensor-validated movement measures,
relative to healthy speakers of the same sex; tremor: 3-7 Hz jaw tremor share). Same sex within each pair.

```
{sel.to_string(index=False)}
```

## How to listen
1. `*_as_pitch__COMPARE_healthy_then_parkinsons.ogg` - the tongue tip (or jaw) moving up and down is heard as pitch going
   up and down. Healthy first, then a chime, then Parkinson's. Listen for how WIDE the melody is: smaller movements =
   narrower, flatter melody. (`octaves_p5_p95` in clip_index.csv gives the width in octaves.)
2. `*_voice_plus_tongue_tip.ogg` - the original voice with its own tongue tip 'singing along'.
3. `IT_vowel_a__jaw_tremor__COMPARE_healthy_then_parkinsons.ogg` - while the speaker holds /a/, the jaw's small
   movements make a steady tone pulse at their real rate. Tremor = a regular 4-6 pulses per second wobble.
   `..._parkinsons_clear_example` is from the top 10% of the Parkinson's group and is labelled as such.
4. `*__original.ogg` - the recordings themselves.

## Honest notes
* Two speakers per comparison: single speakers vary (paper Fig. 7); the group results are in Figs 3-5.
* The jaw-tremor difference is moderate at the median (AUC 0.80 for the group); the clear example shows what the
  measure picks up when it is strong.
""")
    print(idx.to_string(index=False))


if __name__ == "__main__":
    main()
