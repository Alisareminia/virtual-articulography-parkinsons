"""Robustness checks requested in pre-submission review (runs locally in seconds on the saved result tables).

1. Recording condition (IPVS reading): healthy speakers were all recorded in 2017 at 16 kHz; patients in 2017 at 16 kHz
   or in 2016 at 44.1 kHz.
   a. within patients: do the movement measures differ between the two recording conditions?
   b. matched condition: PD vs healthy using only 2017 / 16-kHz recordings.
2. Composite sensitivity: leave each of the four validated measures out of the composite in turn and recompute the
   English AUC and the correlation with UPDRS II-5 (raw and partialled for speech rate).

Outputs: results/27_reviewer_checks/*.csv and report.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "27_reviewer_checks"
KIN = ["TT_range", "TT_speed", "TB_range", "TB_speed", "TB_space", "TD_range", "TD_speed", "LI_range", "LI_speed",
       "LL_range", "LL_speed", "UL_range", "UL_speed", "LA_range", "LA_speed"]
VALID = ["TT_range", "TB_space", "LL_range", "LL_speed"]


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


def effect(d, col, grp):
    """Sex-adjusted difference (grp=1 minus grp=0) in pooled-SD units, HC3 p value, Hedges g."""
    d = d[[col, grp, "male"]].dropna()
    a, b = d.loc[d[grp] == 1, col], d.loc[d[grp] == 0, col]
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    X = sm.add_constant(d[[grp, "male"]].astype(float))
    f = sm.OLS((d[col] - d[col].mean()) / sp, X).fit(cov_type="HC3")
    g = (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9))
    return dict(measure=col, n1=len(a), n0=len(b), beta=float(f.params[grp]), ci_lo=float(f.conf_int().loc[grp, 0]),
                ci_hi=float(f.conf_int().loc[grp, 1]), p=float(f.pvalues[grp]), g=float(g))


def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    res = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)  # noqa: E731
    return spearmanr(res(rx), res(ry))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "results/02_clean_and_shortcut_tests/manifest.csv")
    rr = pd.read_csv(ROOT / "results/20_mouth_bradykinesia/reading_recordings.csv")
    ip = rr[rr.cohort == "IPVS reading"].merge(man[["subject_id", "task", "orig_sr", "session_year"]],
                                               left_on=["sid", "take"], right_on=["subject_id", "task"], how="left")
    ip["cond"] = np.where(ip.orig_sr == 44100, "2016 / 44.1 kHz", "2017 / 16 kHz")
    spk = ip.groupby(["sid", "cond"]).agg({**{c: "mean" for c in KIN}, "label": "first", "male": "first"}).reset_index()

    # 1a within patients: 2016/44.1k (=1) vs 2017/16k (=0)
    pdc = spk[spk.label == 1].assign(old=lambda d: (d.cond == "2016 / 44.1 kHz").astype(int))
    W = pd.DataFrame([effect(pdc, c, "old") for c in KIN]); W["q"] = bh(W.p)
    W.to_csv(OUT / "within_pd_recording_condition.csv", index=False)

    # 1b matched condition: PD vs healthy, 2017 / 16 kHz only
    m = spk[spk.cond == "2017 / 16 kHz"]
    Mt = pd.DataFrame([effect(m, c, "label") for c in KIN]); Mt["q"] = bh(Mt.p)
    full = pd.read_csv(ROOT / "results/20_mouth_bradykinesia/mouth_bradykinesia_effects.csv")
    full = full[full.section == "B IPVS reading"].set_index("measure")[["beta", "q"]].rename(columns={"beta": "beta_all", "q": "q_all"})
    Mt = Mt.join(full, on="measure")
    Mt.to_csv(OUT / "matched_condition_pd_vs_healthy.csv", index=False)

    # 2 composite leave-one-out
    S = pd.read_csv(ROOT / "results/22_head_to_head/speaker_table_dl_and_classic.csv")
    rows = []
    for drop in [None] + VALID:
        keep = [v for v in VALID if v != drop]
        comp = pd.Series(np.nan, index=S.index)
        for coh, g in S.groupby("cohort"):
            Z = []
            for c in keep:
                z = pd.Series(np.nan, index=g.index)
                for mv in (0.0, 1.0):
                    ref = g[(g.label == 0) & (g.male == mv)][c]
                    if ref.notna().sum() < 5:
                        ref = g[g.label == 0][c]
                    msk = g.male == mv
                    z[msk] = (g.loc[msk, c] - ref.mean()) / (ref.std() or 1)
                Z.append(-z)
            comp[g.index] = pd.concat(Z, axis=1).mean(axis=1)
        S["c"] = comp
        r = dict(composite="all four" if drop is None else f"without {drop}")
        for coh, nm in (("MDVR reading", "EN"), ("IPVS reading", "IT")):
            g = S[S.cohort == coh].dropna(subset=["c"])
            r[f"AUC_{nm}"] = roc_auc_score(g.label, g.c)
        pdm = S[(S.cohort == "MDVR reading") & (S.label == 1)].dropna(subset=["c", "updrs2", "speech_rate_syl_s"])
        r["rho_UPDRS_II5"], r["p_UPDRS_II5"] = spearmanr(pdm.c, pdm.updrs2)
        r["rho_UPDRS_II5_rate_adj"], r["p_UPDRS_II5_rate_adj"] = partial_spearman(pdm.c.to_numpy(), pdm.updrs2.to_numpy(),
                                                                                   pdm.speech_rate_syl_s.to_numpy())
        rows.append(r)
    C = pd.DataFrame(rows); C.to_csv(OUT / "composite_leave_one_out.csv", index=False)

    ns = spk.groupby(["cond", "label"]).size().to_dict()
    rep = ["# Reviewer robustness checks", "",
           f"IPVS reading speakers per condition (cond, label): {ns}", "",
           "## 1a. Within patients: 2016 / 44.1 kHz minus 2017 / 16 kHz (sex-adjusted, SD units)", "",
           W.round(3).to_markdown(index=False), "",
           "## 1b. PD vs healthy, 2017 / 16-kHz recordings only (beta_all = original analysis)", "",
           Mt.round(3).to_markdown(index=False), "",
           "## 2. Composite leave-one-out", "", C.round(3).to_markdown(index=False), ""]
    (OUT / "report.md").write_text("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    main()
