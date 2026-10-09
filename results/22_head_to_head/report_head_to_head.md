# Head-to-head: deep-learning articulography vs classic acoustics

## Verdict

| cohort | auc_all_classic | auc_dl_composite | auc_classic_plus_dl | delta | lr_p | unique_share |
|---|---|---|---|---|---|---|
| IPVS reading | 0.98 | 0.68 | 0.98 | -0.00 | 0.98 | 0.67 |
| MDVR reading | 0.84 | 0.83 | 0.87 | 0.03 | 0.00 | 0.50 |

## Cross-validated AUC

| cohort | feature_set | n_features | auc | auc_lo | auc_hi |
|---|---|---|---|---|---|
| IPVS reading | classic timing/prosody | 5 | 0.86 | 0.83 | 0.89 |
| IPVS reading | classic voice quality | 5 | 0.98 | 0.97 | 0.99 |
| IPVS reading | classic articulation (incl. formant movement) | 8 | 0.85 | 0.82 | 0.88 |
| IPVS reading | all classic | 18 | 0.98 | 0.97 | 0.99 |
| IPVS reading | deep learning (validated composite) | 1 | 0.68 | 0.61 | 0.73 |
| IPVS reading | deep learning (all 15) | 15 | 0.83 | 0.75 | 0.89 |
| IPVS reading | all classic + DL composite | 19 | 0.98 | 0.97 | 0.99 |
| IPVS reading | all classic + DL all | 33 | 0.96 | 0.95 | 0.99 |
| MDVR reading | classic timing/prosody | 5 | 0.77 | 0.72 | 0.80 |
| MDVR reading | classic voice quality | 5 | 0.86 | 0.81 | 0.91 |
| MDVR reading | classic articulation (incl. formant movement) | 8 | 0.83 | 0.76 | 0.88 |
| MDVR reading | all classic | 18 | 0.84 | 0.78 | 0.91 |
| MDVR reading | deep learning (validated composite) | 1 | 0.83 | 0.79 | 0.86 |
| MDVR reading | deep learning (all 15) | 15 | 0.76 | 0.70 | 0.81 |
| MDVR reading | all classic + DL composite | 19 | 0.87 | 0.80 | 0.93 |
| MDVR reading | all classic + DL all | 33 | 0.89 | 0.83 | 0.94 |

| cohort | comparison | delta_auc | share_repeats_better |
|---|---|---|---|
| IPVS reading | all classic + DL composite  vs  all classic | -0.00 | 0.26 |
| IPVS reading | all classic + DL all  vs  all classic | -0.02 | 0.04 |
| IPVS reading | deep learning (all 15)  vs  classic articulation (incl. formant movement) | -0.02 | 0.36 |
| MDVR reading | all classic + DL composite  vs  all classic | 0.03 | 0.96 |
| MDVR reading | all classic + DL all  vs  all classic | 0.05 | 0.94 |
| MDVR reading | deep learning (all 15)  vs  classic articulation (incl. formant movement) | -0.06 | 0.04 |

## Likelihood-ratio test

| cohort | best_classic | n | lr_chi2 | p | or_per_sd |
|---|---|---|---|---|---|
| IPVS reading | hammarberg_db, h1h2_db, hnr_db | 45 | 0.00 | 0.98 | 0.76 |
| MDVR reading | F2_speed, h1h2_db, hammarberg_db | 37 | 9.59 | 0.00 | 27.49 |

## Severity (MDVR)

| scale | measure | kind | n | rho | p | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| hy | dl_composite | deep learning | 16 | 0.81 | 0.00 | nan | nan |
| hy | TB_space | deep learning | 16 | -0.78 | 0.00 | nan | nan |
| hy | TT_range | deep learning | 16 | -0.61 | 0.01 | nan | nan |
| hy | speech_rate_syl_s | classic | 16 | -0.83 | 0.00 | nan | nan |
| hy | pause_ratio | classic | 16 | 0.57 | 0.02 | nan | nan |
| hy | f0_sd_st | classic | 16 | -0.34 | 0.20 | nan | nan |
| hy | F1F2_space | classic | 16 | -0.11 | 0.69 | nan | nan |
| hy | F2_speed | classic | 16 | 0.16 | 0.55 | nan | nan |
| hy | cons_centroid_hz | classic | 16 | -0.63 | 0.01 | nan | nan |
| hy | dl_composite | controlling speech_rate_syl_s | partial | 16 | 0.11 | 0.70 | nan | nan |
| hy | dl_composite | controlling F1F2_space | partial | 16 | 0.74 | 0.00 | nan | nan |
| hy | |rho DL composite| - |rho speech rate| | bootstrap difference | 16 | -0.01 | nan | -0.19 | 0.13 |
| updrs3 | dl_composite | deep learning | 16 | 0.75 | 0.00 | nan | nan |
| updrs3 | TB_space | deep learning | 16 | -0.81 | 0.00 | nan | nan |
| updrs3 | TT_range | deep learning | 16 | -0.69 | 0.00 | nan | nan |
| updrs3 | speech_rate_syl_s | classic | 16 | -0.76 | 0.00 | nan | nan |
| updrs3 | pause_ratio | classic | 16 | 0.49 | 0.05 | nan | nan |
| updrs3 | f0_sd_st | classic | 16 | -0.37 | 0.16 | nan | nan |
| updrs3 | F1F2_space | classic | 16 | 0.04 | 0.89 | nan | nan |
| updrs3 | F2_speed | classic | 16 | 0.39 | 0.14 | nan | nan |
| updrs3 | cons_centroid_hz | classic | 16 | -0.57 | 0.02 | nan | nan |
| updrs3 | dl_composite | controlling speech_rate_syl_s | partial | 16 | 0.21 | 0.44 | nan | nan |
| updrs3 | dl_composite | controlling F1F2_space | partial | 16 | 0.78 | 0.00 | nan | nan |
| updrs3 | |rho DL composite| - |rho speech rate| | bootstrap difference | 16 | -0.00 | nan | -0.21 | 0.17 |
| updrs2 | dl_composite | deep learning | 16 | 0.88 | 0.00 | nan | nan |
| updrs2 | TB_space | deep learning | 16 | -0.87 | 0.00 | nan | nan |
| updrs2 | TT_range | deep learning | 16 | -0.71 | 0.00 | nan | nan |
| updrs2 | speech_rate_syl_s | classic | 16 | -0.75 | 0.00 | nan | nan |
| updrs2 | pause_ratio | classic | 16 | 0.33 | 0.21 | nan | nan |
| updrs2 | f0_sd_st | classic | 16 | -0.19 | 0.48 | nan | nan |
| updrs2 | F1F2_space | classic | 16 | 0.04 | 0.87 | nan | nan |
| updrs2 | F2_speed | classic | 16 | 0.37 | 0.16 | nan | nan |
| updrs2 | cons_centroid_hz | classic | 16 | -0.54 | 0.03 | nan | nan |
| updrs2 | dl_composite | controlling speech_rate_syl_s | partial | 16 | 0.79 | 0.00 | nan | nan |
| updrs2 | dl_composite | controlling F1F2_space | partial | 16 | 0.89 | 0.00 | nan | nan |
| updrs2 | |rho DL composite| - |rho speech rate| | bootstrap difference | 16 | 0.14 | nan | 0.01 | 0.34 |

## Unique information

| cohort | n | cv_r2_classic_predicts_dl | unique_share |
|---|---|---|---|
| IPVS reading | 45 | 0.33 | 0.67 |
| MDVR reading | 37 | 0.50 | 0.50 |

## Formant stand-in vs DL

| cohort | movement | classic_measure | classic_beta | classic_p | dl_measure | dl_beta | dl_p | rho_classic_vs_dl |
|---|---|---|---|---|---|---|---|---|
| IPVS reading | jaw opening | F1_range | 0.28 | 0.33 | LI_range | -0.96 | 0.01 | 0.38 |
| IPVS reading | jaw speed | F1_speed | 0.01 | 0.97 | LI_speed | -1.19 | 0.00 | 0.22 |
| IPVS reading | tongue front-back | F2_range | 0.11 | 0.66 | TB_range | -0.34 | 0.26 | 0.25 |
| IPVS reading | tongue speed | F2_speed | -0.59 | 0.05 | TB_speed | -0.55 | 0.10 | -0.05 |
| IPVS reading | tongue working space | F1F2_space | 0.56 | 0.04 | TB_space | -0.37 | 0.23 | 0.16 |
| MDVR reading | jaw opening | F1_range | 0.57 | 0.06 | LI_range | -0.72 | 0.10 | 0.27 |
| MDVR reading | jaw speed | F1_speed | 0.88 | 0.01 | LI_speed | -0.70 | 0.07 | 0.23 |
| MDVR reading | tongue front-back | F2_range | -0.04 | 0.93 | TB_range | -1.02 | 0.00 | 0.15 |
| MDVR reading | tongue speed | F2_speed | 1.18 | 0.01 | TB_speed | -0.79 | 0.04 | -0.18 |
| MDVR reading | tongue working space | F1F2_space | 0.87 | 0.01 | TB_space | -1.27 | 0.00 | -0.26 |