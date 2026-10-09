# Resynthesis check

recordings: 82

## PD vs healthy (sex-adjusted, SD units)

dataset metric  mean_pd  mean_hc   beta  ci_lo  ci_hi     p
   IPVS mcd_db   39.709   44.530 -1.020 -1.558 -0.482 0.000
   IPVS  mel_r    0.943    0.925  0.790  0.264  1.317 0.003
   MDVR mcd_db   43.003   42.375 -0.002 -0.930  0.925 0.996
   MDVR  mel_r    0.857    0.840  0.667  0.067  1.268 0.029

## Association with severity (MDVR patients)

metric  scale  n    rho     p
mcd_db updrs2 16  0.307 0.248
mcd_db updrs3 16  0.117 0.667
mcd_db     hy 16  0.272 0.308
 mel_r updrs2 16 -0.348 0.187
 mel_r updrs3 16 -0.393 0.133
 mel_r     hy 16 -0.386 0.140