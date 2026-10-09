# Sensor-data check (MOCHA-TIMIT real EMA vs SPARC estimates)

Speakers: fsew0, maps0, msak0; sentences: 1378.

## Summary measures (per-sentence agreement, within speaker)

| measure | median_spearman | min_spearman | speakers | pooled_spearman | p | n | used_in_step20_findings | passes |
|---|---|---|---|---|---|---|---|---|
| LL_speed | 0.61 | 0.47 | 3 | 0.57 | 0.00 | 1378 | True | True |
| TB_space | 0.56 | 0.45 | 3 | 0.55 | 0.00 | 1378 | True | True |
| LL_range | 0.54 | 0.51 | 3 | 0.55 | 0.00 | 1378 | True | True |
| TT_range | 0.51 | 0.48 | 3 | 0.53 | 0.00 | 1378 | True | True |
| LI_range | 0.50 | 0.40 | 3 | 0.48 | 0.00 | 1378 | True | False |
| TB_speed | 0.47 | 0.47 | 3 | 0.48 | 0.00 | 1378 | True | False |
| TD_speed | 0.43 | 0.36 | 3 | 0.43 | 0.00 | 1378 | True | False |
| TD_range | 0.41 | 0.40 | 3 | 0.44 | 0.00 | 1378 | True | False |
| TT_speed | 0.41 | 0.38 | 3 | 0.43 | 0.00 | 1378 | True | False |
| LI_speed | 0.37 | 0.36 | 3 | 0.39 | 0.00 | 1378 | True | False |
| LA_speed | 0.57 | 0.38 | 3 | 0.52 | 0.00 | 1378 | False | True |
| TB_range | 0.49 | 0.48 | 3 | 0.52 | 0.00 | 1378 | False | False |
| UL_speed | 0.37 | 0.36 | 3 | 0.43 | 0.00 | 1378 | False | False |
| UL_range | 0.34 | 0.09 | 3 | 0.31 | 0.00 | 1378 | False | False |
| LA_range | 0.33 | 0.27 | 3 | 0.36 | 0.00 | 1378 | False | False |

## Frame-level trajectory agreement (median r per speaker x channel)

| speaker | LIX | LIY | LLX | LLY | TBX | TBY | TDX | TDY | TTX | TTY | ULX | ULY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fsew0 | 0.54 | 0.84 | 0.55 | 0.86 | 0.76 | 0.87 | 0.72 | 0.85 | 0.80 | 0.78 | 0.64 | 0.75 |
| maps0 | 0.65 | 0.80 | -0.05 | 0.82 | 0.73 | 0.86 | 0.74 | 0.90 | 0.66 | 0.81 | 0.75 | 0.50 |
| msak0 | 0.74 | 0.83 | 0.79 | 0.87 | 0.81 | 0.82 | 0.78 | 0.85 | 0.85 | 0.79 | 0.62 | 0.84 |

MOCHA-TIMIT: Queen Margaret University College 1999; research/educational use; licence kept with the data.