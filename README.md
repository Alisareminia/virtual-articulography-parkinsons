# Virtual articulography in Parkinson's disease

Code and derived results for the study *"The mouth moves less: virtual articulography from ordinary voice recordings in
Parkinson's disease"*.

A pretrained audio-only acoustic-to-articulatory inversion network
([SPARC](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding); Cho et al., 2024,
[doi:10.1109/JSTSP.2024.3497655](https://doi.org/10.1109/JSTSP.2024.3497655)) is applied to ordinary voice recordings of
147 speakers (81 with Parkinson's disease) from three public datasets. The recovered tongue, jaw and lip trajectories are
validated against real articulograph sensors (MOCHA-TIMIT) and compared between groups, with clinical speech ratings, for
3–7 Hz tremor and against 18 classic acoustic features.

## Repository layout

| Path | Content |
|---|---|
| `kaggle/NN_*/` | One folder per analysis step: the script or notebook and its `kernel-metadata.json` (inputs, GPU/CPU) |
| `results/NN_*/` | Outputs of steps 19–24: per-speaker and per-recording tables (CSV), statistics, reports |
| `paper/figures_v5/` | Final figures (PDF and 300-dpi PNG) |
| `paper/tables/` | Summary tables (CSV) |
| `tools/build_figures_v5.py`, `tools/figure_v5_cells.py` | Build the visualisation-only figure notebook from the results |
| `tools/pull_outputs.py` | Downloads the small outputs of a finished Kaggle kernel |

Speaker identifiers in all tables are hashed. No audio is included (see *What is not included*).

## Pipeline

All analyses ran as Kaggle kernels; each step reads the outputs of earlier steps through `kernel_sources`.

| Step | File | Hardware | Inputs | What it does |
|---|---|---|---|---|
| 01 | `kaggle/01_data_audit/audit.py` | CPU | IPVS, Figshare, MDVR-KCL (downloaded from Zenodo) | Speaker identity and recording-condition audit |
| 02 | `kaggle/02_clean_and_shortcut_tests/clean.py` | CPU | IPVS, Figshare, MDVR-KCL | Cleaned 16-kHz audio and the recording manifest |
| 03 | `kaggle/03_features_and_meta_analysis/features.py` | CPU | step 02 | Speaker metadata (sex estimated from F0 for MDVR-KCL) |
| 19 | `kaggle/19_virtual_articulography_pilot/virtual_articulography_pilot.py` | GPU | 02, 03 | Vowel-posture validity of the inversion |
| 20 | `kaggle/20_mouth_bradykinesia/mouth_bradykinesia.py` | GPU | 02, 03 | Kinematics, group effects, severity, tremor, reliability |
| 21 | `kaggle/21_sensor_check/sensor_check.py` | GPU | MOCHA-TIMIT (downloaded) | Agreement with real articulograph sensors |
| 22 | `kaggle/22_head_to_head/head_to_head.py` | CPU | 02, 20 | Comparison with classic acoustic features |
| 24 | `kaggle/24_figure_assets/figure_assets.py` | GPU | 02, 03, 20, 22 | Signal assets for the figures |
| 25 | `kaggle/25_figures_v5/figures_v5.ipynb` | CPU | 19, 20, 21, 22, 24 | All figures (drawing only, about 2 minutes) |
| 26 | `kaggle/26_listening/listening.py` | GPU | 02, 03, 20 | Sonified movements (supplementary audio) |
| 27 | `kaggle/27_resynthesis_check/resynthesis_check.py` | GPU | 02, 03 | Resynthesis fidelity in patients vs controls (indirect check of the inversion in PD) |
| — | `tools/reviewer_checks.py` (local, seconds) | CPU | results of 02, 20, 22 | Recording-condition and composite leave-one-out robustness checks → `results/27_reviewer_checks/` |

### Re-running on Kaggle

1. Install the Kaggle CLI and place **your own** API token in `~/.kaggle/` (never commit it).
2. In every `kaggle/NN_*/kernel-metadata.json`, replace the owner `alisaremi` in `id` and in `kernel_sources` with your
   Kaggle username.
3. Add the two input datasets to your account (see *Data*), then push and run the steps in the order of the table:

   ```bash
   kaggle kernels push -p kaggle/01_data_audit
   ```

   Wait for each kernel to finish before pushing a kernel that lists it in `kernel_sources`.
4. Download small outputs (tables, figures) of a finished kernel:

   ```bash
   python tools/pull_outputs.py <your-username>/pd-voice-20-mouth-bradykinesia results/20_mouth_bradykinesia
   ```

### Rebuilding the figures locally from the included results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/build_figures_v5.py
cd paper && jupyter nbconvert --to notebook --execute --inplace --allow-errors figures_v5.ipynb
```

Figures are written to `paper/figures_v5/`. Figures 1, 2, S2, S4 and the graphical abstract also need
`results/24_figure_assets/example_reading.npz` and `mocha_example.npz`, which contain audio and sensor excerpts and are not
redistributed; regenerate them by running step 24 on Kaggle. All other figures build from the files in this repository.

## Data

| Dataset | Content | Source |
|---|---|---|
| MDVR-KCL | English reading, smartphone; H&Y, UPDRS II-5, III-18 | Jaeger et al., Zenodo, [doi:10.5281/zenodo.2867216](https://doi.org/10.5281/zenodo.2867216) |
| IPVS | Italian reading and sustained vowels, microphone | Dimauro et al., 2017, [doi:10.1109/ACCESS.2017.2762475](https://doi.org/10.1109/ACCESS.2017.2762475); [IEEE DataPort](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech); Kaggle copy used: `anshsavla/italian-parkinsons-speech` |
| Figshare voice samples | Sustained /a/, telephone | Iyer et al., 2023, [doi:10.1038/s41598-023-47568-w](https://doi.org/10.1038/s41598-023-47568-w); Kaggle copy used: `drishyatomar/voice-samples-for-parkinsons-and-healthy-controls` |
| MOCHA-TIMIT | Audio with articulograph sensors (validation only) | Wrench, 2000; [CSTR, University of Edinburgh](https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html) (research and educational use) |

Please follow the licence terms of each dataset.

## What is not included

- Raw recordings and any audio derived from them, including the supplementary sonification clips (participants' voices).
- `example_reading.npz` (MDVR-KCL audio excerpts) and `mocha_example.npz` (MOCHA-TIMIT audio and sensor data).
- Kernel logs, which can echo raw dataset file names.
- The manuscript.
