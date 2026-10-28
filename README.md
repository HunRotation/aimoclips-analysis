# AImoclips analysis

Code and de-identified analysis inputs for the manuscript “AImoclips: Benchmarking emotion conveyance in text-to-music generation using a dimensional valence-arousal framework.”

## Reproduce the numerical results

Use **Python 3.10.12** and the exact package versions in [requirements.txt](requirements.txt). The complete pipeline has been verified on Linux with these versions and runs on a CPU. No GPU, audio files, model checkpoints, or additional dataset downloads are needed for the statistical analyses below.

From a terminal on Linux or macOS:

```sh
git clone https://github.com/HunRotation/aimoclips-analysis.git
cd aimoclips-analysis
python --version  # should report Python 3.10.12
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1` instead. Installation requires internet access to download dependencies. Once installed, the analyses use only the included files. Record `git rev-parse HEAD` to identify the code version used.

The runner executes the five stages listed below in order, limits numerical-library threads to one, and stops if a stage fails. Allow several minutes for model fitting and resampling. Wait for `Complete. Numerical outputs are in results/.` before checking the outputs. Rerunning overwrites outputs with the same filenames. Generated results and the virtual environment are excluded from Git.

### Included inputs

All inputs are in [data/](data/). The released dataset is available from the [AImoclips repository](https://github.com/HunRotation/AImoclips) and [Zenodo dataset record](https://doi.org/10.5281/zenodo.23219643).

| Input | Contents |
| --- | --- |
| [ratings_metadata.csv](data/ratings_metadata.csv) | 6,162 valence/arousal ratings from 111 de-identified participants on 991 clips. |
| [clips_metadata.csv](data/clips_metadata.csv) | Metadata and released, rounded mean ratings for 991 clips from six systems and 12 emotion intents. |
| [emotion_intents_metadata.csv](data/emotion_intents_metadata.csv) | Valence/arousal reference coordinates and quadrants for the 12 emotion words. |
| [clap_scores.csv](data/clap_scores.csv) | Saved text–audio alignment scores from Contrastive Language–Audio Pretraining (CLAP), one per clip. |
| [distribution_metrics.csv](data/distribution_metrics.csv) | Saved Fréchet Audio Distance (FAD) values, one per system. |
| [provenance.json](data/provenance.json) | Input provenance, SHA-256 checksums, and the scope of the saved metric inputs. |

The scripts preserve the released rounded clip means for analyses that originally used them. Descriptive quadrant classification and metric associations reconstruct exact means from individual ratings. Preserve this distinction when comparing outputs.

### Analysis stages

| File | Role and analyses |
| --- | --- |
| [run_analysis.py](run_analysis.py) | Runs all five analysis stages in order and saves numerical outputs to `results/`. |
| [descriptive_statistics.py](analysis/descriptive_statistics.py) | Summarizes ratings and clip means, quadrant accuracy/F1 and confusion counts; tests overall accuracy against chance and performs Tukey HSD comparisons. |
| [core_statistics.py](analysis/core_statistics.py) | Estimates inter-rater agreement and variance components; runs ANOVA and robustness checks, centralization sign-flip tests, accuracy/F1 intervals, binomial chance tests, and pairwise permutation tests (S12). |
| [system_deviation_tests.py](analysis/system_deviation_tests.py) | Tests deviations from emotion references with participant- and clip-clustered uncertainty (S8), and calculates system-by-emotion factorial effect sizes (S9). |
| [distance_wald_tests.py](analysis/distance_wald_tests.py) | Tests intended valence/arousal effects on distance from emotion references using HC3-robust Wald tests and partial eta-squared (S4). |
| [metric_associations.py](analysis/metric_associations.py) | Computes CLAP correlations with crossed-bootstrap intervals, emotion-adjusted CLAP regressions, and FAD permutation correlations (Figure 4; S10 summaries and S11 regression coefficients). |
| [metric_common.py](analysis/metric_common.py) | Provides shared input validation, exact clip means, regression weights, and cluster-covariance calculations. |

Running an individual script uses its default settings, but earlier stages must have created its inputs: `core_statistics.py` reads `results/clip_quadrants.csv`, and `system_deviation_tests.py` reads `results/core/anova_robustness.csv`. Use the complete runner for reproduction.

### Outputs and checks

The package writes numerical CSV files and method metadata under `results/`:

| Location | Key outputs |
| --- | --- |
| `results/` | `overall_descriptives.csv`, `categorical_summary.csv`, `confusion_counts.csv`, `figure_deviation_summary.csv`, and `tukey_comparisons.csv`. |
| `results/core/` | `reliability_summary.csv` (inter-rater agreement and variance components), `anova_robustness.csv`, `centralization_robustness.csv`, `accuracy_uncertainty.csv`, `f1_uncertainty.csv`, and `accuracy_pairwise_permutation.csv`. |
| `results/system_deviations/` | `zero_deviation_tests.csv`, `omnibus_effect_sizes.csv`, and `method_metadata.json`. |
| `results/distance/` | `factorial_wald_tests.csv`, `cell_moments.csv`, and `method_metadata.json`. |
| `results/metrics/` | `overall_associations.csv`, `clap_predictor_tests.csv`, `fad_correlations.csv`, `clap_full_coefficients.csv`, `clap_model_fit.csv`, covariance matrices, bootstrap draws, and `method_metadata.json`. |

After a successful run, this check should pass:

```sh
python - <<'PY'
import pandas as pd

overall = pd.read_csv("results/overall_descriptives.csv").iloc[0]
assert (overall.n_clips, overall.n_ratings, overall.n_participants) == (991, 6162, 111)
summary = pd.read_csv("results/categorical_summary.csv")
pooled = summary.loc[summary.model == "Overall"].iloc[0]
assert pooled.correct == 406
assert abs(pooled.accuracy - 406 / 991) < 1e-12
print("Checks passed: 991 clips, 6,162 ratings, 111 participants; accuracy 40.97%.")
PY
```

In PowerShell, save the Python block as a `.py` file and run it with `python`. The scripts also validate input joins, sample counts, and selected independent statistical calculations. Method metadata records input hashes and analysis specifications for the distance, system-deviation, and metric stages.

Use the defaults for the reported resampling results. The core stage uses 2,000 bootstrap repetitions, 20,000 pairwise permutations, and random seed `20260910`. The metric stage uses 20,000 crossed-bootstrap repetitions with random state `20261005` and all 720 permutations of the six systems for FAD correlations. Changing these settings changes resampling-based outputs. Small floating-point differences can occur across numerical-library builds or operating systems.

### Scope of reproduction

This package reproduces the statistical analyses from the included ratings, references, and **saved CLAP/FAD values**. It produces numerical inputs for the manuscript tables and figures, including Figure 4, rather than rendering the figures or compiling the manuscript.

It does not regenerate music clips, recompute audio embeddings or metric scores, or render publication figures. The historical FAD reference-track selection has now been recovered, as recorded in [provenance.json](data/provenance.json). The exact historical checkpoint use and random audio-crop positions remain unverified, so the saved FAD values continue to support the downstream statistical analyses without a claim of exact audio-to-FAD reproduction.

### Recovered FAD reference tracks and model downloads

The [2,000-track reference manifest](reports/analysis/aimoclips_metric_recovery/20261028_reference_track_manifest.csv) includes identifiers, original source URLs, file sizes, and SHA-256 checksums. Sampling the retained metadata with `random_state=42` exactly matches all 2,000 retained reference files. The original parser skipped multi-tag rows, so the eligible sampling pool was 5,107 tracks. This behavior is preserved for recovery and explained in the [recovery report](reports/analysis/aimoclips_metric_recovery/20261028_metric_recovery.md).

The [recovery instructions](reports/analysis/aimoclips_metric_recovery/20261028_recovery_instructions.md) describe manifest extraction, FAD calculation, and separate per-clip CLAP scoring. They also explain the distinction between the intended music checkpoint and the default model used in retained failed-run logs. Seed 42 fixes track selection; no historical seed or offsets were recovered for CLAP's random reference-audio crops.

Use the separate [download helper](reports/analysis/aimoclips_metric_recovery/scripts/download_metric_assets.py) to plan or explicitly download the reference audio and a selected CLAP checkpoint. Checkpoint URLs pin a repository revision, and downloaded files are verified against SHA-256 values. Downloads require `--download`. Reference audio and model weights are not bundled. The helpers and recovered source records are separate from the validated statistical runner; model inference and exact historical metric reproduction have not been validated.

## License

Except where otherwise noted, the author-created code, documentation, and included analysis data in this repository are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/), matching the AImoclips dataset. See [LICENSE](LICENSE) for the full license text.

You may copy, redistribute, and adapt the licensed material, including for commercial purposes, provided that you give appropriate credit, link to the license, and indicate any changes. For attribution, identify the AImoclips analysis authors and link to this repository. When reporting research results, also identify the commit used and cite the AImoclips dataset. Third-party software and any separately identified third-party material retain their own licenses.

The recovered MTG-Jamendo metadata snapshot and metadata fields in the reference manifest retain **CC BY-NC-SA 4.0**, separate from the repository's general CC BY 4.0 license. Reference audio retains its individual upstream licenses. See the [third-party notices](reports/analysis/aimoclips_metric_recovery/data/THIRD_PARTY_NOTICES.md) for attribution and terms.

## Contact

For questions about the analysis code, contact [rotation@kaist.ac.kr](mailto:rotation@kaist.ac.kr).
