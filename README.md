# AImoclips analysis

Code and de-identified analysis inputs for the manuscript “AImoclips: Benchmarking emotion conveyance in text-to-music generation using a dimensional valence-arousal framework.”

| File | Role and analyses |
| --- | --- |
| [run_analysis.py](run_analysis.py) | Runs all five analysis stages in order and saves numerical outputs to `results/`. |
| [descriptive_statistics.py](analysis/descriptive_statistics.py) | Summarizes ratings and clip means, quadrant accuracy/F1 and confusion counts; tests overall accuracy against chance and performs Tukey HSD comparisons. |
| [core_statistics.py](analysis/core_statistics.py) | Estimates inter-rater agreement and variance components; runs ANOVA and robustness checks, centralization sign-flip tests, accuracy/F1 intervals, binomial chance tests, and pairwise permutation tests (S12). |
| [system_deviation_tests.py](analysis/system_deviation_tests.py) | Tests deviations from emotion references with participant- and clip-clustered uncertainty (S8), and calculates system-by-emotion factorial effect sizes (S9). |
| [distance_wald_tests.py](analysis/distance_wald_tests.py) | Tests intended valence/arousal effects on distance from emotion references using HC3-robust Wald tests and partial eta-squared (S4). |
| [metric_associations.py](analysis/metric_associations.py) | Computes CLAP correlations with crossed-bootstrap intervals, emotion-adjusted CLAP regressions, and FAD permutation correlations (Figure 4; S10 summaries and S11 regression coefficients). |
| [metric_common.py](analysis/metric_common.py) | Provides shared input validation, exact clip means, regression weights, and cluster-covariance calculations. |

See [RUNNING.md](RUNNING.md) for setup and execution.

## Contact

For questions about the analysis code, contact [rotation@kaist.ac.kr](mailto:rotation@kaist.ac.kr).
