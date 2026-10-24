#!/usr/bin/env python3
"""Reproduce S4 intended-valence/arousal factorial tests of clip distance.

Uses released clip means, HC3 covariance, equal-weight marginal contrasts and
Type III partial sums of squares, checked independently from cell moments.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/distance"
INPUTS = [
    ROOT / "data" / name
    for name in ["clips_metadata.csv", "emotion_intents_metadata.csv"]
]


def main():
    clips, words = [pd.read_csv(p) for p in INPUTS]
    assert len(clips) == 991 and clips.audio_file.is_unique
    assert words.emotion.nunique() == 12
    c = clips.merge(
        words[["emotion", "gt_valence", "gt_arousal"]],
        on="emotion",
        validate="many_to_one",
    )
    assert not c[["valence", "arousal", "gt_valence", "gt_arousal"]].isna().any().any()
    c["distance"] = np.hypot(c.valence - c.gt_valence, c.arousal - c.gt_arousal)
    c["intended_valence"] = np.where(c.gt_valence > 5, "High", "Low")
    c["intended_arousal"] = np.where(c.gt_arousal > 5, "High", "Low")
    fit = smf.ols(
        "distance ~ C(intended_valence, Sum) * C(intended_arousal, Sum)", c
    ).fit()
    robust = fit.get_robustcov_results(cov_type="HC3")
    cells = [("High", "High"), ("High", "Low"), ("Low", "High"), ("Low", "Low")]
    grid = pd.DataFrame(cells, columns=["intended_valence", "intended_arousal"])
    design_cells = np.asarray(
        build_design_matrices([fit.model.data.design_info], grid)[0]
    )
    term_names = [
        "C(intended_valence, Sum)",
        "C(intended_arousal, Sum)",
        "C(intended_valence, Sum):C(intended_arousal, Sum)",
    ]
    contrasts = [
        np.array([0.5, 0.5, -0.5, -0.5]),
        np.array([0.5, -0.5, 0.5, -0.5]),
        np.array([1.0, -1.0, -1.0, 1.0]),
    ]
    moments = []
    for valence, arousal in cells:
        y = c.loc[
            (c.intended_valence == valence) & (c.intended_arousal == arousal),
            "distance",
        ].to_numpy()
        n = len(y)
        assert n >= 240
        moments.append(
            {
                "intended_valence": valence,
                "intended_arousal": arousal,
                "n": n,
                "mean_distance": y.mean(),
                "sample_variance": y.var(ddof=1),
                "hc3_mean_variance": ((y - y.mean()) ** 2).sum() / (n - 1) ** 2,
            }
        )
    moments = pd.DataFrame(moments)
    ss = sm.stats.anova_lm(fit, typ=3)
    rows = []
    for label, name, contrast in zip(
        ["Intended valence", "Intended arousal", "Intended valence x arousal"],
        term_names,
        contrasts,
    ):
        restriction = (contrast @ design_cells)[None, :]
        wald = robust.wald_test(restriction, use_f=True, scalar=True)
        estimate = float(contrast @ moments.mean_distance)
        variance = float(contrast**2 @ moments.hc3_mean_variance)
        manual_f = estimate**2 / variance
        np.testing.assert_allclose(wald.fvalue, manual_f, rtol=1e-11)
        np.testing.assert_allclose(
            wald.pvalue, stats.f.sf(manual_f, 1, 987), rtol=1e-11
        )
        columns = fit.model.data.design_info.term_name_slices[name]
        keep = np.ones(fit.model.exog.shape[1], dtype=bool)
        keep[columns] = False
        partial_ss = (
            sm.OLS(fit.model.endog, fit.model.exog[:, keep]).fit().ssr - fit.ssr
        )
        np.testing.assert_allclose(partial_ss, ss.loc[name, "sum_sq"], rtol=1e-10)
        rows.append(
            {
                "term": label,
                "statistic": float(wald.fvalue),
                "df_num": int(wald.df_num),
                "df_den": int(wald.df_denom),
                "p_value": float(wald.pvalue),
                "eta_p_squared": partial_ss / (partial_ss + fit.ssr),
                "contrast_estimate": estimate,
                "contrast_variance": variance,
                "partial_ss": partial_ss,
                "residual_ss": fit.ssr,
                "manual_f": manual_f,
                "n_clips": len(c),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "factorial_wald_tests.csv", index=False, float_format="%.12g")
    moments.to_csv(OUT / "cell_moments.csv", index=False, float_format="%.12g")
    metadata = {
        "inputs": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in INPUTS
        },
        "model": "distance ~ intended valence * intended arousal; sum-to-zero coding",
        "outcome": "Euclidean distance of released clip mean VA from fixed word reference",
        "unit": "991 clip means; each clip has equal weight",
        "contrasts": "main effects average equally over the other factor; interaction is difference of differences",
        "covariance": "version 3 heteroscedasticity-consistent (HC3)",
        "inference": "approximate F(1,987); no exact finite-sample distribution claim",
        "effect_size": "partial model sums of squares / (partial SS + residual SS), not derived from robust F",
        "validation": "independent cell-moment HC3 Wald statistics and constrained-model sums of squares",
        "limitation": "shared-listener correlations across clip means are not accounted for",
        "multiplicity": "three factorial hypotheses; reported p-values unadjusted, as in the original factorial analysis",
        "versions": {
            "statsmodels": sm.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (OUT / "method_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        result[
            ["term", "statistic", "df_num", "df_den", "p_value", "eta_p_squared"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
