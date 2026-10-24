#!/usr/bin/env python3
"""Reproduce S8 reference-deviation tests and S9 factorial effect sizes.

Uses individual ratings for clustered zero-deviation tests and released clip
means for the system-by-emotion factorial model, preserving both input scales.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from metric_common import manual_cluster_covariance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/system_deviations"
MODELS = ["AudioLDM2", "MusicGen", "Mustango", "StableAudio", "Suno", "Udio"]
INPUTS = [
    ROOT / "data" / name
    for name in [
        "ratings_metadata.csv",
        "clips_metadata.csv",
        "emotion_intents_metadata.csv",
    ]
]


def read_data():
    ratings, clips, words = [pd.read_csv(p) for p in INPUTS]
    assert len(ratings) == 6162 and ratings.user.nunique() == 111
    assert not ratings.duplicated(["user", "audio_file"]).any()
    assert len(clips) == 991 and clips.audio_file.is_unique
    assert set(clips.model) == set(MODELS) and words.emotion.nunique() == 12
    assert set(ratings.audio_file) == set(clips.audio_file)
    metadata = clips[["audio_file", "model", "emotion"]].merge(
        words[["emotion", "gt_valence", "gt_arousal"]],
        on="emotion",
        validate="many_to_one",
    )
    r = ratings.merge(metadata, on="audio_file", validate="many_to_one")
    counts = r.groupby("audio_file").size()
    r["weight"] = 1 / r.audio_file.map(counts)
    np.testing.assert_allclose(r.groupby("audio_file").weight.sum(), 1)
    return r, clips


def zero_deviation_tests(ratings):
    rows = []
    for model in MODELS:
        g = ratings[ratings.model == model].copy()
        user_codes = pd.factorize(g.user)[0]
        clip_codes = pd.factorize(g.audio_file)[0]
        pair_codes = pd.factorize(pd.MultiIndex.from_arrays([user_codes, clip_codes]))[
            0
        ]
        df = min(g.user.nunique(), g.audio_file.nunique()) - 1
        for trait in ["valence", "arousal"]:
            g["deviation"] = g[trait] - g["gt_" + trait]
            fit = smf.wls("deviation ~ 1", g, weights=g.weight).fit()
            covariance = cov_cluster_2groups(
                fit, user_codes, clip_codes, use_correction=True
            )[0]
            direct = (
                manual_cluster_covariance(fit, user_codes)
                + manual_cluster_covariance(fit, clip_codes)
                - manual_cluster_covariance(fit, pair_codes)
            )
            np.testing.assert_allclose(covariance, direct, rtol=1e-10, atol=1e-12)
            mean = float(fit.params.iloc[0])
            clip_mean = g.groupby("audio_file").deviation.mean().mean()
            np.testing.assert_allclose(mean, clip_mean, rtol=1e-12, atol=1e-12)
            assert covariance[0, 0] > 0
            se = np.sqrt(covariance[0, 0])
            statistic = mean / se
            critical = stats.t.ppf(0.975, df)
            rows.append(
                {
                    "model": model,
                    "trait": trait,
                    "mean_deviation": mean,
                    "se": se,
                    "ci_low": mean - critical * se,
                    "ci_high": mean + critical * se,
                    "t": statistic,
                    "df": df,
                    "p_value": 2 * stats.t.sf(abs(statistic), df),
                    "n_ratings": len(g),
                    "n_clips": g.audio_file.nunique(),
                    "n_participants": g.user.nunique(),
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm"] = multipletests(result.p_value, method="holm")[1]
    result["multiplicity_family"] = "all 12 system-by-dimension zero-deviation tests"
    return result


def omnibus_effect_sizes(clips):
    saved = pd.read_csv(ROOT / "results/core/anova_robustness.csv")
    term_map = {
        "balanced_system": "C(model, Sum)",
        "balanced_emotion": "C(emotion, Sum)",
        "balanced_system_by_emotion_interaction": "C(model, Sum):C(emotion, Sum)",
    }
    rows = []
    for outcome in ["valence", "arousal"]:
        fit = smf.ols(f"{outcome} ~ C(model, Sum) * C(emotion, Sum)", clips).fit()
        table = sm.stats.anova_lm(fit, typ=3)  # No robust F-to-SS conversion.
        design = fit.model.exog
        y = fit.model.endog
        systems = sorted(clips.model.unique())
        words = sorted(clips.emotion.unique())

        def cell(system, word):
            return np.asarray(
                build_design_matrices(
                    [fit.model.data.design_info],
                    pd.DataFrame({"model": [system], "emotion": [word]}),
                )[0]
            )[0]

        cells = {(m, e): cell(m, e) for m in systems for e in words}
        system_means = {
            m: np.mean([cells[m, e] for e in words], axis=0) for m in systems
        }
        word_means = {e: np.mean([cells[m, e] for m in systems], axis=0) for e in words}
        contrasts = {
            "balanced_system": np.vstack(
                [system_means[m] - system_means[systems[-1]] for m in systems[:-1]]
            ),
            "balanced_emotion": np.vstack(
                [word_means[e] - word_means[words[-1]] for e in words[:-1]]
            ),
            "balanced_system_by_emotion_interaction": np.vstack(
                [
                    cells[m, e]
                    - cells[m, words[-1]]
                    - cells[systems[-1], e]
                    + cells[systems[-1], words[-1]]
                    for m in systems[:-1]
                    for e in words[:-1]
                ]
            ),
        }
        robust = fit.get_robustcov_results(cov_type="HC3")
        for term, name in term_map.items():
            columns = fit.model.data.design_info.term_name_slices[name]
            keep = np.ones(design.shape[1], dtype=bool)
            keep[columns] = False
            reduced = sm.OLS(y, design[:, keep]).fit()
            ss_effect = reduced.ssr - fit.ssr
            np.testing.assert_allclose(
                ss_effect, table.loc[name, "sum_sq"], rtol=1e-10, atol=1e-9
            )
            old = saved[(saved.outcome == outcome) & (saved.term == term)].iloc[0]
            # Both restrictions span precisely the same term-column subspace.
            R = contrasts[term]
            np.testing.assert_allclose(R[:, keep], 0, atol=1e-13)
            assert np.linalg.matrix_rank(R[:, ~keep]) == (~keep).sum()
            wald = robust.wald_test(R, use_f=True, scalar=True)
            np.testing.assert_allclose(wald.fvalue, old.statistic, rtol=1e-8)
            assert old.df_den == fit.df_resid
            eta = ss_effect / (ss_effect + fit.ssr)
            rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "statistic": old.statistic,
                    "p_value": old.p_value,
                    "df_num": old.df_num,
                    "df_den": old.df_den,
                    "eta_p_squared": eta,
                    "ss_effect": ss_effect,
                    "ss_residual": fit.ssr,
                    "effect_size_method": "partial SS for full-model balanced hypothesis",
                }
            )
    return pd.DataFrame(rows)


def main():
    ratings, clips = read_data()
    OUT.mkdir(parents=True, exist_ok=True)
    zero = zero_deviation_tests(ratings)
    effects = omnibus_effect_sizes(clips)
    zero.to_csv(OUT / "zero_deviation_tests.csv", index=False, float_format="%.12g")
    effects.to_csv(OUT / "omnibus_effect_sizes.csv", index=False, float_format="%.12g")
    metadata = {
        "inputs": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in INPUTS
        },
        "zero_test": "two-sided finite-corrected participant/clip-cluster t tests",
        "zero_test_weight": "each clip total weight one",
        "zero_test_holm": "all 12 system-by-dimension tests together",
        "zero_test_ci": "unadjusted two-sided t intervals",
        "reference": "the 12 selected normative word scores treated as fixed",
        "scope": "six evaluated systems and selected emotion words",
        "omnibus": "retained HC3 balanced Wald F results on 991 metadata clip means",
        "effect_size": "Type III partial SS verified by constrained model SSE; not derived from robust F",
    }
    (OUT / "method_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        zero[["model", "trait", "mean_deviation", "df", "p_holm"]].to_string(
            index=False
        )
    )
    print(
        effects[["outcome", "term", "statistic", "eta_p_squared"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
