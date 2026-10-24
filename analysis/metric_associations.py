#!/usr/bin/env python3
"""Reproduce Figure 4, S10 CLAP/FAD associations, and S11 full CLAP regressions.

Uses released metric values, exact clip means, the original crossed bootstrap,
emotion-adjusted weighted regressions, and six-system FAD permutation tests.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import patsy
from scipy import stats
import scipy
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from metric_common import INPUTS, ROOT, TRAITS, read_data

OUT = ROOT / "results/metrics"
BOOTSTRAP_REPLICATES = 20000
RANDOM_STATE = 20261005


def direct_covariance(fit, groups):
    """Construct one finite-corrected cluster component independently."""
    codes, levels = pd.factorize(groups)
    scores = np.asarray(fit.model.wexog) * np.asarray(fit.wresid)[:, None]
    totals = np.zeros((len(levels), scores.shape[1]))
    np.add.at(totals, codes, scores)
    n, k = scores.shape
    correction = len(levels) / (len(levels) - 1) * (n - 1) / (n - k)
    bread = np.linalg.inv(fit.model.wexog.T @ fit.model.wexog)
    return correction * bread @ (totals.T @ totals) @ bread


def crossed_pearson_intervals(ratings, clips, replicates=BOOTSTRAP_REPLICATES):
    """Approximate Fisher-z intervals from crossed bootstrap standard errors.

    Positive participant weights re-estimate each observed clip mean. Independent
    clip weights are normalized within the fixed system-by-emotion cells. The
    bootstrap estimates variance only: its mean shift is not used as a bias
    correction. Every replicate contains all observed clips and rating links.
    """
    clip_index = {name: i for i, name in enumerate(clips.audio_file)}
    users = sorted(ratings.user.unique())
    user_index = {name: i for i, name in enumerate(users)}
    uc = np.asarray([user_index[name] for name in ratings.user])
    cc = np.asarray([clip_index[name] for name in ratings.audio_file])
    observed = np.zeros((len(users), len(clips)))
    observed[uc, cc] = 1
    outcomes = {trait: np.zeros_like(observed) for trait in TRAITS}
    for trait in TRAITS:
        outcomes[trait][uc, cc] = ratings[trait]
        np.testing.assert_allclose(
            outcomes[trait].sum(axis=0) / observed.sum(axis=0), clips[trait]
        )
    cells = [
        np.flatnonzero((clips.model == model) & (clips.emotion == emotion))
        for model in sorted(clips.model.unique())
        for emotion in sorted(clips.emotion.unique())
    ]
    assert len(cells) == 72 and all(len(cell) for cell in cells)
    assert sorted(np.concatenate(cells)) == list(range(len(clips)))
    generator = np.random.default_rng(RANDOM_STATE)
    x = clips.clap_score.to_numpy()
    draws = {trait: np.empty(replicates) for trait in TRAITS}
    for start in range(0, replicates, 250):
        size = min(250, replicates - start)
        participant_weights = generator.exponential(size=(size, len(users)))
        clip_weights = generator.exponential(size=(size, len(clips)))
        for cell in cells:
            clip_weights[:, cell] *= (
                len(cell) / clip_weights[:, cell].sum(axis=1)[:, None]
            )
            np.testing.assert_allclose(clip_weights[:, cell].sum(axis=1), len(cell))
        np.testing.assert_allclose(clip_weights.sum(axis=1), len(clips))
        denominators = participant_weights @ observed
        assert (denominators > 0).all()
        dx = x - (clip_weights * x).sum(axis=1)[:, None] / len(clips)
        variance_x = (clip_weights * dx**2).sum(axis=1)
        for trait in TRAITS:
            means = (participant_weights @ outcomes[trait]) / denominators
            dy = means - (clip_weights * means).sum(axis=1)[:, None] / len(clips)
            variance_y = (clip_weights * dy**2).sum(axis=1)
            r_star = (clip_weights * dx * dy).sum(axis=1) / np.sqrt(
                variance_x * variance_y
            )
            assert np.isfinite(r_star).all() and (np.abs(r_star) < 1).all()
            draws[trait][start : start + size] = r_star
    rows = []
    critical = stats.norm.ppf(0.975)
    for trait in TRAITS:
        r = float(stats.pearsonr(x, clips[trait]).statistic)
        transformed = np.arctanh(draws[trait])
        se_z = float(transformed.std(ddof=1))
        low, high = np.tanh(np.arctanh(r) + np.asarray([-1, 1]) * critical * se_z)
        half_intervals = []
        for sample in np.array_split(transformed, 2):
            half_se = sample.std(ddof=1)
            half_intervals.append(
                np.tanh(np.arctanh(r) + np.asarray([-1, 1]) * critical * half_se)
            )
        stability = float(np.max(np.abs(half_intervals[0] - half_intervals[1])))
        assert stability < 0.005, (trait, stability)
        rows.append(
            {
                "trait": trait,
                "r_ci_low": float(low),
                "r_ci_high": float(high),
                "bootstrap_z_se": se_z,
                "bootstrap_replicates": replicates,
                "bootstrap_mean_r": float(draws[trait].mean()),
                "half_sample_ci_max_difference": stability,
                "r_ci_method": "observed-centered Fisher-z interval; crossed participant/clip bootstrap SE",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(draws)


def system_data(c, fad):
    """Calculate the two reported FAD system-mean weightings."""
    s = (
        c.groupby("model", sort=True)
        .agg(
            valence=("valence", "mean"),
            arousal=("arousal", "mean"),
            clap_score=("clap_score", "mean"),
            n_clips=("audio_file", "size"),
        )
        .reset_index()
        .merge(fad[["model", "fad_score"]], on="model", validate="one_to_one")
    )
    equal = (
        c.groupby(["model", "emotion"])[[*TRAITS, "clap_score"]]
        .mean()
        .groupby("model")
        .mean()
    )
    equal = equal.rename(
        columns={name: name + "_equal_emotion" for name in [*TRAITS, "clap_score"]}
    )
    s = s.merge(equal, on="model", validate="one_to_one")
    return s


def permutation_correlation(x, y):
    """Exhaustive, two-sided permutation of six paired system observations."""
    assert len(x) == len(y) == 6
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    observed = float(np.dot(x, y) / denominator)
    values = np.asarray(
        [
            np.dot(x, permutation) / denominator
            for permutation in itertools.permutations(y)
        ]
    )
    assert len(values) == 720
    count = int((np.abs(values) >= abs(observed) - 1e-12).sum())
    return observed, count / len(values), count


def fad_correlations(s):
    rows = []
    for weighting in ("each_clip_equal", "each_emotion_equal"):
        for trait in TRAITS:
            target = (
                trait if weighting == "each_clip_equal" else trait + "_equal_emotion"
            )
            for method in ("pearson",):
                r, p, tail = permutation_correlation(s.fad_score, s[target])
                rows.append(
                    {
                        "metric": "FAD",
                        "trait": trait,
                        "weighting": weighting,
                        "method": method,
                        "r": r,
                        "p_value": p,
                        "n_systems": 6,
                        "permutations": 720,
                        "permutations_as_or_more_extreme": tail,
                        "inference": "two-sided exhaustive permutation; conditional on the six system summaries",
                    }
                )
    result = pd.DataFrame(rows)
    for (weighting, method), ix in result.groupby(
        ["weighting", "method"]
    ).groups.items():
        result.loc[ix, "p_holm"] = multipletests(
            result.loc[ix, "p_value"], method="holm"
        )[1]
        result.loc[ix, "multiplicity_family"] = (
            f"FAD raw valence and arousal: {weighting}; {method}"
        )
    return result


def adjusted_clap_regressions(ratings, clips, pearson):
    """Fit the two reported emotion-adjusted CLAP models and validate covariance."""
    fitted_models = {}
    users = pd.factorize(ratings.user)[0]
    clip_ids = pd.factorize(ratings.audio_file)[0]
    intersections = pd.factorize(pd.MultiIndex.from_arrays([users, clip_ids]))[0]
    pearson_summary = pearson.set_index("trait")
    rows, coefficients = [], []
    for trait in TRAITS:
        formula = f"{trait} ~ clap_score + C(emotion)"
        fit = smf.wls(formula, ratings, weights=ratings.clip_weight).fit()
        clip_fit = smf.ols(formula, clips).fit()
        assert len(fit.params) == 13
        assert np.linalg.matrix_rank(fit.model.wexog) == 13
        np.testing.assert_allclose(fit.params, clip_fit.params, rtol=1e-10, atol=1e-10)
        covariance = cov_cluster_2groups(fit, users, clip_ids, use_correction=True)[0]
        fitted_models[trait] = (fit, clip_fit, covariance)
        direct = (
            direct_covariance(fit, users)
            + direct_covariance(fit, clip_ids)
            - direct_covariance(fit, intersections)
        )
        np.testing.assert_allclose(covariance, direct, rtol=1e-9, atol=1e-10)
        assert np.linalg.eigvalsh(covariance).min() > 0
        index = fit.model.exog_names.index("clap_score")
        slope = float(fit.params.iloc[index])
        se = float(np.sqrt(covariance[index, index]))
        statistic = slope**2 / se**2
        restriction = np.zeros((1, len(fit.params)))
        restriction[0, index] = 1
        library_test = fit.wald_test(
            restriction, cov_p=covariance, use_f=False, scalar=True
        )
        np.testing.assert_allclose(statistic, library_test.statistic, rtol=1e-12)
        p_value = float(stats.chi2.sf(statistic, 1))
        np.testing.assert_allclose(p_value, library_test.pvalue, rtol=1e-12)
        # Independently recover the adjusted slope after centering by word.
        x = clips.clap_score - clips.groupby("emotion").clap_score.transform("mean")
        y = clips[trait] - clips.groupby("emotion")[trait].transform("mean")
        centered_slope = float(x @ y / (x @ x))
        np.testing.assert_allclose(slope, centered_slope, rtol=1e-10, atol=1e-10)
        r = float(stats.pearsonr(clips.clap_score, clips[trait]).statistic)
        np.testing.assert_allclose(r, pearson_summary.loc[trait, "r"], atol=1e-11)
        rows.append(
            {
                "trait": trait,
                "slope": slope,
                "slope_se": se,
                "slope_ci_low": slope - stats.norm.ppf(0.975) * se,
                "slope_ci_high": slope + stats.norm.ppf(0.975) * se,
                "wald_chi2": statistic,
                "df": 1,
                "p_value": p_value,
                "n_clips": len(clips),
                "n_ratings": len(ratings),
                "n_participants": ratings.user.nunique(),
                "n_systems": clips.model.nunique(),
                "n_emotions": clips.emotion.nunique(),
                "n_parameters": len(fit.params),
                "formula": formula,
                "max_covariance_check_difference": float(
                    np.max(np.abs(covariance - direct))
                ),
                "max_clip_fit_difference": float(
                    np.max(np.abs(fit.params - clip_fit.params))
                ),
                "centered_slope_difference": abs(slope - centered_slope),
                "minimum_covariance_eigenvalue": float(
                    np.linalg.eigvalsh(covariance).min()
                ),
            }
        )
        coefficients.extend(
            {
                "trait": trait,
                "term": term,
                "coefficient": float(value),
                "robust_se": float(np.sqrt(covariance[k, k])),
            }
            for k, (term, value) in enumerate(fit.params.items())
        )
    results = pd.DataFrame(rows)
    results["p_holm"] = multipletests(results.p_value, method="holm")[1]
    results["multiplicity_family"] = (
        "two emotion-adjusted CLAP slope tests: valence and arousal"
    )
    OUT.mkdir(exist_ok=True)
    results.to_csv(OUT / "clap_predictor_tests.csv", index=False, float_format="%.12g")
    pd.DataFrame(coefficients).to_csv(
        OUT / "coefficients.csv", index=False, float_format="%.12g"
    )
    return results, fitted_models


def export_full_coefficients(ratings, clips, fitted_models, output: Path):
    """Export all terms, fitted-model summaries, and two-way covariance."""
    output.mkdir(parents=True, exist_ok=True)
    coefficient_rows, model_rows = [], []
    for trait in ("valence", "arousal"):
        formula = f"{trait} ~ clap_score + C(emotion)"
        fit, clip_fit, covariance = fitted_models[trait]
        standard_errors = np.sqrt(np.diag(covariance))
        coefficients = fit.params.to_numpy()
        statistics = coefficients / standard_errors
        critical = stats.norm.ppf(0.975)
        for i, term in enumerate(fit.params.index):
            coefficient_rows.append(
                {
                    "outcome": trait,
                    "term": term,
                    "coefficient": coefficients[i],
                    "cluster_robust_se": standard_errors[i],
                    "ci_95_low": coefficients[i] - critical * standard_errors[i],
                    "ci_95_high": coefficients[i] + critical * standard_errors[i],
                    "wald_z": statistics[i],
                    "wald_chi2": statistics[i] ** 2,
                    "wald_df": 1,
                    "p_unadjusted": stats.chi2.sf(statistics[i] ** 2, 1),
                }
            )
        # Summarize goodness of fit on the actual weighted rating observations.
        # Clip-mean R² is recorded separately because within-clip variation differs.
        restriction = np.eye(len(fit.params))[1:]
        omnibus = fit.wald_test(restriction, cov_p=covariance, use_f=False, scalar=True)
        model_rows.append(
            {
                "outcome": trait,
                "formula": formula,
                "emotion_reference": "angry",
                "n_ratings": int(fit.nobs),
                "n_participants": ratings.user.nunique(),
                "n_clips": ratings.audio_file.nunique(),
                "n_emotions": ratings.emotion.nunique(),
                "n_parameters": len(fit.params),
                "design_rank": np.linalg.matrix_rank(fit.model.wexog),
                "residual_df": int(fit.df_resid),
                "rating_weighted_r_squared": fit.rsquared,
                "rating_weighted_adjusted_r_squared": fit.rsquared_adj,
                "weighted_residual_sum_squares": fit.ssr,
                "weighted_total_sum_squares": fit.centered_tss,
                "rating_weighted_rmse": np.sqrt(fit.ssr / ratings.clip_weight.sum()),
                "clip_mean_r_squared": clip_fit.rsquared,
                "clip_mean_adjusted_r_squared": clip_fit.rsquared_adj,
                "robust_omnibus_chi2": float(omnibus.statistic),
                "robust_omnibus_df": int(restriction.shape[0]),
                "robust_omnibus_p_unadjusted": float(omnibus.pvalue),
                "max_rating_clip_coefficient_difference": np.max(
                    np.abs(fit.params - clip_fit.params)
                ),
                "minimum_covariance_eigenvalue": np.linalg.eigvalsh(covariance).min(),
            }
        )
        pd.DataFrame(
            covariance, index=fit.params.index, columns=fit.params.index
        ).to_csv(output / f"{trait}_cluster_covariance.csv", float_format="%.15g")
    coefficients = pd.DataFrame(coefficient_rows)
    slopes = coefficients.term.eq("clap_score")
    coefficients["p_holm_two_clap_slopes"] = np.nan
    coefficients.loc[slopes, "p_holm_two_clap_slopes"] = multipletests(
        coefficients.loc[slopes, "p_unadjusted"], method="holm"
    )[1]
    models = pd.DataFrame(model_rows)
    coefficients.to_csv(
        output / "clap_full_coefficients.csv", index=False, float_format="%.15g"
    )
    models.to_csv(output / "clap_model_fit.csv", index=False, float_format="%.15g")
    return coefficients, models


def main():
    ratings, clips, fad, rounding = read_data()
    OUT.mkdir(parents=True, exist_ok=True)
    intervals, draws = crossed_pearson_intervals(ratings, clips)
    pearson = pd.DataFrame(
        [
            {
                "trait": trait,
                "r": float(stats.pearsonr(clips.clap_score, clips[trait]).statistic),
                "n_clips": len(clips),
                "n_ratings": len(ratings),
                "n_participants": ratings.user.nunique(),
            }
            for trait in TRAITS
        ]
    ).merge(intervals, on="trait", validate="one_to_one")
    systems = system_data(clips, fad)
    fad_results = fad_correlations(systems)
    slopes, fitted_models = adjusted_clap_regressions(ratings, clips, pearson)
    export_full_coefficients(ratings, clips, fitted_models, OUT)
    for frame, name in [
        (pearson, "overall_associations.csv"),
        (draws, "crossed_bootstrap_draws.csv"),
        (clips, "clip_data.csv"),
        (systems, "system_data.csv"),
        (fad_results, "fad_correlations.csv"),
    ]:
        frame.to_csv(OUT / name, index=False, float_format="%.12g")
    metadata = {
        "inputs_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in INPUTS.items()
        },
        "packages": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "patsy": patsy.__version__,
        },
        "mean_metadata_rounding_difference": rounding,
        "clap_model": "Separate valence/arousal ~ clap_score + C(emotion) models; angry reference; intercept, 11 emotion indicators and one common CLAP slope. No system term or interaction.",
        "clap_weight": "Inverse clip-rating-count weights; each clip has total weight one. Coefficients reproduce OLS on 991 exact clip means.",
        "clap_covariance": "Participant plus clip minus participant-by-clip intersection. Each component uses finite correction G/(G-1) * (N-1)/(N-K); N=6162 and K=13.",
        "clap_tests": "Approximate Wald chi-square(1). Holm correction across the two CLAP slopes. Full coefficient p-values are unadjusted, with slope Holm p-values separately identified.",
        "coefficient_intervals": "Unadjusted normal 95% confidence intervals from participant/clip-clustered standard errors.",
        "goodness_of_fit": "Weighted rating-level R-squared uses weighted centered total sum of squares. Clip-mean R-squared is separately identified. Full-model Wald chi-square(12) tests all non-intercept coefficients jointly.",
        "pearson_intervals": "20,000 independent exponential participant/clip reweightings, batches of 250; clip weights normalized within 72 system-by-emotion cells. Recompute weighted clip means, calculate bootstrap Fisher-z standard errors, and center normal intervals on observed Fisher-z correlations.",
        "random_state": RANDOM_STATE,
        "fad_tests": "Pearson correlations across six system summaries, with equal-clip and equal-emotion weighting. Two-sided exhaustive 720 permutations; Holm across valence/arousal within each weighting.",
        "missing_data": "All retained ratings have both outcomes and all clips join to metric values. No omission or imputation within these analyses.",
        "limitations": "Conditional on six configurations, 12 selected words and observed rating links. Cluster inference assumes no residual dependence among ratings sharing neither participant nor clip. Crossed bootstrap coverage is approximate. Saved metric values are inputs; historical embedding/checkpoint/reference selection is not recomputed.",
    }
    (OUT / "method_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pearson[["trait", "r", "r_ci_low", "r_ci_high"]].to_string(index=False))
    print(
        slopes[["trait", "slope", "slope_se", "wald_chi2", "p_holm"]].to_string(
            index=False
        )
    )
    print(fad_results[["trait", "weighting", "r", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
