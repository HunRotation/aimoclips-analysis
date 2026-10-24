#!/usr/bin/env python3
"""Reproduce agreement, factorial tests, centralization, and quadrant accuracy.

These analyses support Results and S3, S4, S6, S7, S9, and S12 Tables.
Released rounded clip means and the original random-draw order are preserved.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
import warnings

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/core"
RATINGS = ROOT / "data/ratings_metadata.csv"
CLIPS = ROOT / "data/clips_metadata.csv"
EMOTIONS = ROOT / "data/emotion_intents_metadata.csv"
QUADRANTS = ROOT / "results/clip_quadrants.csv"
MODELS = ["AudioLDM2", "MusicGen", "Mustango", "StableAudio", "Suno", "Udio"]
QMAP = {"hv_ha": 1, "lv_ha": 2, "lv_la": 3, "hv_la": 4}


def read_inputs():
    """Load released metadata and the reconstructed quadrant labels."""
    return tuple(pd.read_csv(path) for path in (RATINGS, CLIPS, EMOTIONS, QUADRANTS))


def validate(ratings, clips, emotions, quadrant) -> list[str]:
    errors: list[str] = []
    checks = {
        "users": ratings.user.nunique() == 111,
        "ratings": len(ratings) == 6162,
        "rated_clips": ratings.audio_file.nunique() == 991,
        "clip_rows": len(clips) == 991 and clips.audio_file.nunique() == 991,
        "systems": clips.model.nunique() == 6 and set(clips.model) == set(MODELS),
        "emotions": clips.emotion.nunique() == 12 and emotions.emotion.nunique() == 12,
        "no_duplicate_user_clip": not ratings.duplicated(["user", "audio_file"]).any(),
        "exact_rating_clip_join": set(ratings.audio_file) == set(clips.audio_file),
        "exact_quadrant_clip_join": set(quadrant.audio_file) == set(clips.audio_file),
    }
    for name, ok in checks.items():
        if not ok:
            errors.append(name)
    # Metadata joins are exact, including model/emotion labels.
    if (
        not clips[["audio_file", "model", "emotion"]]
        .merge(
            quadrant[["audio_file", "model", "emotion"]], how="outer", indicator=True
        )["_merge"]
        .eq("both")
        .all()
    ):
        errors.append("exact_quadrant_metadata_join")
    return errors


def prepare(ratings, clips, emotions, quadrant):
    target = emotions.set_index("emotion")[["gt_valence", "gt_arousal"]]
    c = clips.merge(target, left_on="emotion", right_index=True, validate="many_to_one")
    c["quadrant"] = c.emotion.map(emotions.set_index("emotion").quadrant)
    c["intended"] = c.quadrant.map(QMAP)
    c["distance"] = np.hypot(c.valence - c.gt_valence, c.arousal - c.gt_arousal)
    c = c.merge(
        quadrant[["audio_file", "rated_quadrant"]],
        on="audio_file",
        validate="one_to_one",
    )
    return ratings.merge(
        clips[["audio_file", "model", "emotion"]],
        on="audio_file",
        validate="many_to_one",
    ), c


def write(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    # Fixed ordering and float formatting make deterministic comparisons useful.
    df = df.copy()
    df = (
        df.sort_values(list(df.columns), kind="mergesort", na_position="last")
        if len(df)
        else df
    )
    df.to_csv(OUT / name, index=False, float_format="%.10g")


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    out[order] = np.minimum(
        1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    )
    return out


def alpha(data: pd.DataFrame, trait: str) -> float:
    return alpha_level(data, trait, "interval")


def alpha_level(data: pd.DataFrame, trait: str, level: str) -> float:
    if krippendorff is None:
        return float("nan")
    wide = data.pivot(index="user", columns="audio_file", values=trait)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(
            krippendorff.alpha(wide.to_numpy(dtype=float), level_of_measurement=level)
        )


def alpha_counts(counts: np.ndarray) -> float:
    """Fast interval alpha from per-unit 1--9 value counts (for bootstrap)."""
    if krippendorff is None:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(
            krippendorff.alpha(
                value_counts=counts,
                value_domain=np.arange(1, 10),
                level_of_measurement="interval",
            )
        )


def reliability(
    ratings: pd.DataFrame, clips: pd.DataFrame, rng: np.random.Generator, nboot: int
):
    """Estimate interval agreement and crossed variance components for S3."""
    rows, boots = [], []
    for trait in ["valence", "arousal"]:
        y = ratings[["user", "audio_file", trait]].dropna()
        # Genuine crossed random intercepts: one constant top-level group and
        # two explicitly named variance components.
        fit = None
        converged = False
        singular = True
        user_var = clip_var = resid_var = np.nan
        limitation = ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vc = {"participant": "0 + C(user)", "clip": "0 + C(audio_file)"}
                fit = smf.mixedlm(
                    f"{trait} ~ 1",
                    y,
                    groups=np.ones(len(y)),
                    vc_formula=vc,
                    re_formula="0",
                ).fit(reml=True, method="lbfgs", maxiter=300, disp=False)
            converged = bool(fit.converged)
            component_names = list(fit.model.exog_vc.names)
            component_values = dict(zip(component_names, map(float, fit.vcomp)))
            user_var = component_values.get("participant", np.nan)
            clip_var = component_values.get("clip", np.nan)
            resid_var = float(fit.scale)
            singular = bool(
                (not np.isfinite(user_var + clip_var + resid_var))
                or user_var < 1e-10
                or clip_var < 1e-10
            )
            limitation = (
                "crossed participant random intercept + clip variance component; REML"
            )
        except Exception as exc:
            limitation = f"crossed MixedLM failed: {type(exc).__name__}: {exc}"
        if converged and not singular:
            average_n = float(y.groupby("audio_file").size().mean())
            for k, label in [(average_n, "mean_of_observed")]:
                rel = clip_var / (clip_var + (user_var + resid_var) / k)
                rows.append(
                    {
                        "trait": trait,
                        "estimand": label,
                        "k": k,
                        "reliability": rel,
                        "clip_variance": clip_var,
                        "rater_variance": user_var,
                        "residual_variance": resid_var,
                        "n_ratings": len(y),
                        "converged": converged,
                        "singular": singular,
                        "component_names": ";".join(
                            name
                            for name in ["participant", "clip"]
                            if name in component_names
                        ),
                        "method": limitation,
                    }
                )
        else:
            rows.append(
                {
                    "trait": trait,
                    "estimand": "crossed_model_status",
                    "k": np.nan,
                    "reliability": np.nan,
                    "clip_variance": clip_var,
                    "rater_variance": user_var,
                    "residual_variance": resid_var,
                    "n_ratings": len(y),
                    "converged": converged,
                    "singular": singular,
                    "component_names": "participant;clip",
                    "method": limitation,
                }
            )
        observed_alpha = alpha(y, trait)
        count_table = (
            y.assign(_value=y[trait].astype(int))
            .pivot_table(
                index="audio_file", columns="_value", aggfunc="size", fill_value=0
            )
            .reindex(columns=range(1, 10), fill_value=0)
            .to_numpy(dtype=int)
        )
        # A clip bootstrap retains the sparse rater-by-clip pattern by sampling
        # columns, then recomputing interval alpha.
        clips_u = y.audio_file.unique()
        for b in range(nboot):
            sampled = rng.integers(0, len(clips_u), size=len(clips_u))
            boots.append(
                {
                    "trait": trait,
                    "bootstrap": b,
                    "alpha": alpha_counts(count_table[sampled]),
                }
            )
        rows.append(
            {
                "trait": trait,
                "estimand": "interval_krippendorff_alpha",
                "k": np.nan,
                "reliability": observed_alpha,
                "clip_variance": clip_var,
                "rater_variance": user_var,
                "residual_variance": resid_var,
                "n_ratings": len(y),
                "converged": converged,
                "singular": singular,
                "component_names": "participant;clip",
                "method": "interval alpha, sparse wide matrix",
            }
        )
    write(pd.DataFrame(rows), "reliability_summary.csv")
    bdf = pd.DataFrame(boots)
    if len(bdf):
        ci = (
            bdf.groupby("trait")
            .alpha.quantile([0.025, 0.975])
            .unstack()
            .rename(columns={0.025: "ci_low", 0.975: "ci_high"})
            .reset_index()
        )
        bdf = bdf.merge(ci, on="trait", how="left")
    write(bdf, "reliability_bootstrap.csv")


def robust_models(c: pd.DataFrame, rng: np.random.Generator, nboot: int = 2000):
    """Calculate quadrant/factorial tests and exact centralization tests."""
    rows = []
    for outcome in ["valence", "arousal"]:
        m = smf.ols(f"{outcome} ~ C(model, Sum) * C(emotion, Sum)", data=c).fit(
            cov_type="HC3"
        )
        try:
            # Main-effect contrasts give equal weight to the other fixed factor.
            from patsy import build_design_matrices

            emotions = sorted(c.emotion.unique())
            models = sorted(c.model.unique())
            di = m.model.data.design_info

            def X_for(mm, ee):
                return np.asarray(
                    build_design_matrices(
                        [di], pd.DataFrame({"model": mm, "emotion": ee})
                    )[0]
                )

            Xcell = {(mm, ee): X_for([mm], [ee])[0] for mm in models for ee in emotions}
            Xsys = np.vstack(
                [np.mean([Xcell[(mm, ee)] for ee in emotions], axis=0) for mm in models]
            )
            Xemo = np.vstack(
                [np.mean([Xcell[(mm, ee)] for mm in models], axis=0) for ee in emotions]
            )
            Rmap = {
                "balanced_system": np.vstack(
                    [Xsys[i] - Xsys[-1] for i in range(len(models) - 1)]
                ),
                "balanced_emotion": np.vstack(
                    [Xemo[i] - Xemo[-1] for i in range(len(emotions) - 1)]
                ),
                "balanced_system_by_emotion_interaction": np.vstack(
                    [
                        Xcell[(mm, ee)]
                        - Xcell[(mm, emotions[-1])]
                        - Xcell[(models[-1], ee)]
                        + Xcell[(models[-1], emotions[-1])]
                        for mm in models[:-1]
                        for ee in emotions[:-1]
                    ]
                ),
            }
            for term, R in Rmap.items():
                wt = m.wald_test(R, use_f=True, scalar=True)
                statistic = float(np.asarray(wt.fvalue).squeeze())
                pvalue = float(np.asarray(wt.pvalue).squeeze())
                rows.append(
                    {
                        "outcome": outcome,
                        "term": term,
                        "method": "HC3 balanced EMM Wald F",
                        "n": int(m.nobs),
                        "statistic": statistic,
                        "p_value": pvalue,
                        "df_num": float(wt.df_num),
                        "df_den": float(wt.df_denom),
                        "estimand": "joint balanced estimated marginal means; equal weight over 12 fixed emotions",
                        "coding": "sum-to-zero; explicit EMM contrast matrix",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "outcome": outcome,
                    "term": "HC3 Wald F unavailable",
                    "method": "limitation",
                    "n": int(m.nobs),
                    "note": f"{type(exc).__name__}: {exc}",
                }
            )
    # Conventional and heteroscedasticity/rank tests for the quadrant question.
    groups = [
        c.loc[c.quadrant == q, "distance"].to_numpy()
        for q in sorted(c.quadrant.unique())
    ]
    f = stats.f_oneway(*groups)
    kr = stats.kruskal(*groups)
    try:
        from statsmodels.stats.oneway import anova_oneway

        all_values = np.concatenate(groups)
        labels = np.concatenate([np.repeat(i, len(g)) for i, g in enumerate(groups)])
        wf = anova_oneway(all_values, labels, use_var="unequal")
        bf = anova_oneway(all_values, labels, use_var="bf")
        welch = (
            "welch_anova",
            float(wf.statistic),
            float(wf.pvalue),
            float(wf.df[0]),
            float(wf.df[1]),
        )
        brown_mean = (
            "brown_forsythe_mean_anova",
            float(bf.statistic),
            float(bf.pvalue),
            float(bf.df[0]),
            float(bf.df[1]),
        )
    except Exception:
        welch = ("welch_anova", np.nan, np.nan, np.nan, np.nan)
        brown_mean = ("brown_forsythe_mean_anova", np.nan, np.nan, np.nan, np.nan)
    for method, stat, p, dfn, dfd in [
        ("conventional_anova", f.statistic, f.pvalue, 3, len(c) - 4),
        welch,
        brown_mean,
        ("kruskal_wallis", kr.statistic, kr.pvalue, np.nan, np.nan),
    ]:
        rows.append(
            {
                "outcome": "distance",
                "term": "quadrant",
                "method": method,
                "n": len(c),
                "statistic": stat,
                "p_value": p,
                "df_num": dfn,
                "df_den": dfd,
                "estimand": "quadrant distance means",
            }
        )
    write(pd.DataFrame(rows), "anova_robustness.csv")
    # Centralization sensitivity is a paired comparison: perceived clip
    # distance from (5,5) versus the selected external word-reference distance.
    c["neutral_center_distance"] = np.hypot(c.valence - 5, c.arousal - 5)
    c["reference_center_distance"] = np.hypot(c.gt_valence - 5, c.gt_arousal - 5)
    c["centralization_delta"] = c.neutral_center_distance - c.reference_center_distance
    cen_rows = []
    word_values = sorted(c.emotion.unique())
    # Keep all seven bootstrap groups: their draws precede accuracy resampling.
    for model, g in list(c.groupby("model")) + [("overall", c)]:
        word_means = (
            g.groupby("emotion", observed=True)
            .centralization_delta.mean()
            .reindex(word_values)
            .dropna()
            .to_numpy()
        )
        # Primary centralization estimand is the equally weighted mean of 12
        # word means. Enumerate all 2^12 sign flips for exact two-sided p.
        signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(word_means))))
        null_means = signs @ word_means / len(word_means)
        p_exact = float(np.mean(np.abs(null_means) >= abs(word_means.mean()) - 1e-12))
        # Bootstrap these 12 word means directly, not their unequal clip rows.
        bmeans = []
        for _ in range(nboot):
            sampled = rng.choice(word_means, size=len(word_means), replace=True)
            bmeans.append(np.mean(sampled))
        cen_rows.append(
            {
                "model": model,
                "metric": "centralization_delta_center_minus_reference",
                "mean": word_means.mean(),
                "ci_low": np.quantile(bmeans, 0.025),
                "ci_high": np.quantile(bmeans, 0.975),
                "word_level_exact_sign_flip_p": p_exact,
                "n_clips": len(g),
                "n_words": len(word_means),
                "interpretation": "negative means perceived clip mean is closer to (5,5) than selected external word coordinate; primary mean equally weights words",
            }
        )
    cen_df = pd.DataFrame(cen_rows)
    system_ix = cen_df.model != "overall"
    cen_df.loc[system_ix, "word_level_p_holm_across_systems"] = holm(
        cen_df.loc[system_ix, "word_level_exact_sign_flip_p"]
    )
    write(cen_df, "centralization_robustness.csv")


def wilson(x, n, z=1.959963984540054):
    p = x / n if n else np.nan
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - half, ctr + half


def accuracy(ratings, c, rng, nboot: int, nperm: int):
    """Calculate S7 intervals and S12 emotion-stratified permutation tests."""

    def wf1(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        score = 0.0
        for lab in [1, 2, 3, 4]:
            tp = np.sum((y_true == lab) & (y_pred == lab))
            fp = np.sum((y_true != lab) & (y_pred == lab))
            fn = np.sum((y_true == lab) & (y_pred != lab))
            f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            score += np.sum(y_true == lab) / len(y_true) * f
        return float(score)

    rows, frows = [], []
    # Existing clip aggregate labels are primary and retain neutral as error.
    for model, g in c.groupby("model"):
        g = g.reset_index(drop=True)
        y, yp = g.intended.to_numpy(), g.rated_quadrant.to_numpy()
        n, x = len(g), int((y == yp).sum())
        lo, hi = wilson(x, n)
        boot_acc, boot_f1 = [], []
        cell_indices = [cell.index.to_numpy() for _, cell in g.groupby("emotion")]
        for _ in range(nboot):
            ix = np.concatenate(
                [
                    rng.choice(cell, size=len(cell), replace=True)
                    for cell in cell_indices
                ]
            )
            boot_acc.append(np.mean(y[ix] == yp[ix]))
            boot_f1.append(
                f1_score(
                    y[ix],
                    yp[ix],
                    labels=[1, 2, 3, 4],
                    average="weighted",
                    zero_division=0,
                )
            )
        rows.append(
            {
                "model": model,
                "n_clips": n,
                "correct": x,
                "accuracy": x / n,
                "wilson_low": lo,
                "wilson_high": hi,
                "stratified_bootstrap_low": np.quantile(boot_acc, 0.025),
                "stratified_bootstrap_high": np.quantile(boot_acc, 0.975),
                "above_chance_p_raw": stats.binomtest(
                    x, n, 0.25, alternative="greater"
                ).pvalue,
            }
        )
        frows.append(
            {
                "model": model,
                "n_clips": n,
                "weighted_f1": wf1(y, yp),
                "f1_bootstrap_low": np.quantile(boot_f1, 0.025),
                "f1_bootstrap_high": np.quantile(boot_f1, 0.975),
            }
        )
    a = pd.DataFrame(rows)
    a["above_chance_p_holm"] = holm(a.above_chance_p_raw)
    write(a, "accuracy_uncertainty.csv")
    write(pd.DataFrame(frows), "f1_uncertainty.csv")
    # Emotion-stratified label permutation: each pair preserves each emotion's
    # pooled outcome distribution and swaps system labels within emotion.
    pairrows = []
    for A, B in itertools.combinations(MODELS, 2):
        d = c[c.model.isin([A, B])]
        observed = (d.intended == d.rated_quadrant).astype(int).groupby(d.model).mean()
        diff = float(observed[A] - observed[B])
        null = np.empty(nperm)
        null_f1 = np.empty(nperm)
        by_e = []
        for _, g in d.groupby("emotion"):
            isA = (g.model == A).to_numpy()
            predicted = g.rated_quadrant.to_numpy()
            true = g.intended.to_numpy()
            by_e.append((isA, predicted, true))
        true_a = np.concatenate([true[isA] for isA, _, true in by_e])
        true_b = np.concatenate([true[~isA] for isA, _, true in by_e])
        for b in range(nperm):
            vals = {A: [], B: []}
            for isA, predicted, _ in by_e:
                # Shuffle actual predicted quadrant labels within emotion;
                # correctness bits are never permuted.
                shuffled = rng.permutation(predicted)
                vals[A].extend(shuffled[isA])
                vals[B].extend(shuffled[~isA])
            null[b] = np.mean(true_a == vals[A]) - np.mean(true_b == vals[B])
            null_f1[b] = wf1(true_a, vals[A]) - wf1(true_b, vals[B])
        f1_observed = wf1(
            d[d.model == A].intended, d[d.model == A].rated_quadrant
        ) - wf1(d[d.model == B].intended, d[d.model == B].rated_quadrant)
        acc_p = (np.sum(np.abs(null) >= abs(diff)) + 1) / (nperm + 1)
        f1_p = (np.sum(np.abs(null_f1) >= abs(f1_observed)) + 1) / (nperm + 1)
        pairrows.append(
            {
                "model_a": A,
                "model_b": B,
                "accuracy_diff_a_minus_b": diff,
                "p_value": acc_p,
                "accuracy_mc_se": np.sqrt(acc_p * (1 - acc_p) / nperm),
                "accuracy_mc_ci_low": max(
                    0, acc_p - 1.96 * np.sqrt(acc_p * (1 - acc_p) / nperm)
                ),
                "accuracy_mc_ci_high": min(
                    1, acc_p + 1.96 * np.sqrt(acc_p * (1 - acc_p) / nperm)
                ),
                "weighted_f1_diff_a_minus_b": f1_observed,
                "weighted_f1_p_value": f1_p,
                "weighted_f1_mc_se": np.sqrt(f1_p * (1 - f1_p) / nperm),
                "weighted_f1_mc_ci_low": max(
                    0, f1_p - 1.96 * np.sqrt(f1_p * (1 - f1_p) / nperm)
                ),
                "weighted_f1_mc_ci_high": min(
                    1, f1_p + 1.96 * np.sqrt(f1_p * (1 - f1_p) / nperm)
                ),
                "n_permutations": nperm,
                "stratification": "emotion; predicted quadrant labels permuted",
            }
        )
    p = pd.DataFrame(pairrows)
    p["p_holm"] = holm(p.p_value)
    p["weighted_f1_p_holm"] = holm(p.weighted_f1_p_value)
    p["primary_endpoint"] = "accuracy"
    p["secondary_weighted_f1"] = (
        "weighted F1; Holm correction within its 15-pair family"
    )
    write(p, "accuracy_pairwise_permutation.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=20000)
    args = parser.parse_args(argv)
    inputs = read_inputs()
    errors = validate(*inputs)
    if errors:
        raise ValueError("Input checks failed: " + ", ".join(errors))
    if args.check_only:
        print("Core input checks passed.")
        return
    ratings, clips = prepare(*inputs)
    rng = np.random.default_rng(args.seed)
    # This order preserves the published bootstrap/permutation random stream.
    for name, function, arguments in [
        ("agreement", reliability, (ratings, clips, rng, args.bootstrap)),
        (
            "factorial and centralization tests",
            robust_models,
            (clips, rng, args.bootstrap),
        ),
        (
            "accuracy",
            accuracy,
            (ratings, clips, rng, args.bootstrap, args.permutations),
        ),
    ]:
        print(f"Running {name}...", flush=True)
        function(*arguments)
    print(f"Wrote core statistics to {OUT}")


if __name__ == "__main__":
    main()
