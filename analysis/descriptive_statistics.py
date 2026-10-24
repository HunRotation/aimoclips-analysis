"""Reproduce Tables 3–4, S1–S2/S5–S6 and the Figure 1/S5 comparisons.

Exact clip means and quadrant rules retain the original descriptive analysis.
The separate inferential scripts retain their original rounded metadata inputs.
"""

from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
MODELS = ["AudioLDM2", "MusicGen", "Mustango", "StableAudio", "Suno", "Udio"]


def get_intended_quadrant(emotion):
    """Map the twelve prompt words to their intended quadrants."""
    emotion = emotion.lower().strip()
    if emotion in ["happy", "excited", "energetic"]:
        return 1
    elif emotion in ["angry", "scared", "anxious"]:
        return 2
    elif emotion in ["sad", "gloomy", "dull"]:
        return 3
    elif emotion in ["relaxed", "calm", "tranquil"]:
        return 4
    else:
        return 0


def get_rated_quadrant(valence, arousal):
    """Classify exact clip means; either coordinate at five is a mismatch."""
    if valence > 5 and arousal > 5:
        return 1
    elif valence < 5 and arousal > 5:
        return 2
    elif valence < 5 and arousal < 5:
        return 3
    elif valence > 5 and arousal < 5:
        return 4
    else:
        return 0


def prepare_clips():
    ratings = pd.read_csv(DATA / "ratings_metadata.csv")
    metadata = pd.read_csv(DATA / "clips_metadata.csv")
    ratings = ratings.merge(
        metadata[["audio_file", "model", "emotion"]],
        on="audio_file",
        validate="many_to_one",
    )
    clips = (
        ratings.groupby(["audio_file", "model", "emotion"])[["valence", "arousal"]]
        .mean()
        .reset_index()
    )
    clips["intended_quadrant"] = clips.emotion.apply(get_intended_quadrant)
    clips["rated_quadrant"] = clips.apply(
        lambda row: get_rated_quadrant(row.valence, row.arousal), axis=1
    )
    assert len(clips) == 991 and len(ratings) == 6162
    assert ratings.user.nunique() == 111
    assert (clips.intended_quadrant == clips.rated_quadrant).sum() == 406
    clips.to_csv(OUT / "clip_quadrants.csv", index=False)
    return ratings, clips


def categorical_results(clips):
    rows, matrices = [], []
    for model, group in [("Overall", clips), *list(clips.groupby("model"))]:
        true, predicted = group.intended_quadrant, group.rated_quadrant
        report = classification_report(
            true, predicted, labels=[1, 2, 3, 4, 0], zero_division=0, output_dict=True
        )
        row = {
            "model": model,
            "n_clips": len(group),
            "correct": int((true == predicted).sum()),
            "accuracy": float((true == predicted).mean()),
            "weighted_f1": f1_score(
                true, predicted, average="weighted", zero_division=0
            ),
        }
        for quadrant in [1, 2, 3, 4]:
            for metric in ["precision", "recall", "f1-score", "support"]:
                row[f"q{quadrant}_{metric}"] = report[str(quadrant)][metric]
        rows.append(row)
        matrix = confusion_matrix(true, predicted, labels=[1, 2, 3, 4, 0])
        for i, intended in enumerate([1, 2, 3, 4]):
            for j, rated in enumerate([1, 2, 3, 4, 0]):
                matrices.append(
                    dict(
                        model=model,
                        intended_quadrant=intended,
                        rated_quadrant=rated,
                        count=matrix[i, j],
                    )
                )
    pd.DataFrame(rows).to_csv(OUT / "categorical_summary.csv", index=False)
    pd.DataFrame(matrices).to_csv(OUT / "confusion_counts.csv", index=False)
    correct = int((clips.intended_quadrant == clips.rated_quadrant).sum())
    count = len(clips)
    z, p = proportions_ztest(correct, count, value=0.25, alternative="larger")
    pd.DataFrame(
        [
            dict(
                correct=correct,
                n_clips=count,
                accuracy=correct / count,
                z=z,
                p_one_sided=p,
            )
        ]
    ).to_csv(OUT / "overall_accuracy_test.csv", index=False)


def descriptive_results(ratings, clips):
    words = pd.read_csv(DATA / "emotion_intents_metadata.csv")
    c = clips.merge(words, on="emotion", validate="many_to_one")
    c["distance"] = np.sqrt(
        (c.valence - c.gt_valence) ** 2 + (c.arousal - c.gt_arousal) ** 2
    )
    c["distance_center"] = np.sqrt((c.valence - 5) ** 2 + (c.arousal - 5) ** 2)
    c["reference_center"] = np.sqrt((c.gt_valence - 5) ** 2 + (c.gt_arousal - 5) ** 2)
    c["valence_deviation"] = c.valence - c.gt_valence
    c["arousal_deviation"] = c.arousal - c.gt_arousal
    c.to_csv(OUT / "descriptive_clip_data.csv", index=False)
    for groups, name in [
        (["intended_quadrant"], "quadrant_descriptives.csv"),
        (["model", "intended_quadrant"], "system_quadrant_descriptives.csv"),
        (["model"], "system_descriptives.csv"),
    ]:
        table = c.groupby(groups).agg(
            n_clips=("audio_file", "size"),
            valence_mean=("valence", "mean"),
            valence_sd=("valence", "std"),
            arousal_mean=("arousal", "mean"),
            arousal_sd=("arousal", "std"),
            distance_mean=("distance", "mean"),
            distance_sd=("distance", "std"),
            center_mean=("distance_center", "mean"),
            center_sd=("distance_center", "std"),
            reference_center_mean=("reference_center", "mean"),
            reference_center_sd=("reference_center", "std"),
        )
        table.to_csv(OUT / name)
    overall = {
        "n_clips": len(c),
        "n_ratings": len(ratings),
        "n_participants": ratings.user.nunique(),
    }
    for column in [
        "valence",
        "arousal",
        "distance",
        "distance_center",
        "reference_center",
    ]:
        overall[column + "_mean"] = c[column].mean()
        overall[column + "_sd"] = c[column].std()
    pd.DataFrame([overall]).to_csv(OUT / "overall_descriptives.csv", index=False)
    # Figure 2 and S5 use ordinary clip-level errors, distinct from S8's robust CIs.
    deviations = []
    for model, group in c.groupby("model"):
        for trait in ["valence", "arousal"]:
            values = group[f"{trait}_deviation"]
            mean, sd, count = values.mean(), values.std(), len(values)
            se = sd / np.sqrt(count)
            half_width = stats.t.ppf(0.975, count - 1) * se
            deviations.append(
                dict(
                    model=model,
                    outcome=trait,
                    n_clips=count,
                    mean=mean,
                    sd=sd,
                    se=se,
                    ci_low=mean - half_width,
                    ci_high=mean + half_width,
                )
            )
    pd.DataFrame(deviations).to_csv(OUT / "figure_deviation_summary.csv", index=False)
    ratings.groupby("audio_file").size().value_counts().sort_index().rename_axis(
        "ratings_per_clip"
    ).to_csv(OUT / "rating_count_distribution.csv", header=["n_clips"])
    return c


def tukey_results(c):
    rows = []
    result = pairwise_tukeyhsd(c.distance, c.quadrant, alpha=0.05)
    for k, (i, j) in enumerate(
        itertools.combinations(range(len(result.groupsunique)), 2)
    ):
        rows.append(
            dict(
                figure="Figure 1B",
                outcome="distance",
                group_a=result.groupsunique[i],
                group_b=result.groupsunique[j],
                difference_b_minus_a=result.meandiffs[k],
                p_value=result.pvalues[k],
                ci_low=result.confint[k, 0],
                ci_high=result.confint[k, 1],
            )
        )
    for trait in ["valence", "arousal"]:
        groups = [
            c.loc[c.model == model, f"{trait}_deviation"].to_numpy() for model in MODELS
        ]
        result = stats.tukey_hsd(*groups)
        intervals = result.confidence_interval()
        for i, j in itertools.combinations(range(len(MODELS)), 2):
            rows.append(
                dict(
                    figure="S5 Fig",
                    outcome=trait,
                    group_a=MODELS[i],
                    group_b=MODELS[j],
                    difference_b_minus_a=-result.statistic[i, j],
                    p_value=result.pvalue[i, j],
                    ci_low=-intervals.high[i, j],
                    ci_high=-intervals.low[i, j],
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "tukey_comparisons.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ratings, clips = prepare_clips()
    categorical_results(clips)
    c = descriptive_results(ratings, clips)
    tukey_results(c)
    print("Descriptive statistics and quadrant inputs complete.")


if __name__ == "__main__":
    main()
