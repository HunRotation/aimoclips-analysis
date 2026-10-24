"""Load the public data used for manuscript CLAP/FAD and clustered regressions.

CLAP/FAD use exact clip means reconstructed from individual ratings. Other
analyses retain the released rounded means where originally used.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "ratings": ROOT / "data/ratings_metadata.csv",
    "clips": ROOT / "data/clips_metadata.csv",
    "clap": ROOT / "data/clap_scores.csv",
    "fad": ROOT / "data/distribution_metrics.csv",
}
TRAITS = ("valence", "arousal")


def read_data():
    data = {name: pd.read_csv(path) for name, path in INPUTS.items()}
    r, meta, clap, fad = (data[name] for name in INPUTS)
    assert len(r) == 6162 and r.user.nunique() == 111
    assert not r.duplicated(["user", "audio_file"]).any()
    assert len(meta) == 991 and meta.audio_file.is_unique
    assert len(clap) == 991 and clap.audio_file.is_unique
    assert len(fad) == 6 and fad.model.is_unique
    assert set(r.audio_file) == set(meta.audio_file) == set(clap.audio_file)
    assert set(meta.model) == set(fad.model)
    assert meta.model.nunique() == 6 and meta.emotion.nunique() == 12
    assert r[list(TRAITS)].notna().all().all()
    assert r[list(TRAITS)].apply(lambda s: s.between(1, 9)).all().all()
    c = (
        r.groupby("audio_file", sort=True)
        .agg(
            valence=("valence", "mean"),
            arousal=("arousal", "mean"),
            n_ratings=("valence", "size"),
        )
        .reset_index()
        .merge(
            meta[["audio_file", "model", "emotion"]],
            on="audio_file",
            validate="one_to_one",
        )
    )
    comparison = c.merge(
        meta[["audio_file", *TRAITS]],
        on="audio_file",
        suffixes=("_exact", "_metadata"),
        validate="one_to_one",
    )
    max_rounding = {
        trait: float(
            np.max(
                np.abs(comparison[f"{trait}_exact"] - comparison[f"{trait}_metadata"])
            )
        )
        for trait in TRAITS
    }
    assert max(max_rounding.values()) < 0.000501, max_rounding
    c = c.merge(
        clap[["audio_file", "clap_score"]], on="audio_file", validate="one_to_one"
    )
    assert c.clap_score.notna().all() and fad.fad_score.notna().all()
    r = r[["user", "audio_file", *TRAITS]].merge(
        c[["audio_file", "model", "emotion", "clap_score", "n_ratings"]],
        on="audio_file",
        validate="many_to_one",
    )
    r["clip_weight"] = 1.0 / r.n_ratings
    np.testing.assert_allclose(r.groupby("audio_file").clip_weight.sum(), 1)
    return r, c, fad, max_rounding


def manual_cluster_covariance(fit, groups):
    """Independent direct sandwich construction for validating weighted code."""
    scores = fit.model.wexog * np.asarray(fit.wresid)[:, None]
    bread = np.asarray(fit.normalized_cov_params)
    group_codes, unique = pd.factorize(groups)
    totals = np.zeros((len(unique), scores.shape[1]))
    np.add.at(totals, group_codes, scores)
    n, k = scores.shape
    correction = len(unique) / (len(unique) - 1) * (n - 1) / (n - k)
    return correction * bread @ (totals.T @ totals) @ bread
