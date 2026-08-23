"""
Evaluation.

We use a standard implicit-feedback top-N evaluation protocol:

  1. Leave-N-out split: for every user with enough history, randomly hold
     out `config.TEST_HOLDOUT_PER_USER` of their played tracks as the
     test set; everything else stays in the training set.
  2. Train the collaborative-filtering model (and the content model,
     which needs no interaction data) on the training set only.
  3. For each evaluated user, generate a top-N recommendation list using
     ONLY training data, then check how many of the held-out test tracks
     appear in that list.
  4. Aggregate into Precision@N, Recall@N and F1@N, averaged across
     users. This lets us directly compare content-only (alpha=0),
     collaborative-only (alpha=1), and hybrid (alpha=0.5) as required by
     the assignment ("compare the results of different methods").

Precision@N = (# relevant items in top-N) / N
Recall@N    = (# relevant items in top-N) / (# held-out relevant items)
F1@N        = harmonic mean of the two
"""

import logging

import numpy as np
import pandas as pd

from src import config
from src.content_based import ContentModel
from src.collaborative_filtering import CFModel, build_cf_model, user_scores
from src.hybrid_recommender import content_scores_for_profile, _minmax_norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def leave_n_out_split(interactions_df: pd.DataFrame, n: int = config.TEST_HOLDOUT_PER_USER,
                       seed: int = config.RANDOM_SEED):
    """Split interactions into train/test with N held-out plays per
    eligible user (users need > n interactions to be split at all;
    users with too little history stay entirely in train)."""
    rng = np.random.default_rng(seed)
    test_rows = []

    train_frames = []
    for user_id, group in interactions_df.groupby("user_id"):
        if len(group) <= n:
            train_frames.append(group)
            continue
        test_idx = rng.choice(group.index.values, size=n, replace=False)
        test_rows.append(group.loc[test_idx])
        train_frames.append(group.drop(index=test_idx))

    train_df = pd.concat(train_frames).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame(
        columns=interactions_df.columns)

    logger.info("Leave-%d-out split: %d train rows, %d test rows, %d testable users",
                n, len(train_df), len(test_df), test_df["user_id"].nunique())
    return train_df, test_df


def _topn_for_method(content_model: ContentModel, cf_model: CFModel, user_id: str,
                      alpha: float, top_n: int) -> set:
    """Return the set of track_ids in the top-N list for one user under a
    given alpha (0=content-only, 1=CF-only, else hybrid)."""
    liked_tracks = list(cf_model.user_played_tracks.get(user_id, []))
    if not liked_tracks:
        return set()

    if alpha <= 0.0:
        scores = content_scores_for_profile(content_model, liked_tracks)
        scores = scores.rename(columns={"content_score": "score"})
    elif alpha >= 1.0:
        scores = user_scores(cf_model, user_id, exclude_played=False)
        scores = scores.rename(columns={"cf_score": "score"})
    else:
        content_df = content_scores_for_profile(content_model, liked_tracks)
        cf_df = user_scores(cf_model, user_id, exclude_played=False)
        merged = content_df.merge(cf_df, on="track_id", how="left")
        merged["cf_score"] = merged["cf_score"].fillna(merged["cf_score"].min())
        merged["score"] = (
            alpha * _minmax_norm(merged["cf_score"]) + (1 - alpha) * _minmax_norm(merged["content_score"])
        )
        scores = merged

    scores = scores[~scores["track_id"].isin(liked_tracks)]
    top = scores.sort_values("score", ascending=False).head(top_n)
    return set(top["track_id"])


def evaluate_method(content_model: ContentModel, cf_model: CFModel, test_df: pd.DataFrame,
                     alpha: float, top_n: int = config.TOP_N_EVAL,
                     max_users: int = 300, seed: int = config.RANDOM_SEED) -> dict:
    """Compute mean Precision@N / Recall@N / F1@N for one method (alpha)
    over a sample of test users."""
    test_users = test_df["user_id"].unique()
    # cap number of users evaluated to keep runtime reasonable in the app
    if len(test_users) > max_users:
        rng = np.random.default_rng(seed)
        test_users = rng.choice(test_users, size=max_users, replace=False)

    precisions, recalls = [], []
    for user_id in test_users:
        if user_id not in cf_model.user_id_to_row:
            continue  # user fell entirely into train (too few interactions)
        relevant = set(test_df.loc[test_df["user_id"] == user_id, "track_id"])
        if not relevant:
            continue
        recommended = _topn_for_method(content_model, cf_model, user_id, alpha, top_n)
        if not recommended:
            continue
        hits = len(relevant & recommended)
        precisions.append(hits / top_n)
        recalls.append(hits / len(relevant))

    precision = float(np.mean(precisions)) if precisions else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "alpha": alpha,
        "precision_at_n": precision,
        "recall_at_n": recall,
        "f1_at_n": f1,
        "n_users_evaluated": len(precisions),
    }


def compare_methods(content_model: ContentModel, cf_model: CFModel, test_df: pd.DataFrame,
                     top_n: int = config.TOP_N_EVAL, max_users: int = 300) -> pd.DataFrame:
    """Run evaluate_method for content-only, CF-only and hybrid, and
    return a tidy comparison table -- used both in the Streamlit app and
    to justify the choice of algorithm in the written documentation."""
    rows = []
    for label, alpha in [("Content-based only", 0.0),
                          ("Collaborative only", 1.0),
                          ("Hybrid (alpha=0.5)", 0.5)]:
        metrics = evaluate_method(content_model, cf_model, test_df, alpha, top_n, max_users)
        metrics["method"] = label
        rows.append(metrics)
    return pd.DataFrame(rows)[["method", "alpha", "precision_at_n", "recall_at_n",
                                "f1_at_n", "n_users_evaluated"]]


if __name__ == "__main__":
    from src.data_preprocessing import run_preprocessing
    from src.content_based import build_content_model

    tracks_df, interactions_df = run_preprocessing()
    train_df, test_df = leave_n_out_split(interactions_df)

    content_model = build_content_model(tracks_df)
    cf_model = build_cf_model(train_df)

    results = compare_methods(content_model, cf_model, test_df)
    print(results.to_string(index=False))
