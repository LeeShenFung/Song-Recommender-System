"""
Hybrid recommender.

Combines the two "views" of the problem into a single ranked list:

  content_score  -- how similar a candidate track's audio/tag profile is
                     to the tracks the user already likes (works even for
                     brand-new users -> solves cold start).
  cf_score       -- how much the collaborative-filtering latent-factor
                     model thinks this user will enjoy the track, learned
                     purely from the *pattern of what many users played*
                     (captures taste signal content features can't see,
                     e.g. two songs that "go together" for cultural /
                     social reasons despite sounding different).

Both scores are min-max normalised to [0, 1] independently (they live on
different, not-directly-comparable scales) and then combined with a
weighted sum controlled by `alpha`:

    hybrid_score = alpha * cf_score_norm + (1 - alpha) * content_score_norm

  alpha = 1.0  -> pure collaborative filtering
  alpha = 0.0  -> pure content-based filtering
  alpha = 0.5  -> balanced hybrid (default)

For a brand-new user (not present in the collaborative model at all,
i.e. true cold start) we fall back to alpha = 0 automatically, since
there is no collaborative signal to use yet.
"""

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import config
from src.content_based import ContentModel
from src.collaborative_filtering import CFModel, user_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def content_scores_for_profile(content_model: ContentModel, liked_track_ids: list) -> pd.DataFrame:
    """Score EVERY track in the catalogue against an averaged 'taste
    profile' vector built from liked_track_ids. Uses a direct sparse
    dot-product against the full feature matrix (fast: one sparse
    matrix-vector multiply over ~50k items), rather than the kNN index,
    so that a full ranked list is available for hybrid re-scoring.
    """
    rows = [content_model.track_id_to_row[t] for t in liked_track_ids
            if t in content_model.track_id_to_row]
    if not rows:
        raise ValueError("None of the provided track_ids were found in the content model.")

    profile_vector = content_model.feature_matrix[rows].mean(axis=0)
    profile_vector = sp.csr_matrix(profile_vector)

    scores = content_model.feature_matrix.dot(profile_vector.T).toarray().ravel()
    result = pd.DataFrame({
        "track_id": content_model.track_ids,
        "content_score": scores,
    })
    return result


def recommend_hybrid(
    content_model: ContentModel,
    cf_model: CFModel,
    user_id: str,
    top_n: int = 10,
    alpha: float = config.DEFAULT_ALPHA,
    seed_track_ids: list | None = None,
) -> pd.DataFrame:
    """Produce the final top-N hybrid recommendation list for a user.

    Parameters
    ----------
    user_id : str
        A user_id. If it exists in the collaborative model, both
        collaborative and content signals are used. If not (cold start),
        `seed_track_ids` must be supplied and alpha is forced to 0
        (content-only).
    seed_track_ids : list[str], optional
        Tracks the user says they like -- required for cold-start users,
        optional (ignored in favour of real history) for known users.
    """
    known_user = user_id in cf_model.user_id_to_row

    if not known_user and not seed_track_ids:
        raise ValueError(
            f"user_id '{user_id}' has no listening history and no seed_track_ids "
            "were provided -- cannot make a recommendation for a fully cold user."
        )

    # ---- content-based component -----------------------------------------
    if seed_track_ids:
        liked_tracks = seed_track_ids
    else:
        liked_tracks = list(cf_model.user_played_tracks.get(user_id, []))

    content_df = content_scores_for_profile(content_model, liked_tracks)

    if not known_user:
        # true cold start: no collaborative signal available at all
        alpha = 0.0
        merged = content_df.copy()
        merged["cf_score"] = 0.0
    else:
        cf_df = user_scores(cf_model, user_id, exclude_played=False)
        merged = content_df.merge(cf_df, on="track_id", how="left")
        merged["cf_score"] = merged["cf_score"].fillna(merged["cf_score"].min())

    # exclude tracks the user has already listened to / seeded with
    already_seen = set(liked_tracks)
    merged = merged[~merged["track_id"].isin(already_seen)]

    merged["content_score_norm"] = _minmax_norm(merged["content_score"])
    merged["cf_score_norm"] = _minmax_norm(merged["cf_score"])
    merged["hybrid_score"] = (
        alpha * merged["cf_score_norm"] + (1 - alpha) * merged["content_score_norm"]
    )

    result = merged.sort_values("hybrid_score", ascending=False).head(top_n).reset_index(drop=True)
    return result[["track_id", "hybrid_score", "content_score_norm", "cf_score_norm"]]
