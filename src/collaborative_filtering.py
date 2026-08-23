"""
Collaborative-filtering recommender.

The listening-history log gives us *implicit* feedback: a play-count per
(user, track) pair rather than an explicit 1-5 star rating. We turn this
into a user-item matrix and factorise it with Truncated SVD (a standard,
dependency-light latent-factor approach -- equivalent in spirit to
funk-SVD / matrix-factorisation recommenders such as those used in the
Netflix Prize, adapted here for implicit counts):

  1. log1p-transform play counts to reduce the influence of a handful of
     extremely high counts ("power users" / looped tracks).
  2. Build a sparse (n_users x n_tracks) matrix.
  3. Factorise it into U (n_users x k) and V (n_tracks x k) latent factor
     matrices with Truncated SVD.
  4. A user's predicted affinity for any track is simply the dot product
     of that user's latent vector and the track's latent vector.

This captures "people who listened to similar sets of songs also liked
..." style signal that content features alone cannot see.
"""

import pickle
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CFModel:
    user_factors: np.ndarray          # (n_users, k)
    item_factors: np.ndarray          # (n_tracks, k)
    user_id_to_row: dict
    row_to_user_id: np.ndarray
    track_id_to_col: dict
    col_to_track_id: np.ndarray
    user_played_tracks: dict          # user_id -> set(track_id) seen in training data


def _build_sparse_matrix(interactions_df: pd.DataFrame):
    user_ids = interactions_df["user_id"].unique()
    track_ids = interactions_df["track_id"].unique()

    user_id_to_row = {u: i for i, u in enumerate(user_ids)}
    track_id_to_col = {t: i for i, t in enumerate(track_ids)}

    rows = interactions_df["user_id"].map(user_id_to_row).values
    cols = interactions_df["track_id"].map(track_id_to_col).values
    # log1p dampens the effect of very large play counts (implicit-feedback
    # confidence weighting, à la Hu, Koren & Volinsky 2008)
    values = np.log1p(interactions_df["playcount"].values.astype(np.float32))

    matrix = sp.csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(track_ids)),
    )
    return matrix, user_id_to_row, track_id_to_col, user_ids, track_ids


def build_cf_model(interactions_df: pd.DataFrame, n_factors: int = config.N_LATENT_FACTORS) -> CFModel:
    logger.info("Building user-item matrix from %d interactions...", len(interactions_df))
    matrix, user_id_to_row, track_id_to_col, user_ids, track_ids = _build_sparse_matrix(interactions_df)
    logger.info("User-item matrix shape: %s (density %.4f%%)",
                matrix.shape, 100 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]))

    k = min(n_factors, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=config.RANDOM_SEED)
    user_factors = svd.fit_transform(matrix)          # (n_users, k)
    item_factors = svd.components_.T                  # (n_tracks, k)

    user_played_tracks = (
        interactions_df.groupby("user_id")["track_id"].apply(set).to_dict()
    )

    logger.info("Collaborative model trained with k=%d latent factors.", k)
    return CFModel(
        user_factors=user_factors,
        item_factors=item_factors,
        user_id_to_row=user_id_to_row,
        row_to_user_id=user_ids,
        track_id_to_col=track_id_to_col,
        col_to_track_id=track_ids,
        user_played_tracks=user_played_tracks,
    )


def save_cf_model(model: CFModel, path: str = config.CF_MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_cf_model(path: str = config.CF_MODEL_PATH) -> CFModel:
    with open(path, "rb") as f:
        return pickle.load(f)


def user_scores(model: CFModel, user_id: str, exclude_played: bool = True) -> pd.DataFrame:
    """Predicted collaborative-filtering affinity score for every track,
    for a given (known) user_id. Higher = more likely to enjoy.
    """
    if user_id not in model.user_id_to_row:
        raise KeyError(f"user_id {user_id} not found in collaborative model")

    row = model.user_id_to_row[user_id]
    scores = model.item_factors @ model.user_factors[row]  # (n_tracks,)

    result = pd.DataFrame({
        "track_id": model.col_to_track_id,
        "cf_score": scores,
    })

    if exclude_played:
        played = model.user_played_tracks.get(user_id, set())
        result = result[~result["track_id"].isin(played)]

    return result.reset_index(drop=True)


def recommend_for_user(model: CFModel, user_id: str, top_n: int = 10) -> pd.DataFrame:
    scores = user_scores(model, user_id)
    return scores.sort_values("cf_score", ascending=False).head(top_n).reset_index(drop=True)
