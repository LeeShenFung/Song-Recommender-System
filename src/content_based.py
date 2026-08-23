"""
Content-based recommender.

Builds a feature vector for every track from two sources:

  1. TEXT block  -- TF-IDF over `tags` (free-text, comma separated, e.g.
     "rock, alternative, 90s, grunge") concatenated with `genre`.
  2. AUDIO block -- the 9 Spotify audio features (danceability, energy,
     loudness, ...), min-max scaled to [0, 1] so no single feature
     dominates the distance metric.

Both blocks are weighted (see config.TEXT_WEIGHT / config.AUDIO_WEIGHT)
and horizontally stacked into one sparse feature matrix. Similarity
between tracks is then just cosine distance in that combined space,
which we look up efficiently with scikit-learn's NearestNeighbors
(brute-force cosine, which is exact and works well on ~10-50k items --
no need to materialise the full pairwise similarity matrix, which for
50k tracks would be ~2.5 billion entries).

This model needs NO listening-history data at all, so it is also what
we fall back on for brand-new ("cold-start") users.
"""

import pickle
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ContentModel:
    """Container for everything the content-based recommender needs at
    inference time."""
    nn_index: NearestNeighbors
    feature_matrix: sp.csr_matrix
    track_ids: np.ndarray            # row i -> track_id
    track_id_to_row: dict            # track_id -> row i


def build_content_model(tracks_df: pd.DataFrame) -> ContentModel:
    """Fit the TF-IDF + scaler + NearestNeighbors pipeline on tracks_df."""
    logger.info("Building content-based feature space for %d tracks...", len(tracks_df))

    # --- text block -------------------------------------------------------
    text_corpus = (tracks_df["tags"].fillna("") + " " + tracks_df["genre"].fillna(""))
    tfidf = TfidfVectorizer(
        token_pattern=r"[^,\s]+",  # tags are comma/space separated tokens
        max_features=3000,
        min_df=2,
    )
    text_matrix = tfidf.fit_transform(text_corpus)  # sparse, already L2-normalised rows

    # --- audio block --------------------------------------------------------
    scaler = MinMaxScaler()
    audio_matrix = scaler.fit_transform(tracks_df[config.AUDIO_FEATURES].values)
    audio_matrix = sp.csr_matrix(audio_matrix)
    # normalise each row to unit L2 norm so it is comparable to the tf-idf block
    row_norms = np.sqrt(audio_matrix.multiply(audio_matrix).sum(axis=1))
    row_norms[row_norms == 0] = 1.0
    audio_matrix = audio_matrix.multiply(1.0 / row_norms).tocsr()

    # --- combine with configured weights ------------------------------------
    combined = sp.hstack([
        text_matrix * config.TEXT_WEIGHT,
        audio_matrix * config.AUDIO_WEIGHT,
    ]).tocsr()

    nn_index = NearestNeighbors(
        n_neighbors=min(config.TOP_K_NEIGHBORS + 1, combined.shape[0]),
        metric="cosine",
        algorithm="brute",
    )
    nn_index.fit(combined)

    track_ids = tracks_df["track_id"].values
    track_id_to_row = {tid: i for i, tid in enumerate(track_ids)}

    logger.info("Content model ready. Feature matrix shape: %s", combined.shape)
    return ContentModel(
        nn_index=nn_index,
        feature_matrix=combined,
        track_ids=track_ids,
        track_id_to_row=track_id_to_row,
    )


def save_content_model(model: ContentModel, path: str = config.CONTENT_MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_content_model(path: str = config.CONTENT_MODEL_PATH) -> ContentModel:
    with open(path, "rb") as f:
        return pickle.load(f)


def similar_tracks(model: ContentModel, track_id: str, top_n: int = 10) -> pd.DataFrame:
    """Return the top_n most content-similar tracks to a given track_id."""
    if track_id not in model.track_id_to_row:
        raise KeyError(f"track_id {track_id} not found in content model")

    row = model.track_id_to_row[track_id]
    n_query = min(top_n + 1, model.feature_matrix.shape[0])
    distances, indices = model.nn_index.kneighbors(
        model.feature_matrix[row], n_neighbors=n_query
    )
    distances, indices = distances[0], indices[0]

    result = pd.DataFrame({
        "track_id": model.track_ids[indices],
        "content_score": 1.0 - distances,  # cosine similarity = 1 - cosine distance
    })
    # drop the query track itself
    result = result[result["track_id"] != track_id]
    return result.head(top_n).reset_index(drop=True)


def recommend_from_profile(model: ContentModel, liked_track_ids: list, top_n: int = 10) -> pd.DataFrame:
    """Cold-start style recommendation: average the feature vectors of a
    list of tracks the user says they like, then find nearest neighbours
    of that averaged 'taste profile' vector.

    Used both for brand-new users (no listening history at all) and as
    the content-based half of the hybrid score for existing users.
    """
    rows = [model.track_id_to_row[t] for t in liked_track_ids if t in model.track_id_to_row]
    if not rows:
        raise ValueError("None of the provided track_ids were found in the content model.")

    profile_vector = model.feature_matrix[rows].mean(axis=0)
    profile_vector = sp.csr_matrix(profile_vector)

    n_query = min(top_n + len(rows), model.feature_matrix.shape[0])
    distances, indices = model.nn_index.kneighbors(profile_vector, n_neighbors=n_query)
    distances, indices = distances[0], indices[0]

    result = pd.DataFrame({
        "track_id": model.track_ids[indices],
        "content_score": 1.0 - distances,
    })
    result = result[~result["track_id"].isin(liked_track_ids)]
    return result.head(top_n).reset_index(drop=True)
