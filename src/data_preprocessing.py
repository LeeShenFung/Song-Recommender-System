"""
Data preprocessing pipeline.

This script/module is responsible for turning the two raw input files:

  data/raw/Music_Info.csv               (~50,000 tracks, content features)
  data/raw/User_Listening_History.csv   (user_id, track_id, playcount)

into clean, filtered, and appropriately sized artefacts that the
content-based and collaborative-filtering models can consume:

  data/processed/tracks_clean.csv
  data/processed/interactions_clean.csv

Run it directly with:

    python -m src.data_preprocessing

or import `run_preprocessing()` from app.py / a notebook.
"""

import os
import logging

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Music_Info.csv (content side)
# ---------------------------------------------------------------------------
def load_and_clean_tracks(path: str = config.MUSIC_INFO_PATH) -> pd.DataFrame:
    """Load Music_Info.csv and perform basic cleaning.

    Steps:
      - Drop exact duplicate track_ids (keep first).
      - Fill missing `genre` with 'Unknown' (28k tracks have no genre tag,
        but almost all of them still have free-text `tags`, which the
        content model also uses).
      - Fill missing `tags` with an empty string.
      - Clip/repair a few audio features that are occasionally out of the
        expected [0, 1] range due to upstream data issues.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find Music_Info.csv at {path}. "
            "Place the file in data/raw/ before running preprocessing."
        )

    logger.info("Loading Music_Info.csv from %s", path)
    df = pd.read_csv(path)

    before = len(df)
    df = df.drop_duplicates(subset="track_id", keep="first").reset_index(drop=True)
    logger.info("Dropped %d duplicate track_id rows", before - len(df))

    df["genre"] = df["genre"].fillna("Unknown")
    df["tags"] = df["tags"].fillna("")

    # A handful of feature columns can contain values slightly outside
    # [0, 1] due to upstream rounding; clip them defensively so scaling
    # later on behaves predictably.
    zero_one_cols = [
        "danceability", "energy", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence",
    ]
    for col in zero_one_cols:
        df[col] = df[col].clip(0.0, 1.0)

    keep_cols = [
        "track_id", "name", "artist", "tags", "genre", "year",
        *config.AUDIO_FEATURES,
    ]
    df = df[keep_cols]

    logger.info("Cleaned tracks table: %d rows, %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# User_Listening_History.csv (collaborative side)
# ---------------------------------------------------------------------------
def _stream_raw_interactions(tracks_df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Stream the raw listening-history file in bounded chunks instead of
    loading it all into memory at once.

    This is the key trick for a slow/low-RAM machine: pandas never sees
    the whole (potentially multi-million-row) file in one go. We read
    `config.CHUNK_SIZE` rows at a time, immediately drop rows whose
    track_id isn't in Music_Info (cheap, vectorised, shrinks each chunk
    right away), and stop early once we've either:

      - scanned config.MAX_RAW_ROWS_TO_SCAN raw rows, or
      - already seen config.MAX_USERS_SCAN_TARGET unique users,

    whichever happens first. Both limits are configurable in config.py --
    raise them if your machine can handle a bigger sample.
    """
    valid_tracks = set(tracks_df["track_id"])
    dtypes = {"track_id": "str", "user_id": "str", "playcount": "int32"}

    kept_chunks = []
    rows_scanned = 0
    seen_users: set = set()

    reader = pd.read_csv(path, dtype=dtypes, chunksize=config.CHUNK_SIZE)
    for chunk_idx, chunk in enumerate(reader):
        rows_scanned += len(chunk)
        chunk = chunk[chunk["track_id"].isin(valid_tracks)]
        kept_chunks.append(chunk)
        seen_users.update(chunk["user_id"].unique())

        logger.info(
            "Scanned chunk %d (%d raw rows so far, %d kept, %d unique users so far)...",
            chunk_idx + 1, rows_scanned, sum(len(c) for c in kept_chunks), len(seen_users),
        )

        if rows_scanned >= config.MAX_RAW_ROWS_TO_SCAN:
            logger.info("Reached MAX_RAW_ROWS_TO_SCAN (%d) -- stopping scan early.",
                        config.MAX_RAW_ROWS_TO_SCAN)
            break
        if len(seen_users) >= config.MAX_USERS_SCAN_TARGET:
            logger.info("Reached MAX_USERS_SCAN_TARGET (%d unique users) -- stopping scan early.",
                        config.MAX_USERS_SCAN_TARGET)
            break

    df = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame(
        columns=["track_id", "user_id", "playcount"])
    return df


def load_and_filter_interactions(
    tracks_df: pd.DataFrame,
    path: str = config.LISTENING_HISTORY_PATH,
) -> pd.DataFrame:
    """Load, filter, and down-sample the raw listening-history log.

    The raw file can contain several million rows across hundreds of
    thousands of users, which is slow/heavy to fully load on a modest
    laptop. We stream it in bounded chunks (see `_stream_raw_interactions`)
    and then reduce the scanned rows to a dense, reproducible subset
    following the strategy documented in `config.py`:

      1. keep only interactions with a track_id present in Music_Info
         (already applied while streaming, see above)
      2. drop very sparse users / tracks
      3. randomly cap the number of unique users to config.MAX_USERS
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find User_Listening_History.csv at {path}. "
            "Download it from the Kaggle 'Music Recommender System' dataset "
            "(Music Info + User Listening History) and place it in data/raw/ "
            "-- expected columns: track_id, user_id, playcount."
        )

    logger.info("Streaming User_Listening_History.csv from %s in chunks of %d rows "
                "(capped at %d rows / %d users scanned)...",
                path, config.CHUNK_SIZE, config.MAX_RAW_ROWS_TO_SCAN, config.MAX_USERS_SCAN_TARGET)
    df = _stream_raw_interactions(tracks_df, path)
    logger.info("Scanned interactions after track_id filter: %d rows, %d users, %d tracks",
                len(df), df["user_id"].nunique(), df["track_id"].nunique())

    # Prune sparse users/tracks (two passes is enough in practice; a full
    # fixed-point loop is not necessary for this course prototype).
    for _ in range(2):
        user_counts = df.groupby("user_id")["track_id"].transform("count")
        df = df[user_counts >= config.MIN_INTERACTIONS_PER_USER]

        track_counts = df.groupby("track_id")["user_id"].transform("count")
        df = df[track_counts >= config.MIN_INTERACTIONS_PER_TRACK]

    # Cap number of unique users for a manageable, fast-to-train demo.
    unique_users = df["user_id"].unique()
    if len(unique_users) > config.MAX_USERS:
        rng = np.random.default_rng(config.RANDOM_SEED)
        sampled_users = rng.choice(unique_users, size=config.MAX_USERS, replace=False)
        df = df[df["user_id"].isin(sampled_users)]

    df = df.reset_index(drop=True)
    logger.info("Final filtered interactions: %d rows, %d users, %d tracks",
                len(df), df["user_id"].nunique(), df["track_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_preprocessing(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline and cache results to data/processed/.

    If cached CSV files already exist and `force=False`, they are
    loaded directly instead of being recomputed (much faster for repeated
    Streamlit app reloads).
    """
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    if (not force
            and os.path.exists(config.TRACKS_CLEAN_PATH)
            and os.path.exists(config.INTERACTIONS_CLEAN_PATH)):
        logger.info("Loading cached processed data.")
        tracks_df = pd.read_csv(config.TRACKS_CLEAN_PATH)
        interactions_df = pd.read_csv(config.INTERACTIONS_CLEAN_PATH, dtype={"track_id": "str", "user_id": "str", "playcount": "int32"})
        return tracks_df, interactions_df

    tracks_df = load_and_clean_tracks()
    interactions_df = load_and_filter_interactions(tracks_df)

    tracks_df.to_csv(config.TRACKS_CLEAN_PATH, index=False)
    interactions_df.to_csv(config.INTERACTIONS_CLEAN_PATH, index=False)
    logger.info("Saved processed artefacts to %s", config.PROCESSED_DIR)

    return tracks_df, interactions_df


if __name__ == "__main__":
    run_preprocessing(force=True)
