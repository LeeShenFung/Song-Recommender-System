"""
Downsample a very large raw `User_Listening_History.csv` (too big to load
into memory / too slow to process in one go on a modest laptop) into a
smaller, dense, ready-to-use CSV -- WITHOUT ever loading the whole file
into memory at once.

How it works (2 passes over the file, streamed in chunks, so peak memory
usage stays roughly constant no matter how many GB the raw file is):

  PASS 1 (counting):
    Read the raw file in chunks of `config.CHUNK_SIZE` rows. For each
    chunk, keep only rows whose track_id exists in Music_Info.csv, then
    accumulate running totals of "plays per user" and "plays per track"
    in two plain Python dicts.

  Decide which users/tracks qualify:
    - tracks with >= MIN_INTERACTIONS_PER_TRACK total plays
    - users with  >= MIN_INTERACTIONS_PER_USER total plays
    - if there are still more qualifying users than MAX_USERS, randomly
      sample MAX_USERS of them (fixed seed -> reproducible)

  PASS 2 (writing):
    Read the raw file again in chunks. For each chunk, keep only rows
    where BOTH the user and the track qualify, and append them straight
    to the small output CSV on disk (never held fully in memory).

Usage
-----
1. Put your full raw file at: data/raw/User_Listening_History.csv
2. Run:
       python -m src.downsample_large_history
3. The script writes the small result to:
       data/raw/User_Listening_History_sampled.csv
4. Replace the big file with the small one (or just point
   config.LISTENING_HISTORY_PATH at the sampled file) before running
   `streamlit run app.py` / `python -m src.data_preprocessing`.

Tuning for a slow machine
--------------------------
- Lower `config.CHUNK_SIZE` (e.g. to 100_000) if you run out of RAM while
  reading -- it only affects how many rows are read per disk read, not
  correctness.
- Lower `config.MAX_USERS` / raise `MIN_INTERACTIONS_PER_USER` /
  `MIN_INTERACTIONS_PER_TRACK` in `src/config.py` for an even smaller,
  faster-to-train final dataset.
"""

import os
import csv
import time
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_valid_track_ids() -> set:
    tracks = pd.read_csv(config.MUSIC_INFO_PATH, usecols=["track_id"])
    return set(tracks["track_id"])


def _count_pass(input_path: str, valid_tracks: set, chunk_size: int):
    """PASS 1: stream the file once, accumulate per-user and per-track
    total play counts (only for rows whose track is in valid_tracks)."""
    user_counts = defaultdict(int)
    track_counts = defaultdict(int)

    t0 = time.time()
    rows_seen = 0
    for i, chunk in enumerate(pd.read_csv(
            input_path,
            usecols=["track_id", "user_id", "playcount"],
            dtype={"track_id": "str", "user_id": "str", "playcount": "int32"},
            chunksize=chunk_size,
    )):
        chunk = chunk[chunk["track_id"].isin(valid_tracks)]
        rows_seen += len(chunk)

        for uid, cnt in chunk.groupby("user_id").size().items():
            user_counts[uid] += cnt
        for tid, cnt in chunk.groupby("track_id").size().items():
            track_counts[tid] += cnt

        if (i + 1) % 5 == 0:
            logger.info("Pass 1: processed %d chunks (%d valid rows so far, %.1fs elapsed)",
                        i + 1, rows_seen, time.time() - t0)

    logger.info("Pass 1 complete: %d valid rows, %d unique users, %d unique tracks (%.1fs)",
                rows_seen, len(user_counts), len(track_counts), time.time() - t0)
    return user_counts, track_counts


def _write_pass(input_path: str, output_path: str, qualifying_tracks: set,
                 sampled_users: set, chunk_size: int):
    """PASS 2: stream the file again, keep only rows for qualifying
    tracks AND sampled users, append them straight to disk."""
    if os.path.exists(output_path):
        os.remove(output_path)

    t0 = time.time()
    rows_written = 0
    header_written = False
    for i, chunk in enumerate(pd.read_csv(
            input_path,
            usecols=["track_id", "user_id", "playcount"],
            dtype={"track_id": "str", "user_id": "str", "playcount": "int32"},
            chunksize=chunk_size,
    )):
        chunk = chunk[
            chunk["track_id"].isin(qualifying_tracks) & chunk["user_id"].isin(sampled_users)
        ]
        if len(chunk) == 0:
            continue

        chunk.to_csv(
            output_path,
            mode="a",
            header=not header_written,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )
        header_written = True
        rows_written += len(chunk)

        if (i + 1) % 5 == 0:
            logger.info("Pass 2: processed %d chunks (%d rows written so far, %.1fs elapsed)",
                        i + 1, rows_written, time.time() - t0)

    logger.info("Pass 2 complete: %d rows written to %s (%.1fs)",
                rows_written, output_path, time.time() - t0)


def downsample(
    input_path: str = config.LISTENING_HISTORY_PATH,
    output_path: str = config.LISTENING_HISTORY_SAMPLED_PATH,
    chunk_size: int = config.CHUNK_SIZE,
    min_interactions_per_user: int = config.MIN_INTERACTIONS_PER_USER,
    min_interactions_per_track: int = config.MIN_INTERACTIONS_PER_TRACK,
    max_users: int = config.MAX_USERS,
    seed: int = config.RANDOM_SEED,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find {input_path}. Put your full raw "
            "User_Listening_History.csv there first."
        )

    logger.info("Loading valid track_id set from Music_Info.csv...")
    valid_tracks = _load_valid_track_ids()
    logger.info("%d valid tracks in Music_Info.csv", len(valid_tracks))

    user_counts, track_counts = _count_pass(input_path, valid_tracks, chunk_size)

    qualifying_tracks = {t for t, c in track_counts.items() if c >= min_interactions_per_track}
    qualifying_users = [u for u, c in user_counts.items() if c >= min_interactions_per_user]
    logger.info("%d tracks and %d users pass the minimum-interaction thresholds",
                len(qualifying_tracks), len(qualifying_users))

    if len(qualifying_users) > max_users:
        rng = np.random.default_rng(seed)
        qualifying_users = list(rng.choice(qualifying_users, size=max_users, replace=False))
        logger.info("Randomly sampled down to %d users (seed=%d)", max_users, seed)

    sampled_users = set(qualifying_users)

    _write_pass(input_path, output_path, qualifying_tracks, sampled_users, chunk_size)

    logger.info(
        "\nDone. Small file ready at: %s\n"
        "Next step -- do ONE of the following:\n"
        "  (a) delete/rename the big data/raw/User_Listening_History.csv and rename\n"
        "      %s to data/raw/User_Listening_History.csv, or\n"
        "  (b) edit LISTENING_HISTORY_PATH in src/config.py to point at %s directly.\n"
        "Then run: streamlit run app.py",
        output_path, os.path.basename(output_path), output_path,
    )


if __name__ == "__main__":
    downsample()
