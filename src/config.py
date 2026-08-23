"""
Central configuration for the Hybrid Music Recommender System.

All file paths and tunable constants live here so the rest of the
codebase never hard-codes a path or a magic number.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MUSIC_INFO_PATH = os.path.join(RAW_DIR, "Music_Info.csv")
LISTENING_HISTORY_PATH = os.path.join(RAW_DIR, "User_Listening_History.csv")

# Processed artefacts (created by data_preprocessing.py).
# Plain CSV is used instead of Parquet so the project has no dependency
# on pyarrow/fastparquet -- just pandas, which every student already has.
TRACKS_CLEAN_PATH = os.path.join(PROCESSED_DIR, "tracks_clean.csv")
INTERACTIONS_CLEAN_PATH = os.path.join(PROCESSED_DIR, "interactions_clean.csv")
CONTENT_MODEL_PATH = os.path.join(PROCESSED_DIR, "content_model.pkl")
CF_MODEL_PATH = os.path.join(PROCESSED_DIR, "cf_model.pkl")
ID_MAPS_PATH = os.path.join(PROCESSED_DIR, "id_maps.pkl")

# ---------------------------------------------------------------------------
# Sampling / filtering thresholds
# ---------------------------------------------------------------------------
# The raw User_Listening_History.csv from the Kaggle "Music Recommender
# System" dataset (Music Info + Listening History, Last.fm / Spotify based)
# contains several million rows. Training a dense collaborative-filtering
# model on the full file is unnecessary for a course prototype and is slow
# on a laptop, so we down-sample it to a dense, well-behaved subset:
#
#   1. Keep only interactions whose track_id also exists in Music_Info.csv
#      (so every interaction has content features available).
#   2. Keep only users with at least MIN_INTERACTIONS_PER_USER plays
#      (removes one-off / noisy users -> classic "cold" users).
#   3. Keep only tracks with at least MIN_INTERACTIONS_PER_TRACK plays
#      (removes very long-tail tracks that collaborative filtering can
#      never learn a good signal for anyway).
#   4. If, after the above filtering, there are still more than
#      MAX_USERS unique users, randomly sample MAX_USERS of them
#      (with a fixed random seed for reproducibility).
#
# These numbers can be safely increased if you have more RAM / time.
MIN_INTERACTIONS_PER_USER = 20
MIN_INTERACTIONS_PER_TRACK = 10
MAX_USERS = 8000
RANDOM_SEED = 42

# Rows read per chunk when streaming a very large raw CSV (used by both
# data_preprocessing.py's built-in streaming reader and by the standalone
# downsample_large_history.py script). Lower this (e.g. to 50_000-100_000)
# on a machine with little RAM; raise it (e.g. to 2_000_000) on a machine
# with plenty.
CHUNK_SIZE = 200_000

# The raw file itself is never fully loaded into memory at once -- it is
# streamed in chunks of CHUNK_SIZE rows, and data_preprocessing.py's quick
# built-in reader stops early as soon as either MAX_RAW_ROWS_TO_SCAN rows
# have been scanned, or MAX_USERS_SCAN_TARGET unique users have already
# been seen (whichever comes first). This keeps both runtime and RAM
# bounded on a slower machine regardless of how many millions of rows the
# original file actually has. Increase these if you have a faster machine
# and want a larger sample. (The standalone downsample_large_history.py
# script, for very large files, scans the FULL file instead -- see below.)
MAX_RAW_ROWS_TO_SCAN = 3_000_000
MAX_USERS_SCAN_TARGET = MAX_USERS * 3  # scan a bit more than needed, then randomly cap to MAX_USERS

# Where downsample_large_history.py writes its small, ready-to-use output.
# app.py / data_preprocessing.py read from LISTENING_HISTORY_PATH, so after
# running the downsampler you should either point LISTENING_HISTORY_PATH at
# this file or rename/replace the original with it (the script tells you
# which, at the end of its run).
LISTENING_HISTORY_SAMPLED_PATH = os.path.join(RAW_DIR, "User_Listening_History_sampled.csv")

# ---------------------------------------------------------------------------
# Content-based model
# ---------------------------------------------------------------------------
# Audio features used for content-based similarity (numeric, all already
# roughly in comparable ranges after scaling).
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

# Weight given to the audio-feature block vs the text (tags/genre) block
# when they are concatenated into a single feature vector for each track.
AUDIO_WEIGHT = 0.4
TEXT_WEIGHT = 0.6

TOP_K_NEIGHBORS = 50  # neighbours pre-computed per track for content-based lookup

# ---------------------------------------------------------------------------
# Collaborative-filtering model (matrix factorisation via Truncated SVD
# on the log-scaled implicit play-count matrix)
# ---------------------------------------------------------------------------
N_LATENT_FACTORS = 50

# ---------------------------------------------------------------------------
# Hybrid combination
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 0.5  # weight for collaborative score; (1 - alpha) for content

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
TOP_N_EVAL = 10          # Precision@N / Recall@N / F1@N
TEST_HOLDOUT_PER_USER = 2  # leave-N-out per user for the test split
