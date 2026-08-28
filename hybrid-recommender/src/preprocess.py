"""
preprocess.py
=============
One-off offline pipeline that turns the two raw datasets

    data/Music_Info.csv
    data/User Listening History.csv

into a set of lightweight artifacts that the Streamlit app (app.py) loads at
start-up. Doing the heavy lifting here means the app itself starts in a
couple of seconds instead of re-computing everything on every run.

Artifacts written to ./artifacts/ (plain CSV/NPY/NPZ/joblib only -- no
pyarrow/fastparquet dependency needed to run the app):
    music_info.csv              cleaned track catalogue (50,683 tracks)
    content_matrix.npz          sparse content feature matrix (tags + audio)
    content_nn.joblib           fitted sklearn NearestNeighbors (cosine) model
    item_factors.npy            latent item vectors from TruncatedSVD on the
                                 implicit user-item interaction matrix
                                 (one row per track in music_info, zero vector
                                 for tracks nobody has ever played -> cold item)
    interactions.npz            compact (user_idx, track_idx, playcount) arrays
    user_id_categories.npy      int index -> original user_id string
    track_id_categories.npy     int index -> original track_id string  (== the
                                 row order of music_info.csv / item_factors)
    demo_users.csv             ~60 ready-made demo login accounts spanning
                                light / medium / heavy listeners
    eval_users.npy             sample of user indices reserved for the
                                developer evaluation page (held-out songs)

Run once with:  python src/preprocess.py
"""
import hashlib
import os

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

RNG = np.random.default_rng(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
ART_DIR = os.path.join(BASE, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

AUDIO_COLS = [
    "year", "duration_ms", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature",
]


def log(msg):
    print(f"[preprocess] {msg}", flush=True)


def build_music_catalog():
    log("loading Music_Info.csv ...")
    mi = pd.read_csv(os.path.join(DATA_DIR, "Music_Info.csv"))
    mi = mi.drop_duplicates(subset="track_id").reset_index(drop=True)
    mi["tags"] = mi["tags"].fillna("")
    mi["genre"] = mi["genre"].fillna("")
    # fold genre into the tag bag-of-words so it contributes to content similarity
    mi["tag_text"] = (mi["tags"].str.replace(", ", " ", regex=False) + " " +
                       mi["genre"].str.replace(" ", "_", regex=False)).str.strip()
    # CSV (not parquet) so the app has zero dependency on pyarrow/fastparquet
    mi.to_csv(os.path.join(ART_DIR, "music_info.csv"), index=False)
    log(f"catalogue saved: {len(mi)} tracks")
    return mi


def build_content_features(mi: pd.DataFrame):
    log("building content feature matrix (tags TF-IDF + scaled audio features) ...")
    tfidf = TfidfVectorizer(token_pattern=r"[^\s,]+", min_df=2)
    tag_mat = tfidf.fit_transform(mi["tag_text"])
    log(f"  tag vocabulary size: {len(tfidf.vocabulary_)}")

    scaler = MinMaxScaler()
    audio_mat = scaler.fit_transform(mi[AUDIO_COLS].astype(float))
    audio_sparse = sp.csr_matrix(audio_mat)

    # weight the two blocks roughly equally before concatenation
    tag_mat = tag_mat.multiply(1.0 / max(tag_mat.max(), 1e-9))
    audio_sparse = audio_sparse.multiply(1.0 / max(audio_sparse.max(), 1e-9))

    content = sp.hstack([tag_mat, audio_sparse]).tocsr()
    sp.save_npz(os.path.join(ART_DIR, "content_matrix.npz"), content)

    log("  fitting NearestNeighbors (cosine) index ...")
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(content)
    joblib.dump(nn, os.path.join(ART_DIR, "content_nn.joblib"))
    joblib.dump(tfidf, os.path.join(ART_DIR, "tfidf.joblib"))
    joblib.dump(scaler, os.path.join(ART_DIR, "scaler.joblib"))
    log("  content features done")
    return content


def build_interactions_and_cf(mi: pd.DataFrame):
    log("loading User Listening History.csv (~9.7M rows) ...")
    lh = pd.read_csv(
        os.path.join(DATA_DIR, "User Listening History.csv"),
        dtype={"track_id": "category", "user_id": "category", "playcount": "int32"},
    )

    # re-index tracks so they line up 1:1 with music_info row order
    track_index = {tid: i for i, tid in enumerate(mi["track_id"].values)}
    lh = lh[lh["track_id"].isin(track_index)].copy()
    lh["track_idx"] = lh["track_id"].map(track_index).astype("int32")

    user_cats = lh["user_id"].cat.categories.values
    lh["user_idx"] = lh["user_id"].cat.codes.astype("int32")

    interactions = lh[["user_idx", "track_idx", "playcount"]].reset_index(drop=True)
    np.savez_compressed(
        os.path.join(ART_DIR, "interactions.npz"),
        user_idx=interactions["user_idx"].values.astype("int32"),
        track_idx=interactions["track_idx"].values.astype("int32"),
        playcount=interactions["playcount"].values.astype("int32"),
    )
    np.save(os.path.join(ART_DIR, "user_id_categories.npy"), user_cats)
    np.save(os.path.join(ART_DIR, "track_id_categories.npy"), mi["track_id"].values)
    log(f"  interactions saved: {len(interactions)} rows, "
        f"{len(user_cats)} users, {mi.shape[0]} tracks (catalog)")

    # ---- implicit matrix factorisation (item side) ----------------------
    log("  building sparse item-user matrix + TruncatedSVD (k=50) ...")
    n_items = mi.shape[0]
    n_users = len(user_cats)
    weight = np.log1p(interactions["playcount"].values).astype("float32")
    item_user = sp.csr_matrix(
        (weight, (interactions["track_idx"].values, interactions["user_idx"].values)),
        shape=(n_items, n_users),
    )
    svd = TruncatedSVD(n_components=50, random_state=42)
    item_factors = svd.fit_transform(item_user)  # (n_items, 50), zero row = never played
    np.save(os.path.join(ART_DIR, "item_factors.npy"), item_factors.astype("float32"))
    log(f"  explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.3f}")
    return interactions, user_cats


def build_demo_users(interactions: pd.DataFrame, user_cats):
    log("building demo login accounts ...")
    counts = interactions.groupby("user_idx").size()
    light = counts[(counts >= 5) & (counts < 15)].sample(20, random_state=1).index
    medium = counts[(counts >= 15) & (counts < 40)].sample(20, random_state=2).index
    heavy = counts[counts >= 40].sample(20, random_state=3).index

    rows = []
    for bucket_name, idxs in [("light", light), ("medium", medium), ("heavy", heavy)]:
        for i, uidx in enumerate(idxs):
            username = f"{bucket_name}_{i+1:02d}"
            password = "demo123"  # demo-only, plaintext shown to grader on purpose
            pw_hash = hashlib.sha256(password.encode()).hexdigest()
            rows.append({
                "username": username,
                "password_hash": pw_hash,
                "password_plain_DEMO_ONLY": password,
                "user_idx": int(uidx),
                "user_id": user_cats[uidx],
                "n_interactions": int(counts.loc[uidx]),
                "activity_level": bucket_name,
            })
    demo = pd.DataFrame(rows)
    demo.to_csv(os.path.join(ART_DIR, "demo_users.csv"), index=False)
    log(f"  {len(demo)} demo accounts created (see artifacts/demo_users.csv)")

    # separate, larger pool of users held out for the developer evaluation page
    eligible = counts[counts >= 10].index.values
    eval_users = RNG.choice(eligible, size=min(3000, len(eligible)), replace=False)
    np.save(os.path.join(ART_DIR, "eval_users.npy"), eval_users)
    log(f"  {len(eval_users)} users reserved for offline evaluation")


def build_dev_credentials():
    creds = {"admin": hashlib.sha256("aidemo2026".encode()).hexdigest()}
    joblib.dump(creds, os.path.join(ART_DIR, "dev_credentials.joblib"))
    log("developer account -> username: admin / password: aidemo2026 (DEMO ONLY)")


def main():
    mi = build_music_catalog()
    build_content_features(mi)
    interactions, user_cats = build_interactions_and_cf(mi)
    build_demo_users(interactions, user_cats)
    build_dev_credentials()
    log("ALL DONE. Artifacts are in ./artifacts")


if __name__ == "__main__":
    main()
