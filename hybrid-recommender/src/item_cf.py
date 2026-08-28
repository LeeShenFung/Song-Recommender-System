"""
item_cf.py
==========
Item-Based Collaborative Filtering for the developer dashboard.

This implementation reuses the preprocessed artifacts created by Feng's
pipeline instead of loading the original ~9.7M-row listening-history CSV.

Algorithm:
    interactions.npz
        -> select top active users
        -> sparse Item x User playcount matrix
        -> cosine similarity between a seed song and every other song
        -> Top-N similar songs

Important:
- Similarity is based on user playcount behaviour, NOT song metadata.
- music_info.csv is only used to display song information.
- The full 50k x 50k item-item similarity matrix is NOT precomputed.
  Similarity is calculated on demand for the selected seed song.
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:
    def __init__(self, art_dir: str, max_users: int = 1000):
        self.art_dir = art_dir
        self.max_users = max_users
        self._load()

    def _load(self):
        # Track catalogue. Its row order is the same track_idx order used by
        # interactions.npz, guaranteed by preprocess.py.
        self.music_info = pd.read_csv(
            os.path.join(self.art_dir, "music_info.csv")
        ).reset_index(drop=True)

        npz = np.load(os.path.join(self.art_dir, "interactions.npz"))
        inter_user = npz["user_idx"].astype("int32")
        inter_track = npz["track_idx"].astype("int32")
        inter_count = npz["playcount"].astype("float32")

        self.n_items = len(self.music_info)
        self.n_users_total = int(inter_user.max()) + 1

        # Match the notebook idea: use the most active users.
        user_counts = np.bincount(inter_user, minlength=self.n_users_total)
        n_keep = min(self.max_users, np.count_nonzero(user_counts))
        top_users = np.argsort(user_counts)[-n_keep:]
        self.top_users = np.sort(top_users)

        # Map original/global user_idx -> compact local column index.
        global_to_local = np.full(self.n_users_total, -1, dtype="int32")
        global_to_local[self.top_users] = np.arange(len(self.top_users), dtype="int32")

        keep_mask = global_to_local[inter_user] >= 0
        local_users = global_to_local[inter_user[keep_mask]]
        tracks = inter_track[keep_mask]
        counts = inter_count[keep_mask]

        # Item x User sparse matrix.
        # Raw playcount is used here to stay consistent with the notebook.
        self.item_user_matrix = sp.csr_matrix(
            (counts, (tracks, local_users)),
            shape=(self.n_items, len(self.top_users)),
            dtype="float32",
        )

        # Popularity within the selected-user subset, used only to rank
        # search results. It does not affect cosine similarity.
        self.subset_popularity = np.asarray(
            self.item_user_matrix.sum(axis=1)
        ).ravel()

    def search_tracks(self, query: str, limit: int = 12):
        """Case-insensitive title/artist search."""
        q = query.strip().lower()
        if not q:
            return []

        mi = self.music_info
        mask = (
            mi["name"].fillna("").str.lower().str.contains(q, regex=False)
            | mi["artist"].fillna("").str.lower().str.contains(q, regex=False)
        )

        idxs = np.flatnonzero(mask.to_numpy())
        if len(idxs) == 0:
            return []

        # Show the more interacted-with matches first.
        idxs = idxs[np.argsort(-self.subset_popularity[idxs])][:limit]

        results = []
        for idx in idxs:
            row = mi.iloc[idx]
            results.append(
                {
                    "track_idx": int(idx),
                    "track_id": row.get("track_id", ""),
                    "name": row.get("name", ""),
                    "artist": row.get("artist", ""),
                    "genre": row.get("genre", ""),
                    "year": row.get("year", ""),
                    "has_cf_signal": bool(self.item_user_matrix[idx].getnnz() > 0),
                }
            )
        return results

    def recommend_similar_tracks(self, seed_track_idx: int, top_n: int = 10):
        """
        Recommend tracks whose selected-user playcount vectors have the
        highest cosine similarity to the seed track.
        """
        seed_track_idx = int(seed_track_idx)

        if seed_track_idx < 0 or seed_track_idx >= self.n_items:
            raise IndexError("seed_track_idx is outside the catalogue.")

        seed_vec = self.item_user_matrix[seed_track_idx]

        # If none of the selected users listened to this track, item-based CF
        # has no behavioural signal for it.
        if seed_vec.getnnz() == 0:
            return []

        similarities = cosine_similarity(
            seed_vec,
            self.item_user_matrix,
            dense_output=True,
        ).ravel()

        # Exclude the seed song itself.
        similarities[seed_track_idx] = -1.0

        # Only return tracks that actually have collaborative signal.
        valid = self.item_user_matrix.getnnz(axis=1) > 0
        valid[seed_track_idx] = False
        candidate_idxs = np.flatnonzero(valid)

        if len(candidate_idxs) == 0:
            return []

        k = min(int(top_n), len(candidate_idxs))
        candidate_scores = similarities[candidate_idxs]

        # Efficient Top-K selection, then sort those K exactly.
        if k < len(candidate_idxs):
            top_local = np.argpartition(-candidate_scores, k - 1)[:k]
            top_idxs = candidate_idxs[top_local]
        else:
            top_idxs = candidate_idxs

        top_idxs = top_idxs[np.argsort(-similarities[top_idxs])][:k]

        results = []
        for idx in top_idxs:
            row = self.music_info.iloc[idx]
            results.append(
                {
                    "track_idx": int(idx),
                    "track_id": row.get("track_id", ""),
                    "name": row.get("name", ""),
                    "artist": row.get("artist", ""),
                    "genre": row.get("genre", ""),
                    "year": row.get("year", ""),
                    "similarity_score": float(similarities[idx]),
                }
            )
        return results
