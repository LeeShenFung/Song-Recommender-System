"""
recommender.py
===============
Core hybrid music recommender.

Hybrid design
-------------
score(user, seed_song, candidate) =
        alpha * content_similarity(seed_song, candidate)
    + (1 - alpha) * collaborative_similarity(user, candidate)

1) CONTENT-BASED component
   Every track is represented as a TF-IDF bag of its Spotify "tags"/genre
   plus its scaled audio features (danceability, energy, tempo, ...).
   Cosine similarity between the seed song the user typed and every other
   track in the catalogue gives the content score. This part needs no
   listening history at all, so it also covers brand-new / cold-start items.

2) COLLABORATIVE-FILTERING component
   An implicit-feedback item-user matrix (weighted by log(playcount)) is
   factorised once offline with TruncatedSVD -> a 50-dim latent vector per
   track. A user's own taste vector is obtained on the fly by "folding in"
   their play history: the (playcount-weighted) average of the latent
   vectors of the songs they already listened to. The collaborative score
   of a candidate track is then the cosine similarity between the user's
   taste vector and the candidate's latent vector. Tracks nobody has ever
   played get an all-zero latent vector, so the model automatically falls
   back onto the content score for those (graceful cold-item handling).

Both raw score vectors are min-max normalised to [0, 1] before being
combined, so `alpha` behaves predictably regardless of the very different
native scales of a TF-IDF cosine and an SVD cosine.
"""
import os

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sk_normalize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR = os.path.join(BASE, "artifacts")


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


class HybridRecommender:
    def __init__(self, art_dir: str = ART_DIR):
        self.art_dir = art_dir
        self._load_artifacts()

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def _load_artifacts(self):
        self.music_info = pd.read_csv(os.path.join(self.art_dir, "music_info.csv"))
        self.n_items = len(self.music_info)
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.music_info["track_id"].values)}

        content = sp.load_npz(os.path.join(self.art_dir, "content_matrix.npz"))
        self.content_matrix = sk_normalize(content, axis=1)  # rows -> unit L2 norm for cosine via dot

        item_factors = np.load(os.path.join(self.art_dir, "item_factors.npy"))
        norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.item_factors = item_factors / norms  # unit-normalised, zero rows stay zero

        npz = np.load(os.path.join(self.art_dir, "interactions.npz"))
        self.inter_user = npz["user_idx"]
        self.inter_track = npz["track_idx"]
        self.inter_count = npz["playcount"]

        # popularity (global play count) used for search ranking & cold-start fallback
        pop = np.bincount(self.inter_track, minlength=self.n_items)
        self.popularity = pop

        self.user_id_categories = np.load(os.path.join(self.art_dir, "user_id_categories.npy"), allow_pickle=True)
        self.n_users = len(self.user_id_categories)

        # build a per-user index once (fast fold-in lookups) -------------
        order = np.argsort(self.inter_user, kind="stable")
        self._sorted_user = self.inter_user[order]
        self._sorted_track = self.inter_track[order]
        self._sorted_count = self.inter_count[order]
        self._user_boundaries = np.searchsorted(self._sorted_user, np.arange(self.n_users + 1))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def user_history(self, user_idx: int):
        """Return (track_idx array, playcount array) for one user.
        user_idx < 0 (or >= n_users) is treated as a brand-new user with no
        listening history at all -- i.e. the cold-start case -- and returns
        two empty arrays rather than raising or wrapping around via negative
        indexing."""
        if user_idx is None or user_idx < 0 or user_idx >= self.n_users:
            empty = np.array([], dtype="int32")
            return empty, empty
        s, e = self._user_boundaries[user_idx], self._user_boundaries[user_idx + 1]
        return self._sorted_track[s:e], self._sorted_count[s:e]

    def search_tracks(self, query: str, limit: int = 12):
        """Case-insensitive substring search over song name / artist,
        ranked by global popularity so well-known songs surface first."""
        q = query.strip().lower()
        if not q:
            return []
        mi = self.music_info
        mask = mi["name"].str.lower().str.contains(q, na=False) | \
            mi["artist"].str.lower().str.contains(q, na=False)
        hits = mi[mask].copy()
        if hits.empty:
            return []
        hits["pop"] = self.popularity[hits.index.values]
        hits = hits.sort_values("pop", ascending=False).head(limit)
        return [
            {"track_idx": i, "name": r["name"], "artist": r["artist"],
             "year": int(r["year"]), "tags": r["tags"]}
            for i, r in hits.iterrows()
        ]

    def user_taste_vector(self, user_idx: int) -> np.ndarray:
        tracks, counts = self.user_history(user_idx)
        if len(tracks) == 0:
            return np.zeros(self.item_factors.shape[1], dtype="float32")
        w = np.log1p(counts).astype("float32")
        vec = (self.item_factors[tracks] * w[:, None]).sum(axis=0) / w.sum()
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    # ------------------------------------------------------------------ #
    # core scoring
    # ------------------------------------------------------------------ #
    def content_scores(self, seed_track_idx: int) -> np.ndarray:
        seed_vec = self.content_matrix[seed_track_idx]
        return np.asarray(self.content_matrix.dot(seed_vec.T).todense()).ravel()

    def cf_scores(self, user_idx: int) -> np.ndarray:
        user_vec = self.user_taste_vector(user_idx)
        return self.item_factors.dot(user_vec)

    def recommend(self, user_idx: int, seed_track_idx: int, top_n: int = 10,
                   alpha: float = 0.5, exclude_listened: bool = True,
                   exclude_seed: bool = True):
        """Return the top_n hybrid recommendations for one (user, seed song)."""
        c_raw = self.content_scores(seed_track_idx)
        cf_raw = self.cf_scores(user_idx)

        c_norm = _minmax(c_raw)
        cf_norm = _minmax(cf_raw)
        hybrid = alpha * c_norm + (1 - alpha) * cf_norm

        exclude = set()
        if exclude_seed:
            exclude.add(seed_track_idx)
        if exclude_listened:
            listened, _ = self.user_history(user_idx)
            exclude.update(listened.tolist())

        order = np.argsort(-hybrid)
        results = []
        for idx in order:
            if idx in exclude:
                continue
            row = self.music_info.iloc[idx]
            results.append({
                "track_idx": int(idx),
                "name": row["name"],
                "artist": row["artist"],
                "year": int(row["year"]),
                "tags": row["tags"],
                "preview_url": row["spotify_preview_url"],
                "hybrid_score": float(hybrid[idx]),
                "content_score": float(c_norm[idx]),
                "cf_score": float(cf_norm[idx]),
            })
            if len(results) >= top_n:
                break
        return results

    def popular_tracks(self, top_n: int = 10, exclude=None):
        """Fallback recommender: most globally played tracks (used for
        brand-new users with zero history, i.e. the pure cold-start case)."""
        exclude = exclude or set()
        order = np.argsort(-self.popularity)
        out = []
        for idx in order:
            if idx in exclude:
                continue
            row = self.music_info.iloc[idx]
            out.append({
                "track_idx": int(idx), "name": row["name"], "artist": row["artist"],
                "year": int(row["year"]), "tags": row["tags"],
                "preview_url": row["spotify_preview_url"], "plays": int(self.popularity[idx]),
            })
            if len(out) >= top_n:
                break
        return out
