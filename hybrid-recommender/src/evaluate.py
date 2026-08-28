"""
evaluate.py
============
Offline evaluation of the recommender using the standard "leave-one-out"
protocol for implicit-feedback recommenders (the same idea used in the
Neural Collaborative Filtering paper, He et al. 2017):

For every evaluation user (users with >= 10 interactions, sampled during
preprocessing into artifacts/eval_users.npy):
    1. Take their interaction list, sorted by playcount (proxy for how much
       they like a song, since we have no timestamps in this dataset).
    2. Hold out their single most-played song as the "ground truth" target.
    3. Feed the model their *second* most-played song as the seed input
       (this mimics the real app flow: user types a song, gets recs) and
       whatever remains of their history as the collaborative profile.
    4. Generate the Top-K hybrid recommendation list.
    5. Hit = 1 if the held-out ground-truth song appears in that Top-K list.

Because exactly one relevant item exists per user:
    Precision@K = hit / K
    Recall@K    = hit / 1 = hit
    F1@K        = 2 * P * R / (P + R)

These are then averaged over all sampled users. The script compares three
configurations: content-only (alpha=1), collaborative-only (alpha=0), and
the hybrid (alpha=0.5) so the report can show the hybrid actually helps.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommender import HybridRecommender, _minmax  # noqa: E402


def build_holdout(rec: HybridRecommender, user_idx: int):
    """Return (seed_track_idx, holdout_track_idx, profile_tracks, profile_counts)
    or None if the user doesn't have enough distinct songs to evaluate."""
    tracks, counts = rec.user_history(user_idx)
    if len(tracks) < 3:
        return None
    order = np.argsort(-counts)
    tracks, counts = tracks[order], counts[order]
    holdout = tracks[0]
    seed = tracks[1]
    profile_tracks = tracks[2:]
    profile_counts = counts[2:]
    if len(profile_tracks) == 0:
        return None
    return seed, holdout, profile_tracks, profile_counts


def recommend_with_profile(rec: HybridRecommender, seed_track_idx: int,
                            profile_tracks: np.ndarray, profile_counts: np.ndarray,
                            top_n: int, alpha: float, exclude_seed_only: set):
    """Same math as HybridRecommender.recommend but using a *synthetic*
    held-out profile instead of the user's full history (so the ground-truth
    song and the seed song are correctly excluded from the CF fold-in)."""
    c_raw = rec.content_scores(seed_track_idx)
    w = np.log1p(profile_counts).astype("float32")
    vec = (rec.item_factors[profile_tracks] * w[:, None]).sum(axis=0) / w.sum()
    n = np.linalg.norm(vec)
    user_vec = vec / n if n > 0 else vec
    cf_raw = rec.item_factors.dot(user_vec)

    c_norm = _minmax(c_raw)
    cf_norm = _minmax(cf_raw)
    hybrid = alpha * c_norm + (1 - alpha) * cf_norm

    order = np.argsort(-hybrid)
    top = []
    for idx in order:
        if idx in exclude_seed_only:
            continue
        top.append(idx)
        if len(top) >= top_n:
            break
    return top


def evaluate(rec: HybridRecommender, alpha: float, k: int = 10,
             n_users: int | None = None, seed: int = 0):
    eval_users = np.load(os.path.join(rec.art_dir, "eval_users.npy"))
    rng = np.random.default_rng(seed)
    if n_users is not None and n_users < len(eval_users):
        eval_users = rng.choice(eval_users, size=n_users, replace=False)

    hits = 0
    evaluated = 0
    for u in eval_users:
        built = build_holdout(rec, int(u))
        if built is None:
            continue
        seed_track, holdout_track, profile_tracks, profile_counts = built
        top = recommend_with_profile(
            rec, seed_track, profile_tracks, profile_counts, top_n=k,
            alpha=alpha, exclude_seed_only={seed_track},
        )
        hits += int(holdout_track in top)
        evaluated += 1

    precision = hits / (evaluated * k) if evaluated else 0.0
    recall = hits / evaluated if evaluated else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "alpha": alpha, "k": k, "n_evaluated_users": evaluated,
        "hits": hits, "precision_at_k": precision, "recall_at_k": recall,
        "f1_at_k": f1,
    }


def build_holdout_coldstart(rec: HybridRecommender, user_idx: int):
    """Like build_holdout, but only needs 2 songs (seed + ground truth) and
    deliberately keeps NO profile at all -- this is what a brand-new user
    with zero listening history would experience: they type one seed song
    and get recommendations with no collaborative signal whatsoever."""
    tracks, counts = rec.user_history(user_idx)
    if len(tracks) < 2:
        return None
    order = np.argsort(-counts)
    tracks = tracks[order]
    holdout = tracks[0]
    seed = tracks[1]
    return seed, holdout


def evaluate_cold_start(rec: HybridRecommender, k: int = 10,
                         n_users: int | None = None, alpha: float = 1.0, seed: int = 0):
    """Simulate the NEW USER / cold-start experience for a given hybrid
    weight `alpha`. A brand-new user has an all-zero collaborative fold-in
    vector, so their collaborative score is a flat constant across every
    track -- this function makes that explicit rather than hiding it, so
    you can see mathematically why:
        * any alpha > 0 (content contributes something) collapses to the
          SAME ranking as pure content-only (alpha=1), because adding the
          same constant to every item's score can never change the order;
        * alpha = 0 (pure collaborative) degenerates to a tie across every
          track -- there's no signal at all, so ranking is essentially
          arbitrary. This is the clearest demonstration of why pure
          collaborative filtering fails for brand-new users.
    We still reuse real users' held-out most-played song as ground truth so
    Precision/Recall/F1 can be computed, but the model itself is given no
    history to fold in."""
    eval_users = np.load(os.path.join(rec.art_dir, "eval_users.npy"))
    rng = np.random.default_rng(seed)
    if n_users is not None and n_users < len(eval_users):
        eval_users = rng.choice(eval_users, size=n_users, replace=False)

    zero_cf_raw = np.zeros(rec.n_items, dtype="float32")  # no history -> no CF signal
    cf_norm = _minmax(zero_cf_raw)  # constant 0.5 for every track

    hits = 0
    evaluated = 0
    for u in eval_users:
        built = build_holdout_coldstart(rec, int(u))
        if built is None:
            continue
        seed_track, holdout_track = built
        c_raw = rec.content_scores(seed_track)
        c_norm = _minmax(c_raw)
        hybrid = alpha * c_norm + (1 - alpha) * cf_norm

        order = np.argsort(-hybrid, kind="stable")
        top = []
        for idx in order:
            if idx == seed_track:
                continue
            top.append(idx)
            if len(top) >= k:
                break
        hits += int(holdout_track in top)
        evaluated += 1

    precision = hits / (evaluated * k) if evaluated else 0.0
    recall = hits / evaluated if evaluated else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "scenario": "cold_start_new_user", "alpha": alpha, "k": k,
        "n_evaluated_users": evaluated, "hits": hits,
        "precision_at_k": precision, "recall_at_k": recall, "f1_at_k": f1,
    }


def run_comparison_cold_start(rec: HybridRecommender, k: int = 10, n_users: int = 1000,
                               alphas=(0.0, 0.5, 1.0)):
    """Run evaluate_cold_start across several alpha values, so you can see
    concretely that any alpha > 0 ties with content-only, while alpha = 0
    (pure collaborative) collapses."""
    rows = []
    for a in alphas:
        t0 = time.time()
        r = evaluate_cold_start(rec, k=k, n_users=n_users, alpha=a)
        r["seconds"] = round(time.time() - t0, 2)
        rows.append(r)
    return rows


def run_comparison(rec: HybridRecommender, k: int = 10, n_users: int = 1000,
                    alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    rows = []
    for a in alphas:
        t0 = time.time()
        r = evaluate(rec, alpha=a, k=k, n_users=n_users)
        r["seconds"] = round(time.time() - t0, 2)
        rows.append(r)
    return rows


if __name__ == "__main__":
    rec = HybridRecommender()
    rows = run_comparison(rec, k=10, n_users=1000)
    for r in rows:
        label = {0.0: "Collaborative-only", 1.0: "Content-only"}.get(r["alpha"], f"Hybrid (alpha={r['alpha']})")
        print(f"{label:22s} P@{r['k']}={r['precision_at_k']:.4f}  "
              f"R@{r['k']}={r['recall_at_k']:.4f}  F1@{r['k']}={r['f1_at_k']:.4f}  "
              f"(n={r['n_evaluated_users']}, {r['seconds']}s)")
