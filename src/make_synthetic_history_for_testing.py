"""
OPTIONAL, TESTING-ONLY utility.

Generates a small synthetic data/raw/User_Listening_History.csv so you can
run and test the whole app (`streamlit run app.py`) BEFORE you have added
the real dataset. It fakes users with a genre preference so the
collaborative-filtering model has some structure to learn -- it is NOT a
substitute for the real Kaggle listening-history data and must NOT be used
for the final submission / report (the documentation must describe the
real dataset).

Usage:
    python -m src.make_synthetic_history_for_testing
"""

import numpy as np
import pandas as pd

from src import config


def main(n_users: int = 3000, seed: int = config.RANDOM_SEED) -> None:
    rng = np.random.default_rng(seed)
    tracks = pd.read_csv(config.MUSIC_INFO_PATH)

    track_ids = tracks["track_id"].values
    genres = tracks["genre"].fillna("Unknown").values
    top_genres = pd.Series(genres).value_counts().index[:8].tolist()

    # restrict each genre's candidate pool so tracks get repeated across
    # many synthetic users (otherwise the matrix is too sparse to learn from)
    genre_pools = {g: np.where(genres == g)[0][:150] for g in top_genres}

    rows = []
    for u in range(n_users):
        uid = f"synthuser_{u:04d}"
        pref_genre = rng.choice(top_genres)
        candidates = genre_pools[pref_genre]
        n_tracks = rng.integers(20, 60)
        chosen = rng.choice(candidates, size=min(n_tracks, len(candidates)), replace=False)
        for c in chosen:
            rows.append((track_ids[c], uid, int(rng.integers(1, 50))))

    history = pd.DataFrame(rows, columns=["track_id", "user_id", "playcount"])
    history.to_csv(config.LISTENING_HISTORY_PATH, index=False)
    print(f"Wrote {len(history)} synthetic rows to {config.LISTENING_HISTORY_PATH}")
    print("Remember: replace this with the REAL User_Listening_History.csv before submitting.")


if __name__ == "__main__":
    main()
