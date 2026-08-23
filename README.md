# Hybrid Music Recommender System

A hybrid music recommender that blends **content-based filtering** (audio
features + tags/genre) with **collaborative filtering** (implicit-feedback
matrix factorisation on user play counts), built for the TARUMT Artificial
Intelligence group assignment (Recommender System title).

## 1. How the hybrid approach works

| Component | Signal used | Solves |
|---|---|---|
| **Content-based** | Spotify audio features (danceability, energy, valence, tempo, ...) + TF-IDF over `tags`/`genre` | Recommending based on *what a song sounds like*. Works even for a track/user with zero play history (cold start). |
| **Collaborative filtering** | User–track implicit play-count matrix, factorised with Truncated SVD into latent taste vectors | Recommending based on *what similar listeners played*, catching patterns audio features alone can't see. |
| **Hybrid** | `hybrid_score = alpha * cf_score + (1-alpha) * content_score` (both min-max normalised first) | Combines both signals; `alpha` is a tunable slider in the app. New users automatically fall back to `alpha = 0` (pure content-based). |

See `src/content_based.py`, `src/collaborative_filtering.py` and
`src/hybrid_recommender.py` for the full implementation and inline
explanations.

## 2. Project structure

```
music_recommender/
├── app.py                     # Streamlit UI (entry point)
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── Music_Info.csv                 # (provided)
│   │   └── User_Listening_History.csv     # <-- YOU add this file here
│   └── processed/              # auto-generated cache (safe to delete)
└── src/
    ├── config.py                       # all paths + tunable constants
    ├── data_preprocessing.py           # cleaning + down-sampling
    ├── content_based.py                # TF-IDF + audio features + kNN
    ├── collaborative_filtering.py      # Truncated-SVD matrix factorisation
    ├── hybrid_recommender.py           # weighted combination
    ├── evaluate.py                     # leave-N-out Precision/Recall/F1
    └── make_synthetic_history_for_testing.py   # OPTIONAL, testing only
```

## 3. Setup

```bash
pip install -r requirements.txt
```

1. `Music_Info.csv` is already in `data/raw/`.
2. **Add your `User_Listening_History.csv` to `data/raw/`.** Expected
   columns (matching the Kaggle "Music Recommender System" / Last.fm +
   Spotify dataset): `track_id, user_id, playcount`.
3. Run the app:

```bash
streamlit run app.py
```

The first launch will preprocess the data (cleans `Music_Info.csv`,
filters/down-samples the listening history — see "Sampling strategy"
below) and cache the result in `data/processed/`. Delete that folder if
you replace the raw data and want to rebuild from scratch.

If you want to click around the UI before your real listening-history
file is ready, you can generate a small synthetic one for testing only:

```bash
python -m src.make_synthetic_history_for_testing
```

**Do not use the synthetic file for your final submission** — replace it
with the real `User_Listening_History.csv` before writing up results.

## 4. Sampling strategy (why we don't use the raw file as-is)

The raw listening-history log can contain several million rows across a
very large number of users, most of whom only played a handful of songs.
Loading a file like that fully into memory with `pandas.read_csv()` before
doing anything else is exactly what makes preprocessing slow / crash on a
weaker machine. `src/data_preprocessing.py` avoids that entirely:

**Step A -- bounded streaming (never loads the whole file at once).**
The file is read in chunks of `config.CHUNK_SIZE` rows (default 500,000).
Each chunk is immediately filtered down to only the columns/rows we need,
and reading **stops early** as soon as either:
  - `config.MAX_RAW_ROWS_TO_SCAN` raw rows have been scanned (default 3,000,000), or
  - `config.MAX_USERS_SCAN_TARGET` unique users have already been seen (default `MAX_USERS * 3`),

whichever happens first. In practice this means only a few chunks (a few
seconds) are ever read, no matter how large the original file is --
tested with simulated files up to 5,000,000 rows / 165 MB, preprocessing
finishes in 1-3 seconds.

**Step B -- documented, reproducible filtering** on the scanned rows:
1. Keep only interactions whose `track_id` also exists in `Music_Info.csv`
   (applied while streaming, so every interaction has content features).
2. Drop users with fewer than `MIN_INTERACTIONS_PER_USER` plays and tracks
   with fewer than `MIN_INTERACTIONS_PER_TRACK` plays (removes noise the
   collaborative model can't learn from anyway).
3. Cap the number of unique users at `MAX_USERS` (default 8,000), sampled
   with a fixed random seed for reproducibility.

This keeps the user-item matrix dense enough for Truncated SVD to learn
meaningful latent factors quickly on a laptop, while remaining a genuine,
reproducible subset of the real dataset (not synthetic data). If your
machine is fast and you want a bigger sample, just raise the numbers in
`src/config.py` (`CHUNK_SIZE`, `MAX_RAW_ROWS_TO_SCAN`, `MAX_USERS_SCAN_TARGET`,
`MAX_USERS`) and delete `data/processed/` to rebuild.

**Option B -- if your file is EXTREMELY large and you want an unbiased
full-file sample.** Because Step A above stops reading early, the sample
can be mildly biased toward whichever rows come first in the raw file
(e.g. if it's sorted by user_id). For a proper *uniform random* sample of
the entire file, run the standalone two-pass downsampler once, before
starting the app:

```bash
python -m src.downsample_large_history
```

It streams the full file twice in bounded chunks (RAM usage stays flat no
matter how big the file is): pass 1 counts plays per user/track across
the *whole* file, pass 2 writes out a properly random sample of qualifying
users to `data/raw/User_Listening_History_sampled.csv`. Rename it to
replace the original `User_Listening_History.csv` (or point
`LISTENING_HISTORY_PATH` in `src/config.py` at it) and then run the app
as normal -- it will be fast since the file is already small.

## 5. Evaluation methodology

`src/evaluate.py` implements a standard implicit-feedback protocol:

- **Leave-N-out split**: for each user with enough history, `N` of their
  plays are held out as ground truth; the rest becomes training data.
- Both models are retrained on the training split only.
- For each test user, we generate a top-N recommendation list from
  training data alone and measure how many held-out (test) tracks appear
  in it: **Precision@N**, **Recall@N**, **F1@N**.
- `compare_methods()` runs this for `alpha = 0` (content-only), `alpha = 1`
  (collaborative-only) and `alpha = 0.5` (hybrid) so the three approaches
  can be directly compared — this comparison is also shown live in the
  "📊 Evaluation" tab of the Streamlit app, and the resulting table/bar
  chart can be dropped straight into the documentation's Results &
  Discussion section.

## 6. Suggestions for pushing this toward an "Excellent" grade

- Swap Truncated SVD for a proper implicit-ALS model (e.g. the `implicit`
  library) and compare against the SVD baseline.
- Add a content-based re-ranking step that also considers `year`/era to
  avoid recommending only very old or very new tracks.
- Try a learned hybrid (e.g. logistic regression over `[content_score,
  cf_score]` trained to predict held-out interactions) instead of a fixed
  weighted sum.
- Report evaluation results at several values of `alpha` (a small sweep)
  to show the sensitivity analysis in the documentation.
