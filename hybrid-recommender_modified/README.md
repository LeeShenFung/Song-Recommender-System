# Hybrid Music Recommender System (Streamlit)

A prototype built for the *Artificial Intelligence* group assignment
(Title 3: Recommender System). It combines **content-based filtering**
and **collaborative filtering** into one hybrid model, wrapped in a
Streamlit web app with a user login flow and a separate developer login
for model evaluation.

## Dataset

| File | Rows | Description |
|---|---|---|
| `data/Music_Info.csv` | 50,683 tracks | Spotify audio features + tags/genre |
| `data/User Listening History.csv` | 9,711,301 rows | `(user_id, track_id, playcount)` implicit feedback |

## For your teammates (GitHub workflow)

The repo ships with `artifacts/` already committed (all files are well
under GitHub's 100MB per-file limit, biggest is ~40MB), so teammates get a
working app immediately:

```bash
git clone <your-repo-url>
cd project
pip install -r requirements.txt
streamlit run app.py
```

No need to download or place the raw CSVs anywhere — `data/` is
intentionally left empty (`.gitignore` excludes the raw datasets: the full
`User Listening History.csv` is 602MB, over GitHub's limit, and isn't
needed to run the app anyway).

Only put the two raw CSVs back into `data/` if someone wants to modify
`src/preprocess.py` and regenerate the artifacts (e.g. to try a different
SVD dimension or re-weight the content features) — ask whoever has the
original files (from the assignment upload) to share them again, then run
`python src/preprocess.py`.

`.gitattributes` is included so Git treats `.npy` / `.npz` / `.joblib`
files as binary — don't remove it, or a mixed Windows/Mac team could end up
silently corrupting those files via line-ending conversion.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The `artifacts/` folder already contains all pre-computed models (feature
matrices, latent factors, demo accounts), so the app starts in a few
seconds. You do **not** need the raw CSVs to run the app — they are only
needed if you want to regenerate the artifacts yourself:

```bash
# only needed if you want to rebuild artifacts/ from scratch
# put Music_Info.csv and "User Listening History.csv" inside data/
python src/preprocess.py
```

## Demo credentials

**User login** — pick any account from the in-app "Demo accounts" panel,
password is `demo123` for all of them (20 light / 20 medium / 20 heavy
listeners, so you can see how the hybrid model behaves differently for
users with little vs. a lot of history).

**Developer login** — `admin` / `aidemo2026`.

> These are demo-only, plaintext-documented credentials so the tutor can
> log in easily for grading. In a production system passwords would never
> be shown like this.

## App flow

1. **User login** → **type a song name** → **choose how many
   recommendations** → get a ranked, hybrid Top-N list (with score
   breakdown and, where available, a 30-second Spotify audio preview).
2. **Developer login** → run an offline evaluation (Precision@K,
   Recall@K, F1@K) comparing content-only, collaborative-only and hybrid
   configurations, to assess the recommender's accuracy.

## Model design

### 1. Content-based component
Each track's Spotify **tags + genre** are turned into a TF-IDF
bag-of-words vector; **audio features** (danceability, energy, tempo,
valence, loudness, acousticness, etc.) are min-max scaled. The two blocks
are concatenated into one sparse feature vector per track. The content
score of a candidate song is the **cosine similarity** between its vector
and the seed song's vector the user typed in. This part works even for
songs nobody has ever played (no cold-start problem for new items).

### 2. Collaborative-filtering component
An implicit-feedback **item × user** matrix (weighted by `log(1+playcount)`)
is factorised once, offline, with **TruncatedSVD** (k=50) to obtain a
50-dimensional latent vector per track. A logged-in user's own taste
vector is computed on the fly by **folding in** their listening history:
the playcount-weighted average of the latent vectors of the songs they've
already played. The collaborative score of a candidate is the cosine
similarity between the user's taste vector and the candidate's latent
vector. Tracks nobody has played get an all-zero latent vector, so the
model naturally leans on the content score for them (graceful cold-item
handling).

### 3. Hybrid combination
```
hybrid_score = alpha * content_score + (1 - alpha) * collaborative_score
```
Both scores are min-max normalised before blending so `alpha` behaves
predictably. `alpha` is adjustable in the sidebar (0 = pure collaborative,
1 = pure content-based).

## Evaluation methodology (developer page)

Leave-one-out evaluation over a sample of users with ≥10 interactions:
1. Hide the user's **most-played** song as ground truth.
2. Feed the model the user's **second most-played** song as the seed
   input, and the remaining history as the collaborative profile.
3. Count a **hit** if the hidden song appears in the Top-K list.
4. Since exactly one relevant item exists per user:
   `Precision@K = hits / (n_users * K)`, `Recall@K = hits / n_users`,
   `F1@K = 2PR / (P + R)`.

Running this on this dataset shows the collaborative signal is
substantially stronger than content alone (tags/audio features are
coarse), and the hybrid lets you trade the two off — this is a useful,
honest empirical result to discuss in the report's Results & Discussion
section, and a good place to plug in your own additional experiments
(e.g. try different `k` in the SVD, different `K` cut-offs, or add more
content signal such as lyrics).

## Project structure

```
app.py                  Streamlit application (login, user page, dev page)
src/preprocess.py       Offline pipeline: raw CSVs -> artifacts/
src/recommender.py      HybridRecommender class (content + CF + hybrid)
src/evaluate.py         Leave-one-out Precision/Recall/F1 evaluation
artifacts/               Pre-computed models & demo accounts (ships with the app)
data/                    Put the two raw CSV files here only if re-running preprocess.py
requirements.txt
```

## Suggestions for pushing this to an "Excellent" grade

- Swap TruncatedSVD for `implicit`'s ALS or a neural model (NCF /
  two-tower) and compare.
- Add a second content signal (e.g. lyrics embeddings) and show it
  improves content-only Precision/Recall.
- Report results at several `K` and plot Precision-Recall curves.
- Add a short user study: have a few classmates rate the recommendations
  and report satisfaction alongside the offline metrics.
