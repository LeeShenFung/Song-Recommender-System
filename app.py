"""
Streamlit front-end for the Hybrid Music Recommender System.

Run with:

    streamlit run app.py

Features:
  - Existing user mode: pick a real user_id from the listening history and
    get personalised hybrid recommendations, with a slider to move between
    pure content-based and pure collaborative filtering.
  - New / cold-start user mode: search for a few songs you like and get
    content-based recommendations (no listening history required).
  - "Find similar songs" mode: type a song name, get its nearest neighbours
    in the content-feature space.
  - An Evaluation tab that trains on a leave-N-out split and reports
    Precision@N / Recall@N / F1@N for content-only, collaborative-only and
    hybrid, so the three approaches can be compared side by side.
"""

import streamlit as st
import pandas as pd

from src import config
from src.data_preprocessing import run_preprocessing
from src.content_based import build_content_model, similar_tracks, save_content_model
from src.collaborative_filtering import build_cf_model, recommend_for_user
from src.hybrid_recommender import recommend_hybrid
from src.evaluate import leave_n_out_split, compare_methods

st.set_page_config(page_title="Hybrid Music Recommender", page_icon="🎵", layout="wide")


# ---------------------------------------------------------------------------
# Cached resource loading (heavy work happens once per session, not per click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading and preprocessing data...")
def load_data():
    tracks_df, interactions_df = run_preprocessing()
    return tracks_df, interactions_df


@st.cache_resource(show_spinner="Training content-based model...")
def get_content_model(tracks_df):
    return build_content_model(tracks_df)


@st.cache_resource(show_spinner="Training collaborative-filtering model...")
def get_cf_model(interactions_df):
    return build_cf_model(interactions_df)


def track_lookup(tracks_df: pd.DataFrame) -> pd.DataFrame:
    return tracks_df.set_index("track_id")


def enrich(df: pd.DataFrame, tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Attach name / artist / genre columns to a track_id-only result table."""
    return df.merge(tracks_df[["track_id", "name", "artist", "genre", "year"]],
                     on="track_id", how="left")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.title("🎵 Hybrid Music Recommender System")
st.caption(
    "Content-based filtering (audio features + tags/genre) combined with "
    "collaborative filtering (latent-factor matrix factorisation on implicit "
    "play-count data), blended into a single tunable hybrid score."
)

try:
    tracks_df, interactions_df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Expected files:\n\n"
        f"- `{config.MUSIC_INFO_PATH}`\n"
        f"- `{config.LISTENING_HISTORY_PATH}` (columns: track_id, user_id, playcount)\n\n"
        "Place them in `data/raw/` and reload the app."
    )
    st.stop()

content_model = get_content_model(tracks_df)
cf_model = get_cf_model(interactions_df)

tab_user, tab_cold_start, tab_similar, tab_eval = st.tabs(
    ["👤 Recommend for a user", "🆕 New user (cold start)", "🔎 Find similar songs", "📊 Evaluation"]
)

# ---------------------------------------------------------------------------
# Tab 1: recommend for an existing user
# ---------------------------------------------------------------------------
with tab_user:
    st.subheader("Get recommendations for an existing user")

    known_users = sorted(cf_model.user_id_to_row.keys())
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.selectbox("Select a user_id", known_users, key="user_select")
    with col2:
        top_n = st.slider("Number of recommendations", 5, 30, 10, key="user_topn")

    alpha = st.slider(
        "Hybrid weight (alpha) -- 0 = pure content-based, 1 = pure collaborative",
        0.0, 1.0, config.DEFAULT_ALPHA, 0.05, key="user_alpha",
    )

    if st.button("Recommend", key="user_recommend_btn"):
        with st.spinner("Scoring candidates..."):
            recs = recommend_hybrid(content_model, cf_model, user_id, top_n=top_n, alpha=alpha)
            recs = enrich(recs, tracks_df)

        st.write(f"**Top {top_n} recommendations for `{user_id}`** (alpha = {alpha:.2f})")
        st.dataframe(
            recs[["name", "artist", "genre", "year", "hybrid_score", "content_score_norm", "cf_score_norm"]]
            .rename(columns={
                "name": "Track", "artist": "Artist", "genre": "Genre", "year": "Year",
                "hybrid_score": "Hybrid Score", "content_score_norm": "Content", "cf_score_norm": "Collaborative",
            }),
            use_container_width=True, hide_index=True,
        )

        with st.expander("This user's listening history (training data)"):
            history_ids = list(cf_model.user_played_tracks.get(user_id, []))
            hist_df = tracks_df[tracks_df["track_id"].isin(history_ids)][["name", "artist", "genre", "year"]]
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2: cold-start / brand-new user
# ---------------------------------------------------------------------------
with tab_cold_start:
    st.subheader("Get recommendations without any listening history")
    st.caption(
        "Pick a few songs you like. We build a taste profile from their audio "
        "features and tags/genre and find the closest matches -- this is exactly "
        "what happens for a brand-new user who has no play history yet."
    )

    song_options = (tracks_df["name"] + " -- " + tracks_df["artist"] + " [" + tracks_df["track_id"] + "]").tolist()
    picked = st.multiselect("Songs you like", song_options, key="cold_start_songs")
    top_n_cs = st.slider("Number of recommendations", 5, 30, 10, key="cold_start_topn")

    if st.button("Recommend", key="cold_start_btn"):
        if not picked:
            st.warning("Pick at least one song first.")
        else:
            seed_ids = [p.rsplit("[", 1)[1].rstrip("]") for p in picked]
            with st.spinner("Scoring candidates..."):
                recs = recommend_hybrid(
                    content_model, cf_model, user_id="__cold_start__",
                    top_n=top_n_cs, seed_track_ids=seed_ids,
                )
                recs = enrich(recs, tracks_df)
            st.write(f"**Top {top_n_cs} content-based recommendations**")
            st.dataframe(
                recs[["name", "artist", "genre", "year", "hybrid_score"]]
                .rename(columns={"name": "Track", "artist": "Artist", "genre": "Genre",
                                  "year": "Year", "hybrid_score": "Score"}),
                use_container_width=True, hide_index=True,
            )

# ---------------------------------------------------------------------------
# Tab 3: pure content-based "find similar songs"
# ---------------------------------------------------------------------------
with tab_similar:
    st.subheader("Find songs similar to a given song")
    song_options2 = (tracks_df["name"] + " -- " + tracks_df["artist"] + " [" + tracks_df["track_id"] + "]").tolist()
    query = st.selectbox("Pick a song", song_options2, key="similar_song_select")
    top_n_sim = st.slider("Number of similar songs", 5, 30, 10, key="similar_topn")

    if st.button("Find similar songs", key="similar_btn"):
        query_id = query.rsplit("[", 1)[1].rstrip("]")
        sims = similar_tracks(content_model, query_id, top_n=top_n_sim)
        sims = enrich(sims, tracks_df)
        st.dataframe(
            sims[["name", "artist", "genre", "year", "content_score"]]
            .rename(columns={"name": "Track", "artist": "Artist", "genre": "Genre",
                              "year": "Year", "content_score": "Similarity"}),
            use_container_width=True, hide_index=True,
        )

# ---------------------------------------------------------------------------
# Tab 4: evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("Compare content-based, collaborative and hybrid methods")
    st.caption(
        f"Leave-{config.TEST_HOLDOUT_PER_USER}-out evaluation: for each user, "
        f"{config.TEST_HOLDOUT_PER_USER} of their plays are held out as ground "
        f"truth and Precision@{config.TOP_N_EVAL} / Recall@{config.TOP_N_EVAL} / "
        f"F1@{config.TOP_N_EVAL} are measured on the remaining candidates."
    )
    max_eval_users = st.slider("Number of test users to evaluate on", 50, 1000, 300, step=50)

    if st.button("Run evaluation", key="eval_btn"):
        with st.spinner("Splitting data, retraining on the training split, and scoring..."):
            train_df, test_df = leave_n_out_split(interactions_df)
            eval_content_model = build_content_model(tracks_df)  # content model needs no interactions
            eval_cf_model = build_cf_model(train_df)
            results = compare_methods(eval_content_model, eval_cf_model, test_df, max_users=max_eval_users)

        st.dataframe(
            results.rename(columns={
                "method": "Method", "alpha": "Alpha",
                "precision_at_n": f"Precision@{config.TOP_N_EVAL}",
                "recall_at_n": f"Recall@{config.TOP_N_EVAL}",
                "f1_at_n": f"F1@{config.TOP_N_EVAL}",
                "n_users_evaluated": "Users Evaluated",
            }),
            use_container_width=True, hide_index=True,
        )
        st.bar_chart(results.set_index("method")[["precision_at_n", "recall_at_n", "f1_at_n"]])

st.divider()
st.caption(
    f"Catalogue: {len(tracks_df):,} tracks &nbsp;|&nbsp; "
    f"Listening history: {len(interactions_df):,} interactions from "
    f"{interactions_df['user_id'].nunique():,} users on "
    f"{interactions_df['track_id'].nunique():,} tracks (after filtering, see src/config.py)"
)
