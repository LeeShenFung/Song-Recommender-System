"""
Hybrid Music Recommender System — Streamlit prototype
=======================================================
Flow implemented (as required by the assignment brief):

  1. USER LOGIN  ->  2. type a seed song + choose how many recommendations
                     ->  3. hybrid (content-based + collaborative filtering)
                         recommendations are shown

  Separately:
  DEVELOPER LOGIN  -> run offline evaluation (Precision@K, Recall@K, F1@K)
                       to assess the recommender's efficiency and accuracy.

Run with:  streamlit run app.py
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from recommender import HybridRecommender  # noqa: E402
from item_cf import ItemBasedCF  # noqa: E402
from evaluate import run_comparison, run_comparison_cold_start  # noqa: E402

ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

st.set_page_config(page_title="Hybrid Music Recommender", page_icon="🎵", layout="wide")


# --------------------------------------------------------------------------- #
# cached resources / data
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading recommender model & data ...")
def load_recommender():
    return HybridRecommender(ART_DIR)


@st.cache_resource(show_spinner="Loading item-based collaborative filtering ...")
def load_item_cf():
    return ItemBasedCF(ART_DIR, max_users=1000)


@st.cache_data
def load_demo_users():
    return pd.read_csv(os.path.join(ART_DIR, "demo_users.csv"))


@st.cache_data
def load_dev_credentials():
    import joblib

    return joblib.load(os.path.join(ART_DIR, "dev_credentials.joblib"))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# session state
# --------------------------------------------------------------------------- #
for key, default in [("role", None), ("user_idx", None), ("username", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def logout():
    st.session_state["role"] = None
    st.session_state["user_idx"] = None
    st.session_state["username"] = None


# --------------------------------------------------------------------------- #
# LOGIN SCREEN
# --------------------------------------------------------------------------- #
def login_screen():
    st.title("🎵 Hybrid Music Recommender")
    st.caption("Content-based filtering + Collaborative filtering, blended.")

    tab_user, tab_new, tab_dev = st.tabs(
        ["👤 Existing User Login", "🆕 New User (no history)", "🛠️ Developer Login"]
    )

    with tab_user:
        demo = load_demo_users()
        with st.form("user_login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            row = demo[demo["username"] == username.strip()]
            if not row.empty and row.iloc[0]["password_hash"] == sha256(password):
                st.session_state["role"] = "user"
                st.session_state["username"] = username.strip()
                st.session_state["user_idx"] = int(row.iloc[0]["user_idx"])
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_new:
        st.write(
            "Sign up with **no listening history at all** -- this simulates a "
            "brand-new user (the *cold-start* case). No password needed since "
            "this is a demo account that only lives for this session."
        )
        with st.form("new_user_form"):
            new_username = st.text_input(
                "Pick a display name", placeholder="e.g. guest_alex"
            )
            new_submitted = st.form_submit_button(
                "Continue as a new user", type="primary"
            )
        if new_submitted:
            if not new_username.strip():
                st.error("Please enter a display name.")
            else:
                st.session_state["role"] = "user"
                st.session_state["username"] = new_username.strip()
                st.session_state["user_idx"] = -1  # sentinel: no history at all
                st.rerun()

    with tab_dev:
        st.write(
            "Developer access is used to **test and evaluate** the "
            "recommender's efficiency and accuracy (Precision / Recall / F1)."
        )
        with st.form("dev_login_form"):
            dev_user = st.text_input("Developer username")
            dev_pass = st.text_input("Developer password", type="password")
            dev_submit = st.form_submit_button("Log in as developer", type="primary")
        if dev_submit:
            creds = load_dev_credentials()
            if dev_user.strip() in creds and creds[dev_user.strip()] == sha256(
                dev_pass
            ):
                st.session_state["role"] = "dev"
                st.session_state["username"] = dev_user.strip()
                st.rerun()
            else:
                st.error("Invalid developer credentials.")


# --------------------------------------------------------------------------- #
# USER PAGE — get recommendations
# --------------------------------------------------------------------------- #
def user_page(rec: HybridRecommender):
    user_idx = st.session_state["user_idx"]
    is_new_user = user_idx == -1

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['username']}")
        if is_new_user:
            st.warning("🆕 New user — no listening history yet (cold start).")
        else:
            demo = load_demo_users()
            my_row = demo[demo["username"] == st.session_state["username"]].iloc[0]
            st.write(f"Activity level: **{my_row['activity_level']}**")
            st.write(f"Songs in listening history: **{my_row['n_interactions']}**")
        st.divider()
        # Fixed hybrid weight -- users get a straightforward 50/50 blend of
        # content-based and collaborative filtering; the alpha slider is a
        # developer/tuning concept and is intentionally not exposed here.
        alpha = 0.5
        exclude_listened = st.checkbox("Exclude songs I've already played", value=True)
        if st.button("Log out"):
            logout()
            st.rerun()

    st.title("🎵 Get song recommendations")
    if is_new_user:
        st.info(
            "Since you're a brand-new user with no listening history, the "
            "collaborative-filtering part of the model has nothing to learn from "
            "yet (its score is a flat constant for every song). Recommendations "
            "below are effectively driven by **content similarity** to your seed "
            "song only — this is the classic *cold-start problem* in "
            "recommender systems. Log in as an existing demo user instead to see "
            "the full hybrid effect."
        )
    st.write("Step 1 — type a song. Step 2 — choose how many recommendations you want.")

    query = st.text_input("🔎 Song name (or artist)", placeholder="e.g. Mr. Brightside")
    seed_track_idx = None
    if query:
        hits = rec.search_tracks(query, limit=12)
        if not hits:
            st.warning("No matching songs found in the catalogue. Try another title.")
        else:
            options = {
                f"{h['name']} — {h['artist']} ({h['year']})": h["track_idx"]
                for h in hits
            }
            choice = st.selectbox("Select the exact song", list(options.keys()))
            seed_track_idx = options[choice]

    top_n = st.number_input(
        "How many recommendations do you want?",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
    )

    if st.button("🎯 Recommend", type="primary", disabled=seed_track_idx is None):
        results = rec.recommend(
            user_idx=user_idx,
            seed_track_idx=seed_track_idx,
            top_n=int(top_n),
            alpha=alpha,
            exclude_listened=exclude_listened,
        )
        if not results:
            st.info(
                "No recommendations found (try unchecking 'exclude already played')."
            )
        else:
            st.subheader(f"Top {len(results)} recommendations for you")
            for i, r in enumerate(results, 1):
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(
                            f"**{i}. {r['name']}** — {r['artist']} ({r['year']})"
                        )
                        if isinstance(r["tags"], str) and r["tags"]:
                            st.caption(r["tags"])
                        st.progress(
                            r["hybrid_score"],
                            text=f"hybrid score {r['hybrid_score']:.2f}",
                        )
                        st.caption(
                            f"content similarity {r['content_score']:.2f}  ·  "
                            f"collaborative similarity {r['cf_score']:.2f}"
                        )
                    with c2:
                        if isinstance(r["preview_url"], str) and r[
                            "preview_url"
                        ].startswith("http"):
                            st.audio(r["preview_url"])


# --------------------------------------------------------------------------- #
# DEVELOPER PAGE — evaluation
# --------------------------------------------------------------------------- #
def dev_page(rec: HybridRecommender):
    with st.sidebar:
        st.markdown(f"### 🛠️ {st.session_state['username']} (developer)")
        if st.button("Log out"):
            logout()
            st.rerun()

    st.title("🛠️ Developer Dashboard")

    tab_cf, tab_content, tab_hybrid = st.tabs([
        "🤝 Collaborative Filtering",
        "🎼 Content-Based Filtering",
        "🔀 Hybrid Recommender",
    ])

    # ------------------------------------------------------------------ #
    # Collaborative Filtering — user's Item-Based CF
    # ------------------------------------------------------------------ #
    with tab_cf:
        st.header("🤝 Item-Based Collaborative Filtering")
        st.write(
            "This backend compares songs using users' playcount patterns. "
            "Cosine similarity is calculated between item-user vectors; "
            "song metadata is used only for displaying the results."
        )

        st.markdown(
            "**Flow:** interactions → Item × User matrix → "
            "cosine similarity → Top-N similar songs"
        )

        cf = load_item_cf()

        c1, c2 = st.columns([3, 1])
        with c1:
            cf_query = st.text_input(
                "Search a song (title or artist)",
                placeholder="e.g. Mr. Brightside",
                key="cf_song_search",
            )
        with c2:
            cf_top_n = st.number_input(
                "Top-N",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                key="cf_top_n",
            )

        seed_track_idx = None
        if cf_query:
            hits = cf.search_tracks(cf_query, limit=12)
            if not hits:
                st.warning("No matching songs found.")
            else:
                options = {}
                for h in hits:
                    signal = "" if h["has_cf_signal"] else " [no CF signal in selected users]"
                    label = f"{h['name']} — {h['artist']} ({h['year']}){signal}"
                    options[label] = h["track_idx"]

                choice = st.selectbox(
                    "Select the exact song",
                    list(options.keys()),
                    key="cf_exact_song",
                )
                seed_track_idx = options[choice]

        if st.button(
            "▶ Run Item-Based CF",
            type="primary",
            disabled=seed_track_idx is None,
            key="run_item_cf",
        ):
            with st.spinner("Calculating cosine similarity ..."):
                results = cf.recommend_similar_tracks(
                    seed_track_idx=seed_track_idx,
                    top_n=int(cf_top_n),
                )

            if not results:
                st.warning(
                    "This seed song has no collaborative signal among the "
                    "selected active users. Try another song."
                )
            else:
                df_cf = pd.DataFrame(results)
                show_cols = [
                    "name", "artist", "genre", "year", "similarity_score"
                ]
                st.subheader(f"Top {len(df_cf)} similar songs")
                st.dataframe(
                    df_cf[show_cols].style.format(
                        {"similarity_score": "{:.4f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()
        st.caption(
            "Current prototype uses the top 1,000 most active users, matching "
            "the Item-Based CF notebook. Evaluation metrics will be added here "
            "after the group's final common testing protocol is fixed."
        )

    # ------------------------------------------------------------------ #
    # Content-Based — teammate's section
    # ------------------------------------------------------------------ #
    with tab_content:
        st.header("🎼 Content-Based Filtering")
        st.info(
            "Place the Content-Based Filtering backend and its evaluation "
            "here when the teammate's final implementation is ready."
        )

    # ------------------------------------------------------------------ #
    # Hybrid — Feng's existing developer evaluation
    # ------------------------------------------------------------------ #
    with tab_hybrid:
        st.header("🔀 Hybrid Recommender Evaluation")

        st.markdown(
            "**Protocol (leave-one-out, implicit feedback):** for each evaluated user, "
            "their most-played song is hidden as ground truth, their second most-played "
            "song is used as the seed input, and the rest of their history feeds the "
            "collaborative profile. A hit is counted if the hidden song appears in the "
            "Top-K recommendation list."
        )

        col1, col2, col3 = st.columns(3)
        k = col1.slider("K (top-K recommendations)", 5, 30, 10)
        n_users = col2.slider("Number of evaluation users (sampled)", 100, 3000, 500, 100)
        alphas_str = col3.text_input(
            "Alpha values to compare (comma-separated)",
            "0.0, 0.25, 0.5, 0.75, 1.0",
        )

        if st.button("▶ Run evaluation", type="primary"):
            try:
                alphas = tuple(float(a.strip()) for a in alphas_str.split(","))
            except ValueError:
                st.error("Could not parse alpha values.")
                return

            with st.spinner(f"Evaluating on {n_users} held-out users for each alpha ..."):
                rows = run_comparison(rec, k=k, n_users=n_users, alphas=alphas)

            df = pd.DataFrame(rows)
            df["config"] = df["alpha"].apply(
                lambda a: "Collaborative-only" if a == 0 else
                          ("Content-only" if a == 1 else f"Hybrid (α={a})")
            )

            st.subheader("Results")
            st.dataframe(
                df[[
                    "config", "alpha", "k", "n_evaluated_users",
                    "precision_at_k", "recall_at_k", "f1_at_k", "seconds"
                ]],
                use_container_width=True,
            )

            chart_df = df.set_index("config")[[
                "precision_at_k", "recall_at_k", "f1_at_k"
            ]]
            st.bar_chart(chart_df)

            best = df.loc[df["f1_at_k"].idxmax()]
            st.success(
                f"Best F1@{k}: **{best['f1_at_k']:.4f}** "
                f"at alpha = {best['alpha']} ({best['config']})"
            )

        st.divider()
        st.subheader("🆕 Cold-start (new user) evaluation")
        st.markdown(
            "This measures what a **brand-new user with zero listening history** "
            "would experience across different `alpha` (hybrid weight) settings. "
            "Because a new user's collaborative fold-in vector is all-zero, their "
            "collaborative score is a flat constant across every track."
        )

        cs_col1, cs_col2, cs_col3 = st.columns(3)
        cs_k = cs_col1.slider("K (top-K)", 5, 30, 10, key="cs_k")
        cs_n = cs_col2.slider(
            "Number of evaluation users (sampled)",
            100, 3000, 500, 100,
            key="cs_n",
        )
        cs_alphas_str = cs_col3.text_input(
            "Alpha values to compare",
            "0.0, 0.25, 0.5, 0.75, 1.0",
            key="cs_alphas",
        )

        if st.button("▶ Run cold-start evaluation"):
            try:
                cs_alphas = tuple(
                    float(a.strip()) for a in cs_alphas_str.split(",")
                )
            except ValueError:
                st.error("Could not parse alpha values.")
            else:
                with st.spinner(
                    f"Simulating new-user recommendations for {cs_n} users ..."
                ):
                    cs_rows = run_comparison_cold_start(
                        rec,
                        k=cs_k,
                        n_users=cs_n,
                        alphas=cs_alphas,
                    )

                cs_df = pd.DataFrame(cs_rows)
                cs_df["config"] = cs_df["alpha"].apply(
                    lambda a: "Collaborative-only" if a == 0 else
                              ("Content-only" if a == 1 else f"Hybrid (α={a})")
                )

                st.dataframe(
                    cs_df[[
                        "config", "alpha", "k", "n_evaluated_users",
                        "precision_at_k", "recall_at_k", "f1_at_k", "seconds"
                    ]],
                    use_container_width=True,
                )

                cs_chart_df = cs_df.set_index("config")[[
                    "precision_at_k", "recall_at_k", "f1_at_k"
                ]]
                st.bar_chart(cs_chart_df)

        st.divider()
        st.subheader("Model / data summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tracks in catalogue", f"{rec.n_items:,}")
        m2.metric("Registered users (with history)", f"{rec.n_users:,}")
        m3.metric("Total interactions", f"{len(rec.inter_user):,}")
        m4.metric("CF latent dimensions", rec.item_factors.shape[1])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    if st.session_state["role"] is None:
        login_screen()
        return

    rec = load_recommender()
    if st.session_state["role"] == "user":
        user_page(rec)
    elif st.session_state["role"] == "dev":
        dev_page(rec)


if __name__ == "__main__":
    main()
