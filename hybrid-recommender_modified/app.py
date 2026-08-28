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
@st.cache_resource
def load_content_based():
    return ContentBasedRecommender(ART_DIR)

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

    st.title("🛠️ Developer")

    tab_dashboard, tab_cf, tab_content, tab_hybrid = st.tabs([
        "📊 Developer Dashboard",
        "🤝 Collaborative Filtering",
        "🎼 Content-Based",
        "🔀 Hybrid Recommender",
    ])

    # ------------------------------------------------------------------ #
    # Developer Dashboard — same input, three model outputs
    # ------------------------------------------------------------------ #
    with tab_dashboard:
        st.header("🎵 Model Comparison")
        st.caption(
            "Use the same existing user and seed song to compare the outputs "
            "of Content-Based, Item-Based Collaborative Filtering, and Hybrid."
        )

        # The comparison deliberately uses an EXISTING demo user because
        # collaborative/hybrid recommendation needs historical interactions.
        # New users with no history will always for content or hybrid
        demo = load_demo_users().copy()
        user_options = {
            f"{row['username']} — {row['activity_level']} ({int(row['n_interactions'])} songs)": int(row['user_idx'])
            for _, row in demo.iterrows()
        }

        dc1, dc2 = st.columns([2, 1])
        with dc1:
            user_type = st.radio(
                "User type",
                ["Existing User", "New User"],
                horizontal=True,
                key="dev_compare_user_type",
            )

            if user_type == "Existing User":
                demo = load_demo_users().copy()

                user_options = {
                    f"{row['username']} — {row['activity_level']} "
                    f"({int(row['n_interactions'])} songs)": int(row['user_idx'])
                    for _, row in demo.iterrows()
                }

                selected_user_label = st.selectbox(
                    "Select existing user",
                    list(user_options.keys()),
                    key="dev_compare_user",
                )

                compare_user_idx = user_options[selected_user_label]

            else:
                compare_user_idx = -1

                st.info(
                    "New user has no listening history. "
                    "Collaborative Filtering cannot provide personalised recommendations."
                )

        with dc2:
            compare_top_n = st.number_input(
                "Recommendations per model",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                key="dev_compare_top_n",
            )

        compare_query = st.text_input(
            "🔍 Test a song input",
            placeholder="e.g. Baby",
            key="dev_compare_song_query",
        )

        compare_seed_idx = None
        if compare_query:
            compare_hits = rec.search_tracks(compare_query, limit=12)
            if not compare_hits:
                st.warning("No matching songs found in the catalogue.")
            else:
                compare_options = {
                    f"{h['name']} — {h['artist']} ({h['year']})": h['track_idx']
                    for h in compare_hits
                }
                compare_choice = st.selectbox(
                    "Select the exact song",
                    list(compare_options.keys()),
                    key="dev_compare_exact_song",
                )
                compare_seed_idx = compare_options[compare_choice]

        if st.button(
            "▶ Compare Models",
            type="primary",
            disabled=compare_seed_idx is None,
            key="run_model_comparison",
        ):
            cf = load_item_cf()
            if compare_user_idx == -1:
                listened_set = set()
            else:
                listened_tracks, _ = rec.user_history(compare_user_idx)
                listened_set = set(int(x) for x in listened_tracks.tolist())

            # 本來是feng的alforithm Content-only: alpha=1.0 according to recommender.py
            content_results = rec.recommend(
                user_idx=compare_user_idx,
                seed_track_idx=compare_seed_idx,
                top_n=int(compare_top_n),
                alpha=1.0,
                exclude_listened=True,
            )

            # User's Item-Based CF: seed-song cosine similarity from historical
            # playcount patterns; already-listened tracks are excluded here so
            # the comparison behaves like a recommendation list for this user.
            # Not for new user, they have no history and thus no collaborative signal.
            if compare_user_idx == -1:
                cf_results = []
            else:
                cf_results = cf.recommend_similar_tracks(
                    seed_track_idx=compare_seed_idx,
                    top_n=int(compare_top_n),
                    exclude_track_indices=listened_set,
                )

            # Hybrid: fixed 50/50 blend used by the user-facing prototype.
            hybrid_results = rec.recommend(
                user_idx=compare_user_idx,
                seed_track_idx=compare_seed_idx,
                top_n=int(compare_top_n),
                alpha=0.5,
                exclude_listened=True,
            )

            st.subheader("Model Comparison")
            col_content, col_cf, col_hybrid = st.columns(3)

            def _comparison_df(results, source):
                rows = []
                for r in results:
                    idx = int(r['track_idx'])
                    meta = rec.music_info.iloc[idx]
                    row = {
                        'name': r['name'],
                        'artist': r['artist'],
                        'genre': meta.get('genre', ''),
                    }
                    if source == 'cf':
                        row['score'] = r.get('similarity_score', 0.0)
                    else:
                        row['score'] = r.get('hybrid_score', 0.0)
                    rows.append(row)
                return pd.DataFrame(rows)

            with col_content:
                st.markdown("#### Content-Based")
                st.info(
                    "Content-Based Filtering implementation will be added here."
                )

                # ============================================================
                # TEAMMATE CONTENT-BASED ALGORITHM START
                #
                # Example:
                #
                content_model = load_content_based()

                content_results = content_model.recommend(
                    seed_track_idx=compare_seed_idx,
                    top_n=int(compare_top_n),
                )
                #
                content_df = pd.DataFrame(content_results)
                #
                st.dataframe(
                    content_df[['name', 'artist', 'score']],
                    use_container_width=True,
                    hide_index=True,
                )
                #
                # TEAMMATE CONTENT-BASED ALGORITHM END
                # ============================================================

            with col_cf:
                st.markdown("#### Collaborative Filtering")

                if compare_user_idx == -1:
                    st.info(
                        "Not available for new users because there is no listening history."
                    )
                else:
                    cf_df = _comparison_df(cf_results, 'cf')

                    if cf_df.empty:
                        st.warning(
                            "No CF signal for this seed song among the selected active users."
                        )
                    else:
                        st.dataframe(
                            cf_df[['name', 'artist', 'score']]
                            .style.format({'score': '{:.3f}'}),
                            use_container_width=True,
                            hide_index=True,
                        )

            with col_hybrid:
                st.markdown("#### Hybrid Model")
                hybrid_df = _comparison_df(hybrid_results, 'hybrid')
                st.dataframe(
                    hybrid_df.style.format({'score': '{:.3f}'}),
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "CF score = cosine similarity from user playcount patterns. "
                "Artist/genre are shown as metadata and are not used to calculate the CF similarity."
            )
    with tab_cf:
        st.header("🤝 Item-Based Collaborative Filtering")

        st.write(
            "This model recommends songs based on users' playcount patterns "
            "using item-based collaborative filtering and cosine similarity."
        )

        cf = load_item_cf()

        cf_query = st.text_input(
            "Search a song",
            placeholder="e.g. Baby",
            key="cf_tab_song_search",
        )

        cf_top_n = st.number_input(
            "Number of recommendations",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key="cf_tab_top_n",
        )

        cf_seed_idx = None

        if cf_query:
            hits = cf.search_tracks(cf_query, limit=12)

            if not hits:
                st.warning("No matching songs found.")
            else:
                options = {
                    f"{h['name']} — {h['artist']} ({h['year']})":
                    h["track_idx"]
                    for h in hits
                }

                selected = st.selectbox(
                    "Select the exact song",
                    list(options.keys()),
                    key="cf_tab_exact_song",
                )
                cf_seed_idx = options[selected]

            if st.button(
                "▶ Run Collaborative Filtering",
                type="primary",
                disabled=cf_seed_idx is None,
                key="cf_tab_run",
            ):

                results = cf.recommend_similar_tracks(
                    seed_track_idx=cf_seed_idx,
                    top_n=int(cf_top_n),
                )

                if not results:
                    st.warning("No collaborative signal found for this song.")

                else:
                    cf_df = pd.DataFrame(results)

                    st.dataframe(
                        cf_df[
                            ["name", "artist", "similarity_score"]
                        ].rename(
                            columns={"similarity_score": "score"}
                        ).style.format(
                            {"score": "{:.4f}"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

        st.divider()
        st.subheader("📊 Evaluation")

        ev1, ev2 = st.columns(2)

        cf_eval_k = ev1.slider(
            "K (Top-K)",
            min_value=5,
            max_value=30,
            value=10,
            key="cf_eval_k",
        )

        cf_eval_users = ev2.slider(
            "Number of evaluation users",
            min_value=100,
            max_value=3000,
            value=500,
            step=100,
            key="cf_eval_users",
        )

        st.caption(
            "Protocol: leave-one-out implicit feedback. "
            "Most-played song = ground truth; "
            "second most-played song = seed song."
        )

        if st.button(
            "▶ Run CF Evaluation",
            type="primary",
            key="run_cf_evaluation",
        ):

            import time

            with st.spinner(
                f"Evaluating Collaborative Filtering on "
                f"{cf_eval_users} users..."
            ):
                start_time = time.time()

                metrics = cf.evaluate(
                    history_source=rec,
                    k=int(cf_eval_k),
                    n_users=int(cf_eval_users),
                    seed=0,
                )
                elapsed = time.time() - start_time

            m1, m2, m3 = st.columns(3)

            m1.metric(
                f"Precision@{cf_eval_k}",
                f"{metrics['precision_at_k']:.4f}",
            )

            m2.metric(
                f"Recall@{cf_eval_k}",
                f"{metrics['recall_at_k']:.4f}",
            )

            m3.metric(
                f"F1@{cf_eval_k}",
                f"{metrics['f1_at_k']:.4f}",
            )

            st.write(
                f"**Hits:** {metrics['hits']} / "
                f"{metrics['n_evaluated_users']} users "
                f"· **Time:** {elapsed:.2f}s"
            )

        st.divider()
        st.subheader("Model / data summary")
        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Tracks in catalogue",
            f"{cf.n_items:,}"
        )

        s2.metric(
            "Active users used",
            f"{len(cf.top_users):,}"
        )

        s3.metric(
            "Similarity measure",
            "Cosine"
        )

        s4.metric(
            "Feedback type",
            "Playcount"
        )

    with tab_content:
        st.header("🎼 Content-Based Filtering")

        st.info(
            "This section is reserved for the Content-Based Filtering "
            "implementation and evaluation."
        )
        # ============================================================
        # TEAMMATE CONTENT-BASED ALGORITHM
        # ============================================================

        # Put teammate's recommendation demo here.


        st.divider()
        st.subheader("📊 Evaluation")

        # Put teammate's evaluation here.


        st.divider()
        st.subheader("Model / data summary")

        # Put teammate's model/data summary here.

        # ============================================================
        # END CONTENT-BASED SECTION
        # ============================================================

    # ------------------------------------------------------------------ #
    # Hybrid — KEEP the existing Hybrid evaluation tab unchanged
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
