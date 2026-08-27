import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from content_based_recommender import load_and_merge_data, build_content_model, get_content_recommendations

# --- 1. Load Data & Models (Cached for performance) ---
@st.cache_data
def initialize_system():
    # Load data using the unified extraction method
    full_df, unique_songs = load_and_merge_data()
    # Build Content-Based Model
    cosine_sim = build_content_model(unique_songs)
    return full_df, unique_songs, cosine_sim

full_df, unique_songs, cosine_sim = initialize_system()

# --- 2. Teammate Placeholders ---
def get_collab_recommendations(song_name, top_n=5):
    # TODO: Teammate inserts Collaborative Filtering logic here
    # Note: Collab filtering usually predicts based on user_id, 
    # if predicting by song_name, implement Item-Item CF.
    return pd.DataFrame({"name": ["Collab Song A", "Collab Song B"], "artist": ["Artist A", "Artist B"]})

def get_hybrid_recommendations(song_name, top_n=5):
    # TODO: Teammate inserts Hybrid logic here (the "Best" model)
    return pd.DataFrame({"name": ["Hybrid Song A", "Hybrid Song B"], "artist": ["Artist X", "Artist Y"]})

def get_evaluation_metrics():
    # TODO: Teammates populate actual test results here based on rubric metrics
    return {
        "Model": ["Content-Based", "Collaborative", "Hybrid"],
        "Precision": [0.65, 0.70, 0.82],
        "Recall": [0.55, 0.60, 0.78],
        "F1 Score": [0.59, 0.64, 0.80]
    }

# --- 3. UI Layout ---
st.set_page_config(page_title="Song Recommender System", layout="wide")

st.sidebar.title("Navigation")
role = st.sidebar.radio("Select your role:", ("User", "Developer"))

st.title("🎶 Song Recommender System")

if role == "User":
    st.header("User Mode")
    st.write("Discover your next favorite song based on what you already love!")
    
    search_query = st.text_input("🔍 Search for a song name:")
    
    if search_query:
        st.subheader("Results")
        # For the user, we only show the "Best" model based on evaluation
        results = get_hybrid_recommendations(search_query, top_n=5)
        st.table(results)

elif role == "Developer":
    st.header("Developer Dashboard")
    st.write("Compare model outputs and evaluate system accuracy.")
    
    search_query = st.text_input("🔍 Test a song input:")
    
    if search_query:
        st.subheader("Model Comparison")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Content-Based**")
            cb_results = get_content_recommendations(search_query, unique_songs, cosine_sim, top_n=5)
            if isinstance(cb_results, list): # Handle error string
                st.warning(cb_results[0])
            else:
                st.dataframe(cb_results, hide_index=True)
                
        with col2:
            st.markdown("**Collaborative Filtering**")
            cf_results = get_collab_recommendations(search_query, top_n=5)
            st.dataframe(cf_results, hide_index=True)
            
        with col3:
            st.markdown("**Hybrid Model**")
            hy_results = get_hybrid_recommendations(search_query, top_n=5)
            st.dataframe(hy_results, hide_index=True)
            
    st.divider()
    
    # Evaluation Graph Section
    st.subheader("📊 Evaluation Graph")
    metrics_data = get_evaluation_metrics()
    metrics_df = pd.DataFrame(metrics_data)
    
    st.dataframe(metrics_df, hide_index=True)
    
    # Plotting the metrics for the rubric requirement
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics_df["Model"]))
    width = 0.25
    
    ax.bar(x - width, metrics_df["Precision"], width, label='Precision')
    ax.bar(x, metrics_df["Recall"], width, label='Recall')
    ax.bar(x + width, metrics_df["F1 Score"], width, label='F1 Score')
    
    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Model"])
    ax.legend()
    
    st.pyplot(fig)