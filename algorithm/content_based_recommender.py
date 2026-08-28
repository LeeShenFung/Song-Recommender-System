import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_merge_data(music_path="data/Music Info.csv", history_path="data/User Listening History.csv"):
    """
    Automated pipeline to load, merge, and extract features for ALL models.
    Ensures fair extraction time by pulling genre, artist, and playcount together.
    """
    # Load datasets
    music_df = pd.read_csv(music_path)
    listen_df = pd.read_csv(history_path)
    
    # Merge datasets on track_id
    merged_df = pd.merge(listen_df, music_df, on="track_id", how="inner")
    
    # Extract uniform features for all models (Fairness constraint)
    # Content-based uses: track_id, name, artist, tags/genre
    # Collab uses: user_id, track_id, playcount
    final_df = merged_df[['user_id', 'track_id', 'playcount', 'name', 'artist', 'tags', 'genre']]
    
    # Drop duplicates for the content-based item matrix so we only compute unique songs
    unique_songs = final_df.drop_duplicates(subset=['track_id']).reset_index(drop=True)
    
    # Clean text features: replace NaN with empty strings
    unique_songs['genre'] = unique_songs['genre'].fillna('')
    unique_songs['artist'] = unique_songs['artist'].fillna('')
    unique_songs['tags'] = unique_songs['tags'].fillna('')
    
    # Combine features into a single string for content evaluation
    unique_songs['combined_features'] = unique_songs['genre'] + " " + unique_songs['artist'] + " " + unique_songs['tags']
    
    return final_df, unique_songs

def build_content_model(unique_songs_df):
    """
    Builds the TF-IDF matrix and computes Cosine Similarity for songs.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(unique_songs_df['combined_features'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_content_recommendations(song_name, unique_songs_df, cosine_sim, top_n=5):
    """
    Fetches top N recommendations based on song name.
    """
    exact_match = unique_songs_df[unique_songs_df['name'].str.lower() == song_name.lower()]
    
    if not exact_match.empty:
        idx = exact_match.index[0]
    else:
        word_match = unique_songs_df[unique_songs_df['name'].str.contains(rf'\b{song_name}\b', case=False, regex=True, na=False)]
        
        if not word_match.empty:
            idx = word_match.index[0]
        else:
            return ["Song not found in dataset."]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[0:top_n]
    song_indices = [i[0] for i in sim_scores]
    
    return unique_songs_df[['name', 'artist', 'genre']].iloc[song_indices]