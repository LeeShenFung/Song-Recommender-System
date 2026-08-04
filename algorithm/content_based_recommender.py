import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def run_recommender():
    data = pd.read_csv('data/spotify_2015_2025_85k.csv')

    feature_columns = ['danceability', 'energy', 'tempo', 'instrumentalness', 'loudness']

    features = data[feature_columns].fillna(0)

    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    def recommend_songs(song_title, top_n=5):
        matched_songs = data[data['track_name'] == song_title]

        if matched_songs.empty:
            return f"No songs found with the title '{song_title}'."
        
        song_index = matched_songs.index[0]
        target_vector = normalized_features[song_index].reshape(1,-1)
        sim_scores = cosine_similarity(target_vector,normalized_features)[0]
        scores = list(enumerate(sim_scores))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        recommended_indices = [i[0] for i in sorted_scores]
        return data[['track_name','artist_name','genre']].iloc[recommended_indices]
    return recommend_songs

if __name__ == '__main__':
    recommender = run_recommender()
    test_song = "Husband"
    print("Recommendations for:", test_song)
    print(recommender(test_song))