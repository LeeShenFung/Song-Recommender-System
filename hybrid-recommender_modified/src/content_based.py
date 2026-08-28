import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

class ContentBasedRecommender:
    def __init__(self, art_dir="artifacts"):
        print("Loading Content-Based model and data...")
        self.music_info = pd.read_csv(os.path.join(art_dir, "music_info.csv"))
        self.content_matrix = sp.load_npz(os.path.join(art_dir, "content_matrix.npz"))
        self.nn_model = joblib.load(os.path.join(art_dir, "content_nn.joblib"))

    def recommend(self, seed_track_idx, top_n=5):
        song_vector = self.content_matrix[seed_track_idx]
        
        distances, indices = self.nn_model.kneighbors(song_vector, n_neighbors=top_n + 1)
        
        recommended_indices = indices[0][1:] 
        recommended_distances = distances[0][1:] 
        
        results = []
        for idx, dist in zip(recommended_indices, recommended_distances):
            row = self.music_info.iloc[idx]
            sim_score = 1.0 / (1.0 + dist)
            results.append({
                "name": row["name"],
                "artist": row["artist"],
                "genre": row.get("genre", "None"), 
                "score": sim_score                 
            })
        return results

    def evaluate(self, history_source, k=10, n_users=500, seed=42):
        rng = np.random.default_rng(seed)
        
        valid_users = []
        for u in range(history_source.n_users):
            tracks, _ = history_source.user_history(u)
            if len(tracks) >= 2:
                valid_users.append(u)
                
        if len(valid_users) > n_users:
            eval_users = rng.choice(valid_users, size=n_users, replace=False)
        else:
            eval_users = valid_users
            
        hits = 0
        for u in eval_users:
            tracks, counts = history_source.user_history(u)
            order = np.argsort(-counts)
            sorted_tracks = tracks[order]
            
            target_track = sorted_tracks[0] 
            seed_track = sorted_tracks[1]   
            
            song_vector = self.content_matrix[seed_track]
            distances, indices = self.nn_model.kneighbors(song_vector, n_neighbors=k + 1)
            
            rec_indices = indices[0][1:k+1] 
            if target_track in rec_indices:
                hits += 1
                
        n_eval = len(eval_users)
        precision = hits / (n_eval * k) if n_eval > 0 else 0.0
        recall = hits / n_eval if n_eval > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "hits": hits,
            "n_evaluated_users": n_eval,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "f1_at_k": f1
        }