import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

def load_and_train_cb(music_path):
    """加载音乐数据并训练 Content-Based 模型，返回计算好的数据和矩阵"""
    df = pd.read_csv(music_path)
    df = df.drop_duplicates(subset=['track_id']).reset_index(drop=True)
    
    # 清洗空值
    df['tags'] = df['tags'].fillna('')
    df['genre'] = df['genre'].fillna('')
    df['name'] = df['name'].fillna('')
    df['artist'] = df['artist'].fillna('')
    
    # 1. 文本特征 (TF-IDF)
    df['combined_text'] = (df['tags'] + " " + df['genre']).str.lower()
    tfidf = TfidfVectorizer(stop_words='english')
    text_matrix = tfidf.fit_transform(df['combined_text'])
    
    # 2. 数值特征 (MinMaxScaler)
    numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    df[numerical_features] = df[numerical_features].fillna(0)
    scaler = MinMaxScaler()
    num_matrix = scaler.fit_transform(df[numerical_features])
    
    # 3. 合并特征并计算余弦相似度
    combined_matrix = hstack([text_matrix, num_matrix])
    similarity_matrix = cosine_similarity(combined_matrix, combined_matrix)
    
    return df, similarity_matrix

def get_cb_recommendations(song_name, df, similarity_matrix, top_n=10):
    """输入歌名，返回 Top 10 推荐结果的 DataFrame"""
    song_name_lower = str(song_name).lower().strip()
    
    # 精确匹配歌名
    exact_match = df[df['name'].str.lower() == song_name_lower]
    
    if exact_match.empty:
        return pd.DataFrame({"Message": ["Song not found in Content-Based database."]})
        
    idx = exact_match.index[0]
    
    # 获取相似度分数并排序
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 提取前 N 首歌 (排除自己，所以从 1 开始)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    # 返回 UI 需要的干净格式
    result_df = df.iloc[top_indices][['name', 'artist', 'genre']].rename(
        columns={'name': 'Song Name', 'artist': 'Artist', 'genre': 'Genre'}
    )
    return result_df