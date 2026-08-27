import streamlit as st
import pandas as pd
from content_based_recommender import load_and_train_cb, get_cb_recommendations

st.set_page_config(page_title="Music Recommender Engine", page_icon="🎵", layout="wide")

# ==========================================
# 1. 缓存加载数据 (避免每次打字都重新计算)
# ==========================================
@st.cache_resource(show_spinner="Training Content-Based Model...")
def init_cb_model():
    # 确保 data 文件夹下有 Music Info (1).csv
    return load_and_train_cb('data/Music Info.csv')

cb_df, cb_sim_matrix = init_cb_model()

# ==========================================
# 2. 接口函数 (队友在这里填入他们的代码)
# ==========================================
def run_collaborative_filtering(song_name):
    # TODO: 队友 A (Collaborative Filtering 负责人)
    # 请在这里接入你的 Item-Based CF 模型
    # 要求返回一个包含 'Song Name', 'Artist' 列的 pandas DataFrame
    
    # 这是一个假的数据位，等你填入真实代码后删除
    return pd.DataFrame({"Song Name": ["CF Rec 1", "CF Rec 2"], "Artist": ["CF Artist", "CF Artist"]})

def run_hybrid_model(song_name):
    # TODO: 队友 B (Hybrid Model 负责人)
    # 请在这里接入你的 Hybrid 模型 (CB + CF)
    # 要求返回一个包含 'Song Name', 'Artist' 列的 pandas DataFrame
    
    # 这是一个假的数据位，等你填入真实代码后删除
    return pd.DataFrame({"Song Name": ["Hybrid 1", "Hybrid 2"], "Artist": ["Hybrid Artist", "Hybrid Artist"]})

# ==========================================
# 3. 门卫系统 (身份选择)
# ==========================================
if 'role' not in st.session_state:
    st.session_state.role = None

if st.session_state.role is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.title("🎵 AI Music Recommender")
        st.markdown("### Please select your access level:")
        st.write("<br>", unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("👨‍💻 Developer Mode", use_container_width=True):
                st.session_state.role = "Developer"
                st.rerun()
        with btn_col2:
            if st.button("🎧 User Mode", use_container_width=True):
                st.session_state.role = "User"
                st.rerun()

# ==========================================
# 4. Developer Side (开发者测试视角)
# ==========================================
elif st.session_state.role == "Developer":
    with st.sidebar:
        st.success("Mode: **Developer**")
        if st.button("🚪 Switch Role", use_container_width=True):
            st.session_state.role = None
            st.rerun()
            
    st.title("👨‍💻 Developer Dashboard")
    st.caption("Compare 3 Models simultaneously by searching a song name.")
    
    search_query = st.text_input("🔍 Search a Song Name:", placeholder="e.g. Wonderwall")
    
    if search_query:
        st.divider()
        st.markdown(f"### Results for: **{search_query}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("1. Content-Based (Ready)")
            cb_result = get_cb_recommendations(search_query, cb_df, cb_sim_matrix)
            st.dataframe(cb_result, use_container_width=True, hide_index=True)
            
        with col2:
            st.subheader("2. Collaborative")
            st.caption("Waiting for teammate's CF code...")
            cf_result = run_collaborative_filtering(search_query)
            st.dataframe(cf_result, use_container_width=True, hide_index=True)
            
        with col3:
            st.subheader("3. Hybrid Model")
            st.caption("Waiting for teammate's Hybrid code...")
            hybrid_result = run_hybrid_model(search_query)
            st.dataframe(hybrid_result, use_container_width=True, hide_index=True)
            
    # --- 评估图表 (Evaluation Graph) ---
    st.divider()
    st.markdown("### 📊 Offline Evaluation Metrics")
    st.caption("TODO: 队友们，请在这里填入你们测试跑出来的真实分数")
    
    # 等你们在下面填入真实数字后，图表会自动生成
    eval_data = pd.DataFrame({
        "Model": ["Content-Based", "Collaborative", "Hybrid"],
        "Precision": [0.015, 0.000, 0.000], # 换成真实分数
        "Recall": [0.120, 0.000, 0.000],    # 换成真实分数
        "F1 Score": [0.026, 0.000, 0.000]   # 换成真实分数
    })
    
    col_table, col_chart = st.columns([1, 2])
    with col_table:
        st.dataframe(eval_data, use_container_width=True, hide_index=True)
    with col_chart:
        st.bar_chart(eval_data.set_index("Model"))

# ==========================================
# 5. User Side (最终用户视角)
# ==========================================
elif st.session_state.role == "User":
    with st.sidebar:
        st.info("Mode: **User**")
        if st.button("🚪 Switch Role", use_container_width=True):
            st.session_state.role = None
            st.rerun()
            
    st.title("🎧 Music Search Engine")
    st.caption("Discover your next favorite song.")
    
    search_query = st.text_input("🔍 What song do you like?", placeholder="Enter a song name...")
    
    if search_query:
        st.success(f"Curating playlist based on '{search_query}'...")
        
        # 目前暂时用 Content-Based 兜底，等队友 Hybrid 写好了，这里换成 run_hybrid_model
        best_results = get_cb_recommendations(search_query, cb_df, cb_sim_matrix)
        st.table(best_results)