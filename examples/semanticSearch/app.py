import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Endee Semantic Search", page_icon="🔎", layout="wide")

st.markdown("""
<style>

.main-title {
font-size:48px;
font-weight:700;
background: linear-gradient(90deg,#4A90E2,#6C63FF);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.subtitle {
font-size:22px;
color:#555;
margin-bottom:30px;
}

.result-card {
background-color:#f8f9fc;
padding:20px;
border-radius:12px;
margin-bottom:20px;
border-left:6px solid #6C63FF;
box-shadow:0px 4px 10px rgba(0,0,0,0.05);
}

.source-tag {
background:#eef2ff;
padding:5px 10px;
border-radius:6px;
font-size:13px;
font-weight:600;
color:#4A4A4A;
}

</style>
""", unsafe_allow_html=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("embeddings.json","r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
vectors = np.array([item["embedding"] for item in data])

st.markdown('<div class="main-title">🔎 Endee Semantic Search</div>', unsafe_allow_html=True)

st.markdown(
"""
<div class="subtitle">
AI-powered document search using vector embeddings and semantic similarity.
</div>
""",
unsafe_allow_html=True
)

st.divider()
st.markdown("💡 **Try these queries:**")
st.markdown("- What is machine learning?")
st.markdown("- Explain artificial intelligence")
st.markdown("- Why is Python popular?")

st.divider()

query = st.text_input("💬 Ask a question about the documents")

if query:
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, vectors)[0]
    top_indices = similarities.argsort()[-5:][::-1]
    context = ""
    for idx in top_indices:
        context += texts[idx] + ". "

    st.markdown("## 🤖 Generated Answer")

    answer = f"""
Based on the retrieved documents, here is a summary:

{context}
"""

    st.info(answer)

    st.markdown("## 🔎 Search Results")

    for idx in top_indices:

        score = similarities[idx]

        st.markdown(
        f"""
        <div class="result-card">
        <span class="source-tag">📄 {data[idx]["source"]}</span>
        <p style="margin-top:10px;font-size:16px;">{texts[idx]}</p>
        </div>
        """,
        unsafe_allow_html=True
        )

        st.progress(float(score))
        st.caption(f"Relevance score: {score:.3f}")

st.divider()

# Sidebar
with st.sidebar:
    st.title("📚 About")
    st.write("""
This demo shows **semantic search** using embeddings.

Steps:
1️⃣ Documents → embeddings  
2️⃣ User query → embedding  
3️⃣ Cosine similarity search  
4️⃣ Retrieve relevant results
""")

    st.write("⚙️ Model: all-MiniLM-L6-v2")