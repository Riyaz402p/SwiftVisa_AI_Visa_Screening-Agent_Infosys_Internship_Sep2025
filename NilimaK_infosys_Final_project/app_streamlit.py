# SIR PLZ COPY THE PATH OF THIS CODE AND
# THEN OPEN COMMAND PROMPT AND TYPE -> streamlit run paste the copied code
# THEN PRESS ENTER , This takes you to the browser which shows the perfectly working streamlit deployment part .


import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from transformers import pipeline
import os

# ---------- compatibility for older/newer streamlit caching ----------
try:
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except Exception:
    # fallback for older Streamlit
    def cache_resource(func):
        return st.cache(allow_output_mutation=True)(func)
    def cache_data(func):
        return st.cache(func)

# ---------- Data / model loaders ----------
@cache_resource
def load_index_and_model(chunks_path):
    # chunks_path: path to jsonl where each line is {"text": "...", "embedding": [...], "source": "..."}
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    embeddings = np.array([c["embedding"] for c in chunks], dtype="float32")
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return model, index, chunks

@cache_resource
def load_generator(device=-1):
    # device = -1 for CPU, 0+ for GPU index
    return pipeline("text2text-generation",
                    model="google/flan-t5-base",
                    tokenizer="google/flan-t5-base",
                    device=device)

# ---------- Retrieval + generation (adapted from your notebook) ----------

# ---------- Retrieval + generation (adapted from your notebook) ----------
def retrieve_chunks(query, model, index, chunks, top_k=5, min_score=0.3):
    """
    Retrieve top chunks for a query from FAISS index.
    Only return chunks with similarity >= min_score.
    """
    # Encode query
    q_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    faiss.normalize_L2(q_emb.reshape(1, -1)) 
    # Search
    D, I = index.search(q_emb.reshape(1, -1), top_k)  
    results = []
    for score, idx in zip(D[0], I[0]):
        if score >= min_score:
            results.append(chunks[idx])
    return results


# -----------------------------
# 4) Answer generation
# -----------------------------
def generate_answer(query, model, index, chunks, top_k=5, generator=None, min_score=0.3, max_new_tokens=200):
    retrieved = retrieve_chunks(query, model, index, chunks, top_k=top_k, min_score=min_score)
    if not retrieved:
        return "No relevant information found.", []

    context = "\n\n".join([f"Source: {r.get('source', 'Unknown')}\n{r['text']}" for r in retrieved])
    citations = [f"{r.get('source', 'Unknown')} — {r['text']}" for r in retrieved]

    if generator is not None:
        prompt = f"""
        You are an expert eligibility officer.
        Using only the context below, answer the question truthfully.
        If the answer is not in the context, say "I cannot find relevant information."

        Context:
        {context}

        Question: {query}
        Answer:
        """
        output = generator(prompt, max_new_tokens=max_new_tokens)
        return output[0].get("generated_text", output[0].get("text", "")).strip(), citations

    answer = f"Here is what I found based on the documents:\n\n{context}"
    return answer, citations


# ---------- Streamlit UI ----------
st.set_page_config(page_title="RAG Visa-eligibility HMI", layout="wide")
st.title("RAG Visa-Eligibility — Streamlit HMI (Demo)")

# Sidebar config
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload chunks (.jsonl) — each line: {text, embedding, source}", type=["jsonl"])
default_path = st.sidebar.text_input("Chunks JSONL path (if no upload)", value=r"C:\Users\ASUS\OneDrive\Desktop\chunks_with_embeddings_v2.jsonl")
use_generator = st.sidebar.checkbox("Use Generator (LLM) for final answer", value=True)
device_option = st.sidebar.selectbox("Generator device", options=["cpu", "gpu"], index=0)
top_k = st.sidebar.slider("Top K retrieved chunks", 1, 10, 5)
max_new_tokens = st.sidebar.slider("Generator: max_new_tokens", 50, 1000, 200)

# load resources (uploaded file overrides the path)
chunks_path = None
if uploaded is not None:
    # save temporarily
    tmp = "uploaded_chunks.jsonl"
    with open(tmp, "wb") as f:
        f.write(uploaded.getbuffer())
    chunks_path = tmp
else:
    chunks_path = default_path

st.sidebar.markdown("**Index path:**")
st.sidebar.code(chunks_path)

# Load models / index (cached)
status = st.empty()
with status.container():
    st.write("Loading embedding model + FAISS index (cached) ...")
try:
    model, index, chunks = load_index_and_model(chunks_path)
    st.success(f"Loaded index with {len(chunks)} chunks (embedding dim = {index.d})")
except Exception as e:
    st.error(f"Failed to load index/model: {e}")
    st.stop()

generator = None
if use_generator:
    dev = -1 if device_option == "cpu" else 0
    try:
        with st.spinner("Loading generator..."):
            generator = load_generator(device=dev)
    except Exception as e:
        st.error(f"Failed to load generator: {e}")
        generator = None

# Query input
st.markdown("### Ask a question about visa eligibility")
query = st.text_area("Enter your question", value="What are the eligibility requirements for a UK Student Visa?", height=120)
ask = st.button("Get Answer")

if ask and query.strip():
    with st.spinner("Retrieving top chunks ..."):
        retrieved = retrieve_chunks(query, model, index, chunks, top_k=top_k)
    # show retrieved pieces
    st.markdown("#### Retrieved context (top results)")
    for i, r in enumerate(retrieved, start=1):
        src = r.get("source", "Unknown")
        st.markdown(f"**{i}. Source:** {src}")
        st.write(r.get("text", "[no text]"))

    # generate final answer
    with st.spinner("Generating answer ..."):
        answer, citations = generate_answer(query, model, index, chunks, top_k=top_k, generator=generator, max_new_tokens=max_new_tokens)

    st.markdown("### Final Answer")
    st.write(answer)

    st.markdown("### Citations / matched chunks")
    for c in citations:
        st.write("- ", c[:1000])  # truncate very long chunks in UI

    st.download_button("Download answer + citations (txt)", data=answer + "\\n\\nCITATIONS:\\n" + "\\n".join(citations),
                       file_name="answer_and_citations.txt")
else:
    st.info("Enter a question and click **Get Answer**. You can upload a chunks .jsonl or provide a local path in the sidebar.")


# SIR PLZ COPY THE PATH OF THIS CODE AND
# THEN OPEN COMMAND PROMPT AND TYPE -> streamlit run paste the copied code
# THEN PRESS ENTER , This takes you to the browser which shows the perfectly working streamlit deployment part .