from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize as sk_normalize
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rapidfuzz import fuzz
import numpy as np
import faiss
import re
import os

from db_handler import get_all_messages, insert_message

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

encoder = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generator_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
generator = pipeline("text-generation", model=generator_model, tokenizer=tokenizer)

INDEX_PATH = "data/faiss.index"

dimension = 384
text_data = get_all_messages()

if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(dimension)
    if text_data:
        embeddings = encoder.encode([t.lower() for t in text_data])
        embeddings = sk_normalize(embeddings)
        index.add(np.array(embeddings))

def normalize_text(text):
    return re.sub(r'\d+', '<num>', text.lower())

def semantic_search(query, top_k=10, threshold=0.4):
    if not text_data:
        return []
    query_vec = encoder.encode([query])
    query_vec = sk_normalize(query_vec)
    distances, indices = index.search(query_vec, top_k)

    seen = set()
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= len(text_data):
            continue
        similarity = 1 - (dist / 4.0)
        if similarity >= threshold:
            text = text_data[idx]
            if text not in seen:
                seen.add(text)
                results.append((similarity, text))
    results.sort(key=lambda x: x[0], reverse=True)
    return results

def fuzzy_search(query, threshold=60):
    results = []
    for text in text_data:
        score = fuzz.token_set_ratio(query.lower(), text.lower())
        if score >= threshold:
            results.append((score / 100.0, text))
    results.sort(key=lambda x: x[0], reverse=True)
    return results

def cross_encoder_rerank(query, candidates):
    if not candidates:
        return []
    texts = [text for _, text in candidates]
    pairs = [[query, t] for t in texts]
    rerank_scores = reranker.predict(pairs)
    reranked = list(zip(rerank_scores, texts))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked

def generate_answer(query, context):
    prompt = f"Given the conversation:\n{context}\nAnswer the user's question:\n{query}"
    result = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return result[0]['generated_text'].strip()

def get_answer(query):
    print(f"🔍 Searching for: {query}")
    norm_query = normalize_text(query)
    results = semantic_search(norm_query)
    reranked = cross_encoder_rerank(norm_query, results)
    if not reranked:
        print("⚠️ No semantic results, trying fuzzy search...")
        reranked = fuzzy_search(norm_query)
        if not reranked:
            return "❌ Sorry, no relevant information found."
    top_text = reranked[0][1]
    print(f"📌 Top Context: {top_text}")
    return generate_answer(query, top_text)

def index_new_message(msg):
    norm_msg = normalize_text(msg)
    if norm_msg not in text_data:
        text_data.append(norm_msg)
        insert_message(norm_msg)
        embedding = encoder.encode([norm_msg])
        embedding = sk_normalize(embedding)
        index.add(embedding)
        faiss.write_index(index, INDEX_PATH)
        return True
    return False

__all__ = ["normalize_text", "get_answer", "index_new_message"]
