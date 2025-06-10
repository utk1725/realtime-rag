from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize
import faiss
import numpy as np
import os
import json
from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv
load_dotenv()

print("✅ SLACK_BOT_TOKEN:", os.getenv("SLACK_BOT_TOKEN"))
print("✅ SLACK_APP_TOKEN:", os.getenv("SLACK_APP_TOKEN"))



print("⏳ Loading Flan-T5 model...")
MODEL_ID = "google/flan-t5-base"

flan_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
flan_pipe = pipeline("text2text-generation", model=flan_model, tokenizer=flan_tokenizer)

print("✅ Flan-T5 model loaded.")

import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import re

def normalize_text(text):
    return re.sub(r'\d+', '<num>', text.lower())

WATCH_FILE = "data/live_input.txt"
INDEX_PATH = "data/faiss.index"
MSG_PATH = "data/messages.json"

model = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
dimension = 384
index = None
text_data = []

class RealTimeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("live_input.txt"):
            with open(WATCH_FILE, "r") as f:
                lines = f.read().strip().split("\n")
                new_message = lines[-1].strip()
                if new_message and new_message not in text_data:
                    print(f"\n📥 New chat message: {new_message}")
                    add_message(new_message)

def start_watcher():
    event_handler = RealTimeHandler()
    observer = Observer()
    observer.schedule(event_handler, path='data', recursive=False)
    observer.start()
    print("👂 Real-time chat listener started...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def load_data():
    global text_data, index
    if os.path.exists(MSG_PATH):
        with open(MSG_PATH, "r") as f:
            text_data = json.load(f)
    else:
        text_data = []

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(dimension)
        if text_data:
            embeddings = model.encode(text_data)
            embeddings = normalize(embeddings)
            index.add(np.array(embeddings))

def save_data():
    with open(MSG_PATH, "w") as f:
        json.dump(text_data, f)
    faiss.write_index(index, INDEX_PATH)

def add_message(message):
    norm_msg = normalize_text(message)
    if norm_msg in [normalize_text(m) for m in text_data]:
        print("⚠️ Message already exists.")
        return
    embedding = model.encode([norm_msg])[0]
    embedding = normalize([embedding])[0]
    index.add(np.array([embedding]))
    text_data.append(message)
    save_data()
    print("✅ Message added and saved.")

def semantic_search(query, top_k=10, threshold=0.7):
    if len(text_data) == 0:
        return []
    query_vec = model.encode([query])
    query_vec = normalize(query_vec)
    D, I = index.search(query_vec, top_k)

    seen = set()
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx >= len(text_data):
            continue
        similarity = 1 - (score / 4.0)
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

def generate_answer(query, context):
    prompt = f"Given the conversation:\n{context}\nAnswer the user's question:\n{query}"
    result = flan_pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return result[0]['generated_text'].strip()

def rerank_results(query, candidates):
    if not candidates:
        return []
    texts = [text for _, text in candidates]
    pairs = [[query, t] for t in texts]
    rerank_scores = reranker.predict(pairs)
    reranked = list(zip(rerank_scores, texts))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked

def search(query, top_k=10, threshold=0.7, fuzzy_threshold=60):
    if len(text_data) == 0:
        print("⚠️ No data available.")
        return

    norm_query = normalize_text(query)

    semantic_results = semantic_search(norm_query, top_k=top_k, threshold=threshold)
    reranked_results = rerank_results(norm_query, semantic_results)

    if len(reranked_results) == 0:
        print("⚠️ No semantic matches found, trying fuzzy search...")
        fuzzy_results = fuzzy_search(norm_query, threshold=fuzzy_threshold)
        if not fuzzy_results:
            print("❌ No fuzzy matches found either.")
            return
        reranked_results = fuzzy_results

    if reranked_results:
        top_text = reranked_results[0][1]
        print("\n🔍 Top Match:")
        print(f"- {top_text}")
        answer = generate_answer(query, top_text)
        print("\n🤖 LLM Answer:")
        print(answer)
    else:
        print("❌ No results found.\n")

def main():
    load_data()
    watcher_thread = threading.Thread(target=start_watcher, daemon=True)
    watcher_thread.start()

    while True:
        print("📌 Options: [1] Add Message  [2] Search Query  [0] Exit")
        choice = input("Your choice: ").strip()
        if choice == "1":
            msg = input("Enter your message to store: ")
            add_message(msg)
        elif choice == "2":
            query = input("Enter your query: ")
            search(query)
        elif choice == "0":
            print("👋 Exiting.")
            break
        else:
            print("❌ Invalid choice.\n")

if __name__ == "__main__":
    main()
