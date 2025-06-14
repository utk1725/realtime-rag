
````markdown
 🧠 Real-Time Slack-integrated Retrieval-Augmented Generation (RAG) System

This project is a real-time, production-level RAG system designed to **learn from live Slack chats**, store them in a **local SQLite database**, and **respond instantly** to user queries within Slack channels.
It uses **semantic search** with **FAISS** and **SentenceTransformers**, a **CrossEncoder reranker**, and **TinyLLaMA** for final response generation — all deployed locally in a resource-efficient Docker container.

---

 🚀 Features

- ✅ Real-time chat ingestion from specified Slack channels
- 🔍 Semantic and fuzzy search with CrossEncoder reranking
- 🧠 Local knowledge base auto-updated via SQLite
- 🧾 Responses generated using TinyLLaMA 1.1B (Flan-T5 fallback supported)
- 🐳 Dockerized for easy deployment
- 📥 Supports live updates with no downtime
- ⚡ Slack bot integration using `slack_bolt`

---

 🧩 Project Structure & Modes

This project is **modular** and can operate in two modes:

| Component           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `app.py`            | **Standalone CLI RAG engine**. Can be used without Slack and integrated with|
|                     |any platform like MS Teams, internal chat tools, or REST APIs. Great for     |
|                     |testing, batch indexing, or non-Slack environments.                          |
| `slack_listener.py` | **Slack-integrated version**. Listens to Slack messages in real time and    |
|                     | uses `app.py` as the backend to serve answers directly within Slack threads.|
|                     | Acts as a plug-and-play example of real-world usage.                        |

 ✅ You can integrate `app.py` into any custom frontend, chatbot, or third-party system.

---

 🛠️ Tech Stack

| Layer         | Tools/Libs Used                                 |
|--------------|--------------------------------------------------|
| Embedding     | `sentence-transformers/all-MiniLM-L6-v2`        |
| Vector Search | `FAISS`                                         |
| Reranking     | `cross-encoder/ms-marco-MiniLM-L-6-v2`          |
| LLM           | `TinyLLaMA-1.1B-Chat-v1.0` via `transformers`   |
| Bot Layer     | `Slack Bolt SDK`                                |
| Storage       | `SQLite`                                        |
| Deployment    | `Docker`, `Python 3.10+`                        |

---

 📦 Installation & Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/realtime-rag.git
cd realtime-rag
````

### 2. Create `.env` file

```bash
cp .env.example .env
```

Edit `.env` and set the following:

```env
SLACK_BOT_TOKEN=your-bot-token
SLACK_APP_TOKEN=your-app-level-token
MONITORED_CHANNELS=channel1,channel2
```

### 3. Build Docker image

```bash
docker build -t realtime-rag .
```

---

## ▶️ Running the Slack Bot

```bash
docker run --env-file .env --memory=6g --memory-swap=7g -it realtime-rag
```

Once you see:

```
✅ Slack listener started...
⚡️ Bolt app is running!
```

You're live!

---

 ⚙️ Running the Local RAG Engine (No Slack)

To test the CLI-based version or integrate with other platforms like Microsoft Teams:

```bash
python app.py
```

This will:

* Watch `live_input.txt` for new input (via Watchdog)
* Store data in SQLite
* Serve answers via CLI prompts

---

 🗃️ File Structure

```
.
├── app.py                  # Local/CLI mode RAG engine
├── slack_listener.py       # Slack listener + integration
├── rag_engine.py           # Encapsulated RAG logic (embed, search, rerank)
├── db_handler.py           # SQLite DB handler
├── Dockerfile              # Docker build file
├── requirements.txt
├── .env                    # Your environment variables
└── README.md
```

---

 📈 Future Improvements

* Add streaming generation
* Deploy via LangServe or Kubernetes
* Web dashboard for analytics
* External plugin support for Teams, Discord, etc.

---

📹 Demo Video : https://www.loom.com/share/791244f336bb46a08098081f8d325318?sid=12ac826f-1673-48aa-b4ee-e5e48db9866c



---

 👨‍💻 Developer

**Utkarsh Singh**

* 📧 Email: [utkarshthakur17022002@gmail.com](mailto:utkarshthakur17022002@gmail.com)
* 💼 LinkedIn: [utkarshsingh1702](https://www.linkedin.com/in/utkarshsingh1702)

If you found this project helpful, ⭐ the repo and reach out on LinkedIn!

---

 📄 License

This project is licensed under the [MIT License](LICENSE).

```
