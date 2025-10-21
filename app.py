# =============================================
# 🚀 GSBG AI Chatbot (with Frontend UI + Smart Cache)
# =============================================
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import sqlite3, os, json, logging, threading, time
from sentence_transformers import SentenceTransformer
import ollama

# -------------------- CONFIG --------------------
app = Flask(__name__, static_folder="static")
PORT = 5000
KB_PATH = "kb"
DB_PATH = "database/chatlogs.db"
MODEL_NAME = "mistral:latest"

# -------------------- INIT --------------------
os.makedirs("database", exist_ok=True)
os.makedirs(KB_PATH, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info("🚀 Starting GSBG Chatbot...")

# -------------------- LOAD KNOWLEDGE BASE --------------------
try:
    with open(f"{KB_PATH}/kb_chunks.json", "r", encoding="utf-8") as f:
        kb_chunks = json.load(f)
    kb_embeddings = np.load(f"{KB_PATH}/kb_embeddings.npy")
    kb_embeddings = kb_embeddings / np.linalg.norm(kb_embeddings, axis=1, keepdims=True)
    logging.info(f"✅ Knowledge base loaded with {len(kb_chunks)} chunks.")
except Exception as e:
    logging.warning(f"⚠️ No KB found or failed to load: {e}")
    kb_chunks, kb_embeddings = [], np.zeros((0,))

# -------------------- EMBEDDING MODEL --------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
query_cache = {}

# -------------------- STATIC RESPONSES --------------------
STATIC_RESPONSES = {
    "hello": "👋 Hello! I’m the GSBG AI Assistant. How can I help you today?",
    "hi": "Hi there! I’m GSBG Technologies’ AI assistant — ready to assist you.",
    "who are you": "I’m GSBG Technologies’ AI Assistant. GSBG is a Salesforce CRM and Consulting Partner specializing in CRM solutions and FinTech consulting.",
    "what is gsbg": "GSBG Technologies is a Salesforce CRM and Consulting Partner delivering cloud, CRM, and financial technology solutions.",
    "contact": "📞 You can reach GSBG Technologies at +91-XXXXXXXXXX or email us at contact@gsbgtech.com.",
    "email": "✉️ Our official email is contact@gsbgtech.com.",
    "services": "We specialize in Salesforce CRM implementation, automation, financial technology consulting, and AI-driven business solutions.",
}

# -------------------- PRELOAD MODEL --------------------
def preload_model():
    try:
        logging.info("🧠 Preloading Ollama model into memory...")
        ollama.pull(MODEL_NAME)
        ollama.chat(model=MODEL_NAME, messages=[{"role": "system", "content": "Warmup"}])
        logging.info(f"✅ Model {MODEL_NAME} is ready.")
    except Exception as e:
        logging.warning(f"⚠️ Model preload failed: {e}")

threading.Thread(target=preload_model).start()

# -------------------- DATABASE --------------------
def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT UNIQUE,
            answer TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

ensure_db()

def get_cached_answer(query):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT answer FROM logs WHERE query = ?", (query,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def save_answer(query, answer, confidence):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO logs (query, answer, confidence) VALUES (?, ?, ?)",
        (query, answer, confidence),
    )
    conn.commit()
    conn.close()

# -------------------- SEMANTIC SEARCH --------------------
def semantic_search(query, top_k=5):
    if kb_embeddings.size == 0:
        return [], []
    if query in query_cache:
        query_vec = query_cache[query]
    else:
        query_vec = embedding_model.encode([query])[0]
        query_vec = query_vec / np.linalg.norm(query_vec)
        query_cache[query] = query_vec
    sims = np.dot(kb_embeddings, query_vec)
    top_indices = sims.argsort()[::-1][:top_k]
    return [kb_chunks[i] for i in top_indices], sims[top_indices]

# -------------------- RESPONSE GENERATION --------------------
def generate_response(context, query):
    try:
        system_prompt = (
            "You are GSBG Technologies' official AI assistant. "
            "GSBG Technologies is a Salesforce CRM and Consulting Partner providing CRM solutions, FinTech consulting, and AI-driven business automation. "
            "Always answer confidently and concisely. If unsure, politely recommend contacting GSBG’s support."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Relevant Info:\n{context}\n\nUser Question: {query}"}
        ]

        start = time.time()
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        duration = time.time() - start
        logging.info(f"🧠 Model responded in {duration:.2f}s")

        answer = response.get("message", {}).get("content", "").strip()
        if not answer:
            answer = "I'm sorry, I couldn’t find the information you’re looking for."
        return answer
    except Exception as e:
        logging.error(f"❌ LLM Error: {e}")
        return "I'm sorry, something went wrong. Please try again later."

# -------------------- CHAT ENDPOINT --------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    query = data.get("query", "").strip().lower()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    logging.info(f"💬 User query: {query}")

    # Static responses
    for key, reply in STATIC_RESPONSES.items():
        if key in query:
            return jsonify({"answer": reply, "confidence": 1.0, "source": "static"})

    # Cached response
    cached = get_cached_answer(query)
    if cached:
        logging.info("📦 Serving cached response.")
        return jsonify({"answer": cached, "confidence": 0.95, "source": "cache"})

    # Knowledge Base
    chunks, sims = semantic_search(query)
    if not chunks:
        return jsonify({"answer": "I couldn’t find relevant information in the knowledge base."})

    context = "\n\n".join(c.get("text", "") for c in chunks)
    confidence = float(np.max(sims)) if len(sims) > 0 else 0.0

    answer = generate_response(context, query)
    save_answer(query, answer, confidence)

    return jsonify({"answer": answer, "confidence": round(confidence, 3), "source": "model"})

# -------------------- CHAT UI --------------------
@app.route("/chatbot")
def serve_chat_ui():
    return send_from_directory("templates", "chat.html")

# -------------------- HEALTH --------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "message": "✅ GSBG AI Chatbot is running",
        "model": MODEL_NAME,
        "kb_chunks": len(kb_chunks),
        "port": PORT
    })

# -------------------- START SERVER --------------------
if __name__ == "__main__":
    logging.info(f"🌐 Visit Chatbot at: http://127.0.0.1:{PORT}/chatbot")
    app.run(host="0.0.0.0", port=PORT)
