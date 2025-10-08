import os
import re
import json
import pickle
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
import requests
from twilio.rest import Client


import os
from dotenv import load_dotenv

load_dotenv()
# Optional FAISS
try:
    import faiss
except Exception:
    faiss = None

# ---------------- CONFIG ----------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

DB_FILE      = "conversations.db"
INDEX_FILE   = "website_index.faiss"
CHUNKS_FILE  = "website_chunks.pkl"
EMB_FILE     = "website_embeddings.npy"

TOP_K = int(os.getenv("TOP_K", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.60))

# ---------------- FLASK APP ----------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GSBG-Chatbot")

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            response TEXT,
            sources TEXT,
            best_similarity REAL,
            created_at TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS embedding_cache (
            query TEXT PRIMARY KEY,
            vector BLOB,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_cached_embedding(query):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT vector FROM embedding_cache WHERE query = ?", (query,))
    row = cur.fetchone()
    conn.close()
    if row and row[0]:
        return pickle.loads(row[0])
    return None

def set_cached_embedding(query, vector):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "REPLACE INTO embedding_cache (query, vector, created_at) VALUES (?, ?, ?)",
        (query, pickle.dumps(vector), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def log_conversation(query, response, sources=None, best_similarity=None):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversations (query, response, sources, best_similarity, created_at) VALUES (?, ?, ?, ?, ?)",
        (query, response, json.dumps(sources) if sources else None, best_similarity, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

# ---------------- EMBEDDINGS ----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query_local(q):
    cached = get_cached_embedding(q)
    if cached is not None:
        return cached
    v = embed_model.encode([q], normalize_embeddings=True)
    v = np.array(v, dtype="float32")
    set_cached_embedding(q, v)
    return v

def find_similar_conversations(query, top_k=3, threshold=None):
    thr = SIMILARITY_THRESHOLD if threshold is None else float(threshold)
    try:
        q_emb = embed_query_local(query)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)

        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT query, response FROM conversations")
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return []

        queries = [row[0] for row in rows]
        responses = [row[1] for row in rows]

        past_embs = np.vstack([embed_query_local(q).reshape(1, -1) for q in queries]).astype("float32")
        sims = (past_embs @ q_emb.T).flatten()
        top_idxs = np.argsort(sims)[-top_k:][::-1]

        return [(queries[i], responses[i], float(sims[i])) for i in top_idxs if float(sims[i]) > thr]

    except Exception:
        logger.exception("Failed to find similar conversations")
        return []

# ---------------- KNOWLEDGE BASE ----------------
index, chunks_meta, embeddings = None, [], None

def load_kb():
    global index, chunks_meta, embeddings
    try:
        if faiss and os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE) and os.path.exists(EMB_FILE):
            index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, "rb") as f:
                chunks_meta = pickle.load(f)
            embeddings = np.load(EMB_FILE)
            logger.info("Loaded FAISS index with %d chunks", len(chunks_meta))
        else:
            chunks_meta = [{"text": "GSBG Technologies is a Salesforce consulting company.", "source": "gsbg_intro"}]
    except Exception:
        logger.exception("Failed to load KB")

load_kb()

def aggregate_context(candidates, max_chars=1500, max_chunks=5):
    seen, parts, sources, count = set(), [], [], 0
    for item in sorted(candidates, key=lambda x: x[1], reverse=True):
        m = item[0]; text = m.get("text", "")
        source = m.get("source", "From KB")
        key = text.strip()[:200]
        if key in seen:
            continue
        seen.add(key)
        parts.append(text.strip())
        sources.append(source)
        count += 1
        if sum(len(p) for p in parts) >= max_chars or count >= max_chunks:
            break
    return ("\n\n".join(parts), sources)

def find_chunks_by_keywords(keywords, max_add=3):
    found = []
    kws = [k.lower() for k in keywords]
    for idx, meta in enumerate(chunks_meta):
        text = meta.get("text", "")
        src = meta.get("source", "")
        lower = (text + " " + src).lower()
        if any(k in lower for k in kws):
            found.append((meta, 0.95, idx))
    unique, out = [], []
    for m, s, i in found:
        src = m.get("source")
        if src and src in unique:
            continue
        if src:
            unique.append(src)
        out.append((m, s, i))
        if len(out) >= max_add:
            break
    return out

def prefer_gsbg(candidates):
    return sorted(candidates, key=lambda item: item[1] + (0.5 if "gsbg" in item[0].get("text", "").lower() else 0), reverse=True)

# ---------------- LLM ----------------
def strip_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip() if text else text

def generate_with_ollama(prompt, model=None):
    model = model or OLLAMA_MODEL
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        url = f"{OLLAMA_URL}/api/generate"
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        text = data.get("response") or data.get("text") or str(data)
        return strip_think_blocks(text)
    except Exception:
        logger.exception("Ollama request failed")
        return None

# ---------------- TWILIO ----------------
def send_sms_to_user(user_number, user_query):
    if not twilio_client or not TWILIO_FROM_NUMBER:
        logger.error("Twilio not configured")
        return None
    body = f"Hello! GSBG Support received your query:\n\"{user_query}\"\nOur team will contact you shortly."
    message = twilio_client.messages.create(
        body=body,
        from_=TWILIO_FROM_NUMBER,
        to=user_number
    )
    return message.sid

# ---------------- CONVERSATIONAL ----------------
def handle_conversational(query):
    GREETINGS = [re.compile(r"\bhello\b", re.I), re.compile(r"\bhi\b", re.I), re.compile(r"\bhey\b", re.I)]
    SMALLTALK = [re.compile(r"\bhow are you\b", re.I), re.compile(r"\bwhat's up\b", re.I)]
    if any(p.search(query) for p in GREETINGS):
        return "Hello! Welcome to GSBG Technologies. How can I help you today?", ["system"]
    if any(p.search(query) for p in SMALLTALK):
        return "I'm great, thank you for asking. How can I assist you with GSBG or Salesforce?", ["system"]
    return None, None

# ---------------- FLASK ENDPOINT ----------------
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    user_number = data.get("phone")  # User phone for SMS
    escalation = False

    if not query:
        return jsonify({"response": "Please ask a question.", "sources": ["system"], "escalation": False}), 400

    # 1. Greetings / Smalltalk
    convo_ans, convo_src = handle_conversational(query)
    if convo_ans:
        return jsonify({"response": convo_ans, "sources": convo_src, "escalation": False})

    # 2. Similar conversation cache
    similar = find_similar_conversations(query, top_k=1)
    if similar:
        past_query, past_response, sim = similar[0]
        if sim > 0.75:
            return jsonify({"response": past_response, "sources": ["cache"], "escalation": False})

    # 3. Knowledge Base retrieval
    candidates = []
    if index and faiss:
        try:
            q_emb = embed_query_local(query)
            D, I = index.search(q_emb, TOP_K)
            for idx, sim in zip(I[0], D[0]):
                if 0 <= idx < len(chunks_meta):
                    candidates.append((chunks_meta[idx], float(sim)))
        except Exception:
            logger.exception("FAISS search failed")
    if not candidates:
        candidates = [(m, 0.95) for m, _, _ in find_chunks_by_keywords([query], max_add=TOP_K)]

    candidates = prefer_gsbg(candidates)
    context, sources = aggregate_context(candidates, max_chars=1500, max_chunks=5)

    # 4. LLM prompt
    prompt = f"""
You are an intelligent assistant for GSBG Technologies.
Use ONLY the context below to answer. Summarize, do not copy. Cite sources clearly.
If unsure, say: "I’m sorry, I don’t have that information — please contact our support."

Context:
{context}

Question: {query}
Answer (short paragraph or bullets, source at end):
"""
    answer = generate_with_ollama(prompt) or context[:500]
    answer = strip_think_blocks(answer)

    # 5. Escalation
    if ("urgent" in query.lower() or "don't understand" in query.lower() or 
        "help" in query.lower() or answer.startswith("I’m sorry")):
        escalation = True
        if not user_number:
            log_conversation(query, answer, sources=sources)
            return jsonify({
                "response": "I’m sorry I couldn't assist. Please enter your phone number so our GSBG support team can contact you via SMS.",
                "sources": sources,
                "escalation": True
            })
        else:
            sms_sid = send_sms_to_user(user_number, query)
            log_conversation(query, f"Escalated: {answer}", sources=sources)
            return jsonify({
                "response": f"Our support team has been notified and you will receive an SMS shortly.",
                "sources": sources,
                "escalation": True
            })

    log_conversation(query, answer, sources=sources)
    return jsonify({"response": answer, "sources": sources, "escalation": False})

@app.route("/")
def home():
    return render_template("chat.html")

if __name__ == "__main__":
    init_db()
    logger.info("Starting GSBG Chatbot")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
