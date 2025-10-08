# build_kb_auto.py
import os
import pickle
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import hashlib

INDEX_FILE = "website_index.faiss"
CHUNKS_FILE = "website_chunks.pkl"
EMB_FILE = "website_embeddings.npy"
FOLDER = "knowledge"

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += "\n" + content
    return text.strip()

def chunk_text(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def build_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms
    return embeddings

def hash_folder(folder):
    """Generate a hash based on all PDF contents in folder."""
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(folder):
        for file in sorted(files):
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                with open(path, "rb") as f:
                    hash_md5.update(f.read())
    return hash_md5.hexdigest()

def build_kb():
    all_chunks = []
    all_texts = []

    print(f"📚 Building knowledge base from PDFs in '{FOLDER}' ...")
    for file in sorted(os.listdir(FOLDER)):
        if not file.endswith(".pdf"):
            continue
        path = os.path.join(FOLDER, file)
        print(f"📄 Reading {file} ...")
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        print(f"✅ {file}: {len(chunks)} chunks")
        for c in chunks:
            all_chunks.append({"text": c, "source": file})
            all_texts.append(c)

    print("🧠 Generating embeddings ...")
    embeddings = build_embeddings(all_texts)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    np.save(EMB_FILE, embeddings)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    with open(".kb_hash", "w") as f:
        f.write(hash_folder(FOLDER))

    print("✅ Knowledge base built successfully!\n")

def check_and_update():
    """Check if PDFs changed; rebuild if needed."""
    new_hash = hash_folder(FOLDER)
    old_hash = ""
    if os.path.exists(".kb_hash"):
        with open(".kb_hash") as f:
            old_hash = f.read().strip()

    if new_hash != old_hash:
        print("🔄 PDFs changed — rebuilding knowledge base...")
        build_kb()
    else:
        print("✅ PDFs unchanged — using existing FAISS index.\n")

if __name__ == "__main__":
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"📂 Created folder: {FOLDER}")
    check_and_update()
