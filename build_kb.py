import os
import fitz  # PyMuPDF
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Paths ---
input_files = ["knowledge/KB.pdf", "knowledge/KB2.pdf"]
output_dir = "kb"
os.makedirs(output_dir, exist_ok=True)

# --- Initialize embedding model ---
print("🔍 Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Extract text from PDFs ---
def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

# --- Chunking utility ---
def chunk_text(text, max_len=800):
    """
    Split text into smaller overlapping chunks for better embedding quality.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_len:
            current += para + " "
        else:
            chunks.append(current.strip())
            current = para + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks

# --- Combine all texts ---
print("📘 Extracting text from PDFs...")
all_chunks = []
for pdf in input_files:
    text = extract_text_from_pdf(pdf)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "text": chunk,
            "metadata": {"source": os.path.basename(pdf), "chunk_id": i}
        })

print(f"✅ Total chunks created: {len(all_chunks)}")

# --- Generate embeddings ---
print("🧠 Generating embeddings...")
texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# --- Save outputs ---
np.save(os.path.join(output_dir, "kb_embeddings.npy"), embeddings)
with open(os.path.join(output_dir, "kb_chunks.json"), "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("\n✨ Knowledge base built successfully!")
print(f"📄 Saved: {output_dir}/kb_chunks.json")
print(f"📈 Saved: {output_dir}/kb_embeddings.npy")
