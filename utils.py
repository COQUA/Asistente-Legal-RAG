from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=False)

import os, json, argparse
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai

EMBED_MODEL = "text-embedding-004" 

def read_pdfs(data_dir: str):
    docs = []
    for name in sorted(os.listdir(data_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(data_dir, name)
        rdr = PdfReader(path)
        for i, page in enumerate(rdr.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append({"source": path, "page": i, "text": text})
    return docs

def chunk_text(text: str, size=1000, overlap=200):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if start >= len(text): break
    return chunks

def embed_many(texts):
    vecs = []
    for t in texts:
        r = genai.embed_content(model=EMBED_MODEL, content=t)
        vecs.append(r["embedding"])
    return np.array(vecs, dtype=np.float32)

def main(data_dir="data", store_dir="storage"):
    os.makedirs(store_dir, exist_ok=True)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            f"Falta GOOGLE_API_KEY en el entorno.\n"
            f"- cwd: {os.getcwd()}\n"
            f"- .env encontrado: {find_dotenv(usecwd=True) or 'NO'}\n"
            f"- Asegúrate de tener una línea: GOOGLE_API_KEY=TU_CLAVE (sin espacios)"
        )
    genai.configure(api_key=api_key)

    raw_pages = read_pdfs(data_dir)
    all_chunks, metas = [], []
    for item in raw_pages:
        for ch in chunk_text(item["text"], size=1000, overlap=200):
            all_chunks.append(ch)
            metas.append({"source": os.path.basename(item["source"]), "page": item["page"], "text": ch[:1200]})

    if not all_chunks:
        raise SystemExit("No se encontró texto en PDFs dentro de 'data/'.")

    embs = embed_many(all_chunks)
    np.save(os.path.join(store_dir, "embeddings.npy"), embs)
    with open(os.path.join(store_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"Ingesta OK: {len(all_chunks)} chunks → storage/embeddings.npy + storage/meta.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--store_dir", default="storage")
    args = ap.parse_args()
    main(args.data_dir, args.store_dir)
