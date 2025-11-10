from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=False)

import os, json, argparse
import numpy as np
import google.generativeai as genai

EMBED_MODEL = "text-embedding-004"
CHAT_MODEL  = "gemini-2.0-flash"  # o "gemini-1.5-pro"

def load_index(store_dir="storage"):
    embs = np.load(os.path.join(store_dir, "embeddings.npy"))              # (N, D)
    with open(os.path.join(store_dir, "meta.json"), "r", encoding="utf-8") as f:
        metas = json.load(f)
    return embs, metas

def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    # Devuelve x normalizada por fila, evitando div/0
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9               # (N, 1)
    return x / norms

def embed_query(q: str) -> np.ndarray:
    r = genai.embed_content(model=EMBED_MODEL, content=q)
    return np.array(r["embedding"], dtype=np.float32)                      # (D,)

def search(query: str, k=5, store_dir="storage"):
    embs, metas = load_index(store_dir)
    embs = embs.astype(np.float32, copy=False)                             # (N, D)
    embs_norm = _l2norm_rows(embs)                                         # (N, D)

    q = embed_query(query).astype(np.float32)                              # (D,)
    q = q / (np.linalg.norm(q) + 1e-9)                                     # (D,)

    sims = embs_norm @ q                                                   # (N,)
    top = sims.argsort()[::-1][:k]
    return [(float(sims[i]), metas[i]) for i in top]

def build_context(hits):
    parts, srcs = [], []
    for score, m in hits:
        tag = f"[{m['source']} p.{m['page']}]"
        parts.append(f"{tag} {m['text']}")
        srcs.append(tag)
    return "\n\n".join(parts), ", ".join(sorted(set(srcs)))

def answer(question: str, k=5, store_dir="storage"):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        from dotenv import find_dotenv
        return (
            "Falta GOOGLE_API_KEY en el entorno.\n"
            f"- cwd: {os.getcwd()}\n"
            f"- .env encontrado: {find_dotenv(usecwd=True) or 'NO'}\n"
            "- Formato: GOOGLE_API_KEY=TU_CLAVE (sin espacios)"
        )
    genai.configure(api_key=api_key)

    hits = search(question, k=k, store_dir=store_dir)
    if not hits:
        return "No puedo responder con certeza con los documentos cargados."

    context, srcs = build_context(hits)

    system = (
        "Eres un asistente RAG de consultas legales sobre alquileres en Argentina. "
        "Responde SOLO con el CONTEXTO; si no alcanza, dilo. Incluye SIEMPRE citas (archivo p.X)."
    )
    prompt = (
        f"{system}\n\nPregunta: {question}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"Instrucciones:\n- Responde exclusivamente con el CONTEXTO.\n- Cita (archivo p.X)."
    )

    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(prompt, generation_config={"temperature": 0})
    text = (resp.text or "").strip()
    if not text:
        return "No pude generar una respuesta con los documentos disponibles."
    return f"{text}\n\nFuentes: {srcs}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--store_dir", default="storage")
    args = ap.parse_args()
    print(answer(args.ask, k=args.k, store_dir=args.store_dir))
