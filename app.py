import streamlit as st
from rag import answer
import os
from utils import main as ingest_main

st.set_page_config(page_title="Asistente Alquileres (RAG)", page_icon="")
st.title(" Asistente de Consultas Legales — Ley de Alquileres (RAG)")
st.caption("Responde solo con lo que está en los PDFs y cita archivo + página.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Sesión")
    if st.button(" Limpiar chat"):
        st.session_state.history = []
        st.rerun()
    st.markdown("Colocá tus PDFs en `data/` o subilos abajo y presioná 'Subir y procesar' para generar embeddings (usa `utils.py`).")

st.markdown("### Subir PDFs (leyes y contratos)")
uploaded_files = st.file_uploader(
    "Subí uno o más archivos PDF (pueden ser leyes, contratos, etc.)",
    type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} archivo(s) listos para subir.")
    if st.button("Subir y procesar"):
        os.makedirs("data", exist_ok=True)
        saved = []
        for f in uploaded_files:
            # Guardar en data/ con el nombre original (sobrescribe si existe)
            dest = os.path.join("data", f.name)
            try:
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                saved.append(dest)
            except Exception as e:
                st.error(f"No se pudo guardar {f.name}: {e}")

        if saved:
            with st.spinner("Procesando PDFs y generando embeddings... esto puede tardar según tu cuota de API"):
                try:
                    ingest_main(data_dir="data", store_dir="storage")
                    st.success("Ingesta completada: embeddings y metadatos guardados en /storage ")
                except Exception as e:
                    st.error(f"Error durante la ingestión: {e}")

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

query = st.chat_input("Escribe tu pregunta legal…")
if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de conocimiento…"):
            try:
                resp = answer(query)
            except Exception as e:
                resp = (
                    f"Hubo un error: {e}\n\n" 
                    "¿Corriste `python utils.py` o subiste PDFs y configuraste GOOGLE_API_KEY?"
                )
        st.markdown(resp)
    st.session_state.history.append(("assistant", resp))
