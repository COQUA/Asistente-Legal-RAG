import streamlit as st
from rag import answer

st.set_page_config(page_title="Asistente Alquileres (RAG)", page_icon="⚖️")
st.title("⚖️ Asistente de Consultas Legales — Ley de Alquileres (RAG)")
st.caption("Responde solo con lo que está en los PDFs y cita archivo + página.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Sesión")
    if st.button("🧹 Limpiar chat"):
        st.session_state.history = []
        st.rerun()
    st.markdown("Colocá tus PDFs en `data/` y corré `python ingest.py` antes de usar el chat.")

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
                resp = f"Hubo un error: {e}\n\n¿Corriste `python ingest.py` y configuraste OPENAI_API_KEY?"
        st.markdown(resp)
    st.session_state.history.append(("assistant", resp))
