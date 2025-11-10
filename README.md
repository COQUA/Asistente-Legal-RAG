# ⚖️ Asistente de Consultas Legales — Ley de Alquileres (RAG)

Descripción breve
-----------------
Este proyecto es un asistente local tipo RAG (Retrieval-Augmented Generation) diseñado para responder consultas legales sobre contratos y leyes de alquileres, usando los textos que le proveas en PDFs. Es útil para abogados, estudiantes de derecho o equipos legales que quieran buscar rápidamente referencias en sus documentos (leyes, contratos, cláusulas) y obtener respuestas citando el archivo y la página.

Problema real que resuelve
--------------------------
- Muchas consultas legales requieren buscar texto concreto en leyes y contratos dispersos en múltiples PDFs.
- Este proyecto permite indexar (generar embeddings) de tus PDFs y luego realizar preguntas en lenguaje natural; el sistema recupera los fragmentos relevantes y genera una respuesta basada únicamente en esos fragmentos, citando archivo y página.
- Beneficiarios: profesionales legales, equipos de cumplimiento, y cualquier persona que necesite respuestas rápidas basadas en su propia documentación.

Demo (GIF animado)
-------------------
Aquí hay una demostración visual cortaque muestra la app en acción:

![Demo de la app](assets/demo.gif)

Notas:
- El GIF está incluido en `assets/demo.gif` (si no se ve en la vista previa de GitHub, abrí el archivo directamente en la carpeta `assets/`).

Stack tecnológico
------------------
- Python
- Streamlit — interfaz web (archivo `app.py`).
- google-generativeai — cliente para embeddings y generación (modelos: `text-embedding-004` y `gemini-2.0-flash`).
- pypdf — lectura y extracción de texto de PDFs.
- numpy — manejo de vectores y guardado/carga (`.npy`).
- python-dotenv — carga de variables de entorno desde `.env`.

Archivos principales
-------------------
- `app.py`: interfaz Streamlit. Permite subir PDFs, lanzar la ingestión y realizar consultas.
- `utils.py`: script de ingestión. Lee PDFs (`data/`), los divide en chunks, genera embeddings via API y guarda `storage/embeddings.npy` y `storage/meta.json`.
- `rag.py`: carga índice (embeddings + metadatos), busca los fragments más relevantes para una consulta, construye el contexto y llama al modelo de chat para generar la respuesta.
- `requirements.txt`: dependencias del proyecto.

Instalación
-----------
1. Clonar el repositorio:

```bash
git clone https://github.com/COQUA/Asistente-Legal-RAG.git
cd Asistente-Legal-RAG
```

2. Crear y activar un entorno virtual (recomendado):

```bash
python -m venv .venv
source .venv/bin/activate   # Git Bash / Linux / macOS
# En Windows PowerShell: .\.venv\Scripts\Activate.ps1
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Añadir la clave de API de Google Generative AI en un archivo `.env` en la raíz del repo:

```
GOOGLE_API_KEY=tu_clave_aqui
```

Notas: No subas tu `.env` a repos remotos.

Uso (rápido)
-------------
1. Preparar PDFs: coloca los PDFs que quieras indexar dentro de la carpeta `data/` (creala si no existe), o súbelos desde la interfaz.

2. Ingesta (generar embeddings):

```bash
python utils.py 
```

Esto generará `storage/embeddings.npy` y `storage/meta.json`.

3. Ejecutar la interfaz:

```bash
streamlit run app.py
```

La UI permite limpiar el chat, subir PDFs, procesarlos y hacer preguntas en lenguaje natural.

Uso alternativo desde la línea de comandos (sin UI):

```bash
python rag.py --ask "¿Qué dice la ley sobre ...?"
```

Consideraciones importantes
-------------------------
- Requiere conexión a la API de Google Generative AI y la variable `GOOGLE_API_KEY` configurada.
- Cada embedding y cada generación consume cuota/coste; revisá tu plan de Google Cloud.
- `pypdf` NO hace OCR: si tus PDFs son imágenes escaneadas, necesitás un paso de OCR (p. ej. Tesseract) antes de la ingestión.
- `utils.py` hace llamadas secuenciales a la API para cada chunk — para colecciones grandes considerá batching o límites de tasa.

Licencia y contacto
-------------------
- Este repo es un trabajo Practico para la Materia SAC (2025) de la carrera TSDSM en el IPF.
