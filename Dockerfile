# -----------------------------
# 1. Base Image
# -----------------------------
FROM python:3.10-slim

# Create app directory
WORKDIR /app

# -----------------------------
# 2. Install system dependencies
# -----------------------------
# Required for FAISS, numpy, pypdf, pdf rendering, and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    curl \
    git \
    libatlas-base-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 3. Copy application files
# -----------------------------
COPY requirements.txt .
COPY app.py .
COPY document_indexer.py .
COPY rag_engine.py .
COPY artefacts ./artefacts

# -----------------------------
# 4. Install Python dependencies
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 5. Expose Streamlit port
# -----------------------------
EXPOSE 8501

# -----------------------------
# 6. Healthcheck for Render
# -----------------------------
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# -----------------------------
# 7. Run Streamlit App
# -----------------------------
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]