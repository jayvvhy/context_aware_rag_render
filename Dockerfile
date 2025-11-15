FROM python:3.10-slim

WORKDIR /app

# System dependencies for FAISS, numpy, pypdf
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    curl \
    git \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY app.py .
COPY document_indexer.py .
COPY rag_engine.py .
COPY artefacts ./artefacts

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
