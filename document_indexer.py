from pathlib import Path
import os
import pickle
import json

import numpy as np
from tqdm import tqdm
import faiss
from pypdf import PdfReader
from openai import OpenAI


class DocumentIndexer:
    def __init__(self, base_dir="artefacts", embedder_name="text-embedding-3-small", 
                 chunk_size=8000, overlap=200, api_key=None):

        self.embedder_name = embedder_name
        self.client = OpenAI(api_key=api_key)
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.index = None
        self.chunks = []
        self.chunk_embeddings = None

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _read_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def _split_into_chunks(self, text, source_filename):
        chunks = []
        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": source_filename
                })
                chunk_id += 1
            start += self.chunk_size - self.overlap
        return chunks

    def load_documents(self, path, save_pkl=True, pkl_path=None):
        pdf_files = []
        if path.endswith(".pdf"):
            pdf_files = [path]
        else:
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith(".pdf"):
                        pdf_files.append(os.path.join(root, f))

        all_chunks = []
        for pdf in pdf_files:
            print(f"Reading: {pdf}")
            try:
                text = self._read_pdf(pdf)
                source = os.path.basename(pdf)
                chunks = self._split_into_chunks(text, source)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"⚠️ Failed to process {pdf}: {e}")

        self.chunks = all_chunks
        print(f"✅ Processed {len(pdf_files)} PDFs → {len(all_chunks)} total chunks")

        if pkl_path is None:
            pkl_path = self.base_dir / "chunks" / "all_chunks.pkl"

        if save_pkl:
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pkl_path, "wb") as f:
                pickle.dump(self.chunks, f)
            print(f"💾 Saved all chunks → {pkl_path}")

            json_path = pkl_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            print(f"💾 Saved JSON chunks → {json_path}")

    def _embed_texts(self, chunks, batch_size=20, max_token_limit=8000):
        texts = []
        for c in chunks:
            t = c["text"]
            if len(t) > max_token_limit * 4:
                print(f"⚠️ Truncating oversized chunk from {c['source']}")
                t = t[:max_token_limit * 4]
            texts.append(t)

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.embedder_name,
                input=batch
            )
            embeddings.extend([d.embedding for d in response.data])

        return np.array(embeddings, dtype="float32")

    def build_faiss_index(self):
        print(f"🔹 Creating embeddings using {self.embedder_name}...")
        embeddings = self._embed_texts(self.chunks)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        self.chunk_embeddings = embeddings
        print(f"✅ FAISS index built with {index.ntotal} vectors")

    def save_index(self, index_path=None, embeddings_path=None):
        if index_path is None:
            index_path = self.base_dir / "FAISS_index" / "vector_store.faiss"
        if embeddings_path is None:
            embeddings_path = self.base_dir / "chunk_embeddings" / "chunk_embeddings.npy"
        if self.index is None:
            raise ValueError("No FAISS index to save. Build the index first.")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        np.save(embeddings_path, self.chunk_embeddings)
        print(f"💾 Saved FAISS index → {index_path}")
        print(f"💾 Saved chunk embeddings → {embeddings_path}")

    def load_index(self, index_path=None, embeddings_path=None, chunks_pkl=None):
        if index_path is None:
            index_path = self.base_dir / "FAISS_index" / "vector_store.faiss"
        if embeddings_path is None:
            embeddings_path = self.base_dir / "chunk_embeddings" / "chunk_embeddings.npy"
        if chunks_pkl is None:
            chunks_pkl = self.base_dir / "chunks" / "all_chunks.pkl"
        with open(chunks_pkl, "rb") as f:
            self.chunks = pickle.load(f)
        self.index = faiss.read_index(str(index_path))
        self.chunk_embeddings = np.load(embeddings_path)
        print(f"📂 Loaded FAISS index with {self.index.ntotal} vectors.")

    def build_vector_store_from_raw(self, path):
        self.load_documents(path)
        self.build_faiss_index()
        self.save_index()

    def load_vector_store_from_disk(self):
        self.load_index()