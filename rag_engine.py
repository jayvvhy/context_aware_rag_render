import numpy as np
from openai import OpenAI


class ContextAwareRAG:
    def __init__(self, indexer, generator_name="gpt-4o-mini",
                 max_history=3, max_tokens=100_000, current_weight=2.0,
                 api_key=None):

        self.client = OpenAI(api_key=api_key)
        self.index = indexer.index
        self.chunks = indexer.chunks
        self.embedder_name = indexer.embedder_name
        self.history = []
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.current_weight = current_weight
        self.generator_name = generator_name

    def _build_weighted_query_embedding(self, query):
        response = self.client.embeddings.create(
            input=query,
            model=self.embedder_name
        )
        current_emb = np.array(response.data[0].embedding)

        if not self.history:
            return current_emb

        prev_queries = [q for q, _ in self.history[-self.max_history:]]
        prev_embs = []
        for pq in prev_queries:
            resp = self.client.embeddings.create(
                input=pq, model=self.embedder_name
            )
            prev_embs.append(resp.data[0].embedding)

        prev_embs = np.mean(prev_embs, axis=0)
        combined = (self.current_weight * current_emb + prev_embs) / (self.current_weight + 1)
        combined /= np.linalg.norm(combined)
        return combined

    def _build_prompt(self, query, retrieved_chunks):
        history_text = ""
        for i, (q, a) in enumerate(self.history[-self.max_history:]):
            history_text += f"Previous Q{i+1}: {q}\nPrevious A{i+1}: {a}\n\n"

        context = "\n".join([c["text"] for c in retrieved_chunks])
        prompt = f"{history_text}Context:\n{context}\n\n{query}"

        if len(prompt) > self.max_tokens * 4:
            prompt = prompt[-self.max_tokens * 4:]
        return prompt

    def query(self, query, k=3, max_new_tokens=2000, return_context=False):
        q_emb = self._build_weighted_query_embedding(query).reshape(1, -1)
        D, I = self.index.search(np.array(q_emb).astype("float32"), k)
        retrieved_chunks = [self.chunks[i] for i in I[0]]

        prompt = self._build_prompt(query, retrieved_chunks)

        response = self.client.chat.completions.create(
            model=self.generator_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.3
        )
        output = response.choices[0].message.content

        self.history.append((query, output))

        if return_context:
            return {
                "query": query,
                "answer": output,
                "retrieved_chunks": retrieved_chunks,
                "prompt": prompt,
                "faiss_scores": D[0].tolist(),
            }
        else:
            return output