import os
import io
import faiss
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader
import google.generativeai as genai
import json
import re

# --------------------------
# SINGLE SOURCE OF TRUTH: HARDCODED KEY
# --------------------------
HARDCODED_GEMINI_KEY = "AIzaSyBKLx2mlurxH-s7RZevCoEJiaCfMXcqbQY"

# Configure Gemini once
genai.configure(api_key=HARDCODED_GEMINI_KEY)


def _chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - chunk_overlap, 0)
    return chunks


class SessionIndex(BaseModel):
    index: faiss.IndexFlatIP
    embeddings: np.ndarray
    chunks: List[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class RAGEngine:
    def __init__(self, model_name: str = None) -> None:

        # Embedding model
        embedding_model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._embedder = SentenceTransformer(embedding_model)

        # Store indexes per session
        self._session_to_index: Dict[str, SessionIndex] = {}

        # FORCE hardcoded Gemini key
        self._api_key = HARDCODED_GEMINI_KEY
        self._gemini_model_name = "gemini-2.5-flash-lite"

    # ------------ PDF Extraction -------------
    async def extract_and_index(self, session_id, upload_file) -> Tuple[str, Dict]:
        content_text = await self._extract_pdf_text(upload_file)
        chunks = _chunk_text(content_text)
        embeddings = self._encode(chunks)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        self._session_to_index[session_id] = SessionIndex(
            index=index,
            embeddings=embeddings,
            chunks=chunks
        )

        return content_text, {"chunks_indexed": len(chunks)}

    async def _extract_pdf_text(self, upload_file) -> str:
        data = await upload_file.read()
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    # ------------ Embeddings & Search -------------
    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = self._embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return emb.astype("float32")

    def _num_chunks(self, session_id: str) -> int:
        sess = self._session_to_index.get(session_id)
        return len(sess.chunks) if sess else 0

    def _search(self, session_id: str, query: str, k: int) -> List[int]:
        sess = self._session_to_index.get(session_id)
        if not sess:
            return []
        q = self._encode([query])
        D, I = sess.index.search(q, k)
        return I[0].tolist()

    def _top_k_context(self, session_id: str, query: str, k: int) -> str:
        sess = self._session_to_index.get(session_id)
        if not sess:
            return ""
        idxs = self._search(session_id, query, min(k, len(sess.chunks)))
        return "\n\n".join([sess.chunks[i] for i in idxs if i < len(sess.chunks)])

    # ------------ Gemini Helpers -------------
    async def _gemini_questions(self, prompt: str) -> List[str]:
        """Always uses the hardcoded Gemini key. No env checks."""
        model = genai.GenerativeModel(self._gemini_model_name)
        resp = await model.generate_content_async(prompt)

        text = (resp.text or "[]").strip()
        text = re.sub(r"```(?:json)?\n?|```", "", text)

        # Attempt parse
        try:
            return json.loads(text)
        except:
            pass

        # Try extracting JSON array manually
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                pass

        # If everything fails, extract question-like strings
        lines = [l.strip() for l in text.splitlines() if "?" in l]
        return lines[:20]

    async def _gemini_json(self, prompt: str) -> Dict:
        """Same: always uses hardcoded Gemini key."""
        model = genai.GenerativeModel(self._gemini_model_name)
        resp = await model.generate_content_async(prompt)
        raw = (resp.text or "{}").strip()

        raw = re.sub(r"```(?:json)?\n?|```", "", raw)

        # Direct parse
        try:
            return json.loads(raw)
        except:
            pass

        # Try balanced extraction
        start = raw.find("{")
        if start != -1:
            stack = 0
            for i in range(start, len(raw)):
                if raw[i] == "{":
                    stack += 1
                elif raw[i] == "}":
                    stack -= 1
                if stack == 0:
                    try:
                        obj = json.loads(raw[start:i+1])
                        return obj
                    except:
                        break
        return {}

    # ------------ Question generation -------------
    async def generate_technical_questions(self, session_id, role, count_role, count_resume):
        # Domain questions
        domain_prompt = (
            f"You are an expert interviewer. Generate {count_role} role-based technical questions for {role}. "
            "Return ONLY a JSON array of strings."
        )
        domain_questions = await self._gemini_questions(domain_prompt)

        # Resume questions
        resume_ctx = self._top_k_context(session_id, "resume project", 10)
        resume_prompt = (
            f"Generate {count_resume} resume-based technical questions.\n\n"
            f"Resume:\n{resume_ctx}\n\n"
            "Return ONLY JSON array."
        )
        resume_questions = await self._gemini_questions(resume_prompt)

        return domain_questions[:count_role] + resume_questions[:count_resume]

    async def generate_hr_questions(self, session_id, count=10):
        ctx = self._top_k_context(session_id, "behavioral personality", 5)
        prompt = (
            f"Generate {count} HR behavioral questions.\n"
            f"Context:\n{ctx}\n"
            "Return ONLY JSON array."
        )
        return await self._gemini_questions(prompt)

    # ------------ Evaluation -------------
    async def evaluate_answers(self, session_id, role, technical_answers, hr_answers):
        resume_ctx = self._top_k_context(session_id, "projects achievements", 10)

        prompt = (
            "Evaluate the following interview answers.\n\n"
            f"Role: {role}\n\n"
            f"Resume context:\n{resume_ctx}\n\n"
            f"Technical:\n{json.dumps(technical_answers)}\n\n"
            f"HR:\n{json.dumps(hr_answers)}\n\n"
            "Return a JSON object with technical, hr, overall."
        )

        return await self._gemini_json(prompt)
