import os
import io
import faiss
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()


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

	# Allow FAISS and numpy types
	model_config = ConfigDict(arbitrary_types_allowed=True)


class RAGEngine:
	def __init__(self, model_name: str = None) -> None:
		# Use environment variable for embedding model
		embedding_model = model_name or os.getenv("RAG_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
		self._embedder = SentenceTransformer(embedding_model)
		self._session_to_index: Dict[str, SessionIndex] = {}
		api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBKLx2mlurxH-s7RZevCoEJiaCfMXcqbQY")
		if api_key:
			genai.configure(api_key=api_key)
		self._has_gemini_key = bool(api_key)
		self._gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

	async def extract_and_index(self, session_id, upload_file) -> Tuple[str, Dict]:
		content_text = await self._extract_pdf_text(upload_file)
		chunks = _chunk_text(content_text)
		embeddings = self._encode(chunks)
		index = faiss.IndexFlatIP(embeddings.shape[1])
		faiss.normalize_L2(embeddings)
		index.add(embeddings)
		self._session_to_index[session_id] = SessionIndex(index=index, embeddings=embeddings, chunks=chunks)
		return content_text, {"chunks_indexed": len(chunks)}

	async def generate_technical_questions(self, session_id: str, role: str, count_role: int, count_resume: int) -> List[str]:
		if not self._has_gemini_key:
			raise RuntimeError("GEMINI_API_KEY not configured")
		# Generate domain-specific questions first
		domain_prompt = (
    f"You are an expert technical interviewer for FRESHERS applying for the role: {role}.\n"
    f"Generate exactly {count_role} technical interview questions that are EASY to MEDIUM difficulty, NOT too basic, and strictly contain NO coding questions.\n\n"

    "### Question Requirements:\n"
    "- NO coding questions (no writing code, no output prediction, no algorithms implementation).\n"
    "- Avoid very basic questions such as 'What is a variable?' or 'Define HTML'.\n"
    "- Focus on **role-relevant core concepts**, fundamentals, architecture understanding, and real-world applications.\n"
    "- Questions should be conceptual, scenario-based, or explanation-based.\n"
    "- Difficulty should be EASY → MEDIUM only (no hard or advanced topics).\n"
    "- Cover the most IMPORTANT areas expected from freshers for this role.\n\n"

    "### Types of questions expected:\n"
    "- Conceptual understanding (e.g., OOP, HTTP, threading basics, APIs, database concepts)\n"
    "- Real-world scenarios (e.g., how components interact, how systems behave)\n"
    "- Practical knowledge without coding (e.g., how REST works, what is state management, indexing benefits)\n"
    "- Explanation-based questions with examples\n\n"

    "### Example of acceptable questions:\n"
    "- Explain the difference between a process and a thread.\n"
    "- What happens internally when you hit an API endpoint?\n"
    "- How does a REST API differ from a SOAP API?\n"
    "- What is referential integrity in databases?\n"
    "- How does the browser rendering pipeline work?\n"
    "- What is event-driven architecture?\n"
    "- Explain ACID properties with an example.\n\n"

    "### Output Format:\n"
    "Return ONLY a JSON array of strings. No numbering, no bullets, no explanations.\n"
    "Example: [\"Question 1\", \"Question 2\", \"Question 3\"]"
)

		domain_questions = await self._gemini_questions(domain_prompt)
		
		# Generate resume-based questions
		resume_context = self._top_k_context(session_id=session_id, query=f"Key achievements and projects for {role}", k=min(8, self._num_chunks(session_id)))
		resume_prompt = (
			f"You are an expert interviewer. Generate exactly {count_resume} technical questions based on the candidate's resume.\n"
			f"Use the resume context to create personalized questions about their specific experience, projects, and achievements.\n"
			"Make questions specific to their background and projects mentioned.\n"
			"Return ONLY a JSON array of strings without numbering or bullet points.\n\n"
			f"Resume context:\n{resume_context}"
		)
		resume_questions = await self._gemini_questions(resume_prompt)
		
		# Combine both sets of questions
		all_questions = domain_questions[:count_role] + resume_questions[:count_resume]
		return all_questions

	async def generate_hr_questions(self, session_id: str, count: int = 10) -> List[str]:
		if not self._has_gemini_key:
			raise RuntimeError("GEMINI_API_KEY not configured")
		context = self._top_k_context(session_id=session_id, query="behavioral strengths and culture fit", k=min(6, self._num_chunks(session_id)))
		prompt = (
    f"You are an empathetic and engaging HR interviewer. "
    f"Generate exactly {count} thoughtful behavioral interview questions that explore the candidate’s values, personality, and workplace behavior.\n\n"
    "### Style & Tone Guidelines:\n"
    "- Warm, conversational, and approachable, as if in a real HR interview.\n"
    "- Keep questions natural and professional, avoid corporate jargon or robotic phrasing.\n"
    "- Use open-ended phrasing like 'Can you share...', 'I’d love to hear about...', "
    "'What was on your mind when...', 'How did you handle...', 'Looking back, how would you...'.\n"
    "- Vary question structures to avoid repetition.\n"
    "- Cover general areas such as teamwork, leadership, adaptability, motivation, conflict resolution, feedback, values, and communication.\n"
    "- Each question must be 1 sentence, 15–25 words, no yes/no framing.\n"
    "- No numbering, no bullets, no explanations.\n\n"
    "### Output:\n"
    "Return ONLY a JSON array of strings (questions)."
)

		return await self._gemini_questions(prompt)

	def _num_chunks(self, session_id: str) -> int:
		return len(self._session_to_index.get(session_id, SessionIndex(index=faiss.IndexFlatIP(1), embeddings=np.zeros((0, 1), dtype=np.float32), chunks=[])).chunks)

	def _encode(self, texts: List[str]) -> np.ndarray:
		emb = self._embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
		return emb.astype("float32")

	def _search(self, session_id: str, query: str, k: int = 5) -> List[int]:
		sess = self._session_to_index.get(session_id)
		if not sess or sess.embeddings.shape[0] == 0:
			return []
		q = self._encode([query])
		D, I = sess.index.search(q, min(k, sess.embeddings.shape[0]))
		return I[0].tolist()

	def _top_k_context(self, session_id: str, query: str, k: int = 5) -> str:
		idxs = self._search(session_id, query, k)
		sess = self._session_to_index.get(session_id)
		if not sess:
			return ""
		return "\n\n".join([sess.chunks[i] for i in idxs if i < len(sess.chunks)])

	async def _extract_pdf_text(self, upload_file) -> str:
		data = await upload_file.read()
		reader = PdfReader(io.BytesIO(data))
		pages = []
		for page in reader.pages:
			pages.append(page.extract_text() or "")
		return "\n".join(pages)

	async def _gemini_questions(self, prompt: str) -> List[str]:
		api_key = os.getenv("GEMINI_API_KEY", "")
		if not api_key:
			# Enforce Gemini-only behavior
			raise RuntimeError("GEMINI_API_KEY not configured")
		
		try:
			model = genai.GenerativeModel(self._gemini_model_name)
			print(f"🤖 Using Gemini model: {self._gemini_model_name}")
			resp = await model.generate_content_async(prompt)
			text = resp.text or "[]"
			print(f"📝 Gemini response length: {len(text)} characters")
			
			# Remove common markdown code fences
			import re
			text = re.sub(r"```(?:json)?\n?|```\n?", "", text)
			
			# Try to parse raw text first
			try:
				arr = json.loads(text)
				questions = [str(q).strip() for q in arr if str(q).strip()]
				print(f"✅ Parsed JSON array directly from Gemini")
				return questions[:20]
			except Exception:
				# Try to extract a balanced JSON array from the model output
				def _extract_balanced_array(s: str) -> str | None:
					start = s.find('[')
					if start == -1:
						return None
					stack = 0
					for i in range(start, len(s)):
						if s[i] == '[':
							stack += 1
						elif s[i] == ']':
							stack -= 1
						if stack == 0:
							return s[start:i+1]
					return None
				
				json_str = _extract_balanced_array(text)
				if json_str is None:
					# Fallback: loose regex match
					m = re.search(r"\[.*?\]", text, flags=re.S)
					json_str = m.group(0) if m else text
				
				# Repair common trailing commas like [1,2,]
				json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
				
				try:
					arr = json.loads(json_str)
					questions = [str(q).strip() for q in arr if str(q).strip()]
					print(f"✅ Generated {len(questions)} questions from Gemini (recovered)")
					return questions[:20]
				except Exception:
					# Fallback: extract questions heuristically from text
					lines = [l.strip("- •* 1234567890.") for l in text.splitlines() if l.strip()]
					questions = [l for l in lines if len(l) > 10 and "?" in l]
					print(f"📋 Extracted {len(questions)} questions from text")
					return questions[:20]
		except Exception as e:
			print(f"❌ Gemini API error: {e}")
			# Propagate non-parse errors to the API layer
			raise

	async def evaluate_answers(self, session_id: str, role: str, technical_answers: List[Dict[str, str]], hr_answers: List[Dict[str, str]]) -> Dict:
			if not self._has_gemini_key:
				raise RuntimeError("GEMINI_API_KEY not configured")

			resume_context = self._top_k_context(session_id=session_id, query=f"Key projects, achievements, and responsibilities for {role}", k=min(10, self._num_chunks(session_id)))
			prompt = (
				"You are an expert interview evaluator. Score answers fairly and consistently.\n"
				"Requirements:\n"
				"- Score each technical and HR answer from 0 to 100.\n"
				"- Provide 1-2 sentence feedback per answer.\n"
				"- Consider the role context and the candidate's resume context.\n\n"
				f"Role: {role}\n\n"
				f"Resume context (ground your evaluation on this):\n{resume_context}\n\n"
				"Technical QA (array of objects with question, answer):\n"
				f"{json.dumps(technical_answers, ensure_ascii=False)}\n\n"
				"HR QA (array of objects with question, answer):\n"
				f"{json.dumps(hr_answers, ensure_ascii=False)}\n\n"
				"Return ONLY a single JSON object with the following shape:\n"
				"{\n"
				"  \"technical\": { \"answers\": [{ \"question\": str, \"answer\": str, \"score\": int, \"feedback\": str }], \"score\": int },\n"
				"  \"hr\": { \"answers\": [{ \"question\": str, \"answer\": str, \"score\": int, \"feedback\": str }], \"score\": int },\n"
				"  \"overall\": int\n"
				"}"
			)

			# Call Gemini and parse JSON
			parsed = await self._gemini_json(prompt)

			tech_parsed = parsed.get("technical", {}).get("answers", []) if isinstance(parsed.get("technical", {}), dict) else []
			hr_parsed = parsed.get("hr", {}).get("answers", []) if isinstance(parsed.get("hr", {}), dict) else []

			merged_tech = []
			for i, orig in enumerate(technical_answers):
				score = None
				feedback = None
				if i < len(tech_parsed):
					candidate = tech_parsed[i]
					score = candidate.get("score") if isinstance(candidate, dict) else None
					feedback = candidate.get("feedback") if isinstance(candidate, dict) else None
				# normalize score to 0-100
				try:
					score = int(score) if score is not None else 0
				except Exception:
					score = 0
				# If candidate didn't provide an answer (empty transcript), force score to 0
				answer_text = (orig.get("answer") or "").strip()
				if not answer_text:
					score = 0
					feedback = feedback or "No answer provided."
				# Convert percentage score (0-100) to marks out of 5 per question
				marks = int(round((score / 100.0) * 5)) if isinstance(score, (int, float)) else 0
				merged_tech.append({
					"question": orig.get("question"),
					"answer": orig.get("answer"),
					"type": orig.get("type", "role"),
					"score": score,
					"marks": marks,
					"feedback": feedback or ""
				})

			merged_hr = []
			for i, orig in enumerate(hr_answers):
				score = None
				feedback = None
				if i < len(hr_parsed):
					candidate = hr_parsed[i]
					score = candidate.get("score") if isinstance(candidate, dict) else None
					feedback = candidate.get("feedback") if isinstance(candidate, dict) else None
				try:
					score = int(score) if score is not None else 0
				except Exception:
					score = 0
				answer_text = (orig.get("answer") or "").strip()
				if not answer_text:
					score = 0
					feedback = feedback or "No answer provided."
				marks = int(round((score / 100.0) * 5)) if isinstance(score, (int, float)) else 0
				merged_hr.append({
					"question": orig.get("question"),
					"answer": orig.get("answer"),
					"score": score,
					"marks": marks,
					"feedback": feedback or ""
				})

			# Scoring configuration (per user's requested mapping)
			MAX_ROLE_QUESTIONS = 7
			MAX_RESUME_QUESTIONS = 8
			MAX_HR_QUESTIONS = 5
			MARKS_PER_QUESTION = 5

			MAX_ROLE_MARKS = MAX_ROLE_QUESTIONS * MARKS_PER_QUESTION  # 35
			MAX_RESUME_MARKS = MAX_RESUME_QUESTIONS * MARKS_PER_QUESTION  # 40
			MAX_HR_MARKS = MAX_HR_QUESTIONS * MARKS_PER_QUESTION  # 25
			# Totals
			tech_marks = sum([a.get("marks", 0) for a in merged_tech])
			role_marks = sum([a.get("marks", 0) for a in merged_tech if a.get("type") == "role"])
			resume_marks = sum([a.get("marks", 0) for a in merged_tech if a.get("type") == "resume"])
			hr_marks = sum([a.get("marks", 0) for a in merged_hr])
			total_marks = int(role_marks + resume_marks + hr_marks)

			# Percent helpers
			def to_percent(marks, max_marks):
				return int(round((marks / max_marks) * 100)) if max_marks and marks is not None else 0

			role_percent = to_percent(role_marks, MAX_ROLE_MARKS)
			resume_percent = to_percent(resume_marks, MAX_RESUME_MARKS)
			tech_percent = to_percent(role_marks + resume_marks, MAX_ROLE_MARKS + MAX_RESUME_MARKS)
			hr_percent = to_percent(hr_marks, MAX_HR_MARKS)
			overall_percent = to_percent(total_marks, MAX_ROLE_MARKS + MAX_RESUME_MARKS + MAX_HR_MARKS)

			result = {
				"technical": {
					"answers": merged_tech,
					"score": int(role_marks + resume_marks),
					"role_score": int(role_marks),
					"resume_score": int(resume_marks),
					"role_percent": role_percent,
					"resume_percent": resume_percent,
					"technical_percent": tech_percent
				},
				"hr": {
					"answers": merged_hr,
					"score": int(hr_marks),
					"hr_percent": hr_percent
				},
				"overall": int(total_marks),
				"overall_percent": overall_percent
			}

			# Generate consolidated feedback (technical, hr, communication, tips) based on the candidate's answers.
			# This produces 3-4 concise, non-generic points per category derived from the actual answers/feedback.
			summary_prompt = (
				"You are an expert evaluator. Based ONLY on the provided answered Q&A and any per-answer feedback, "
				"generate concise, actionable feedback items. Do NOT produce generic or pre-defined phrases — base them on the answers.\n\n"
				"Input JSON:\n"
				f"{{\n  \"technical_answers\": {json.dumps(merged_tech, ensure_ascii=False)},\n  \"hr_answers\": {json.dumps(merged_hr, ensure_ascii=False)}\n}}\n\n"
				"Output MUST be a single JSON object with these keys: \n"
				"{\n  \"technical_feedback\": [string],  # 3-4 concise points about technical strengths and improvement opportunities\n"
				"  \"hr_feedback\": [string],         # 3-4 concise points about HR/behavioral strengths and improvement opportunities\n"
				"  \"communication_feedback\": [string], # 3-4 concise points about communication style, clarity, and structure\n"
				"  \"tips_to_improve\": [string]     # 3-4 actionable tips (study/resources/practice) tailored to the candidate's answers\n"
				"}\n"
			)

			try:
				summary_parsed = await self._gemini_json(summary_prompt)
			except Exception:
				summary_parsed = {}

			# Attach feedback summary to result for downstream consumption
			result["feedback_summary"] = {
				"technical_feedback": summary_parsed.get("technical_feedback") if isinstance(summary_parsed.get("technical_feedback"), list) else [],
				"hr_feedback": summary_parsed.get("hr_feedback") if isinstance(summary_parsed.get("hr_feedback"), list) else [],
				"communication_feedback": summary_parsed.get("communication_feedback") if isinstance(summary_parsed.get("communication_feedback"), list) else [],
				"tips_to_improve": summary_parsed.get("tips_to_improve") if isinstance(summary_parsed.get("tips_to_improve"), list) else [],
			}

			return result

	async def _gemini_json(self, prompt: str) -> Dict:
		api_key = os.getenv("GEMINI_API_KEY", "")
		if not api_key:
			raise RuntimeError("GEMINI_API_KEY not configured")
		# Call the model and safely parse its JSON output
		try:
			model = genai.GenerativeModel(self._gemini_model_name)
			resp = await model.generate_content_async(prompt)
		except Exception as e:
			# Propagate API/network errors so caller can handle them
			print(f"❌ Gemini API error: {e}")
			raise

		text = (resp.text or "{}").strip()
		import re
		# Remove markdown code fences that often wrap model outputs
		text = re.sub(r"```(?:json)?\n?|```\n?", "", text)

		# Helper: try to parse JSON string, returning None on failure
		def _try_parse(s: str):
			try:
				return json.loads(s)
			except Exception:
				return None

		# 1) Try direct parse
		parsed = _try_parse(text)
		if parsed is not None:
			return parsed

		# 2) Try to extract the first balanced JSON object
		start = text.find('{')
		if start != -1:
			stack = 0
			for i in range(start, len(text)):
				if text[i] == '{':
					stack += 1
				elif text[i] == '}':
					stack -= 1
				if stack == 0:
					json_str = text[start:i+1]
					# Repair simple trailing commas
					json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
					parsed = _try_parse(json_str)
					if parsed is not None:
						return parsed

		# 3) Fallback: loose regex match for JSON object
		m = re.search(r"\{[\s\S]*\}", text)
		if m:
			json_str = m.group(0)
			json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
			parsed = _try_parse(json_str)
			if parsed is not None:
				return parsed

		# 4) Last resort: return empty dict (avoid raising a parse-only error)
		print("❌ Gemini JSON parse error: failed to recover valid JSON from model output")
		print("--- Raw Gemini output (truncated) ---")
		print((text[:2000] + '...') if len(text) > 2000 else text)
		return {}


