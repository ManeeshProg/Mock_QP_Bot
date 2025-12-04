export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://3.109.217.174:8000";
console.log("🔵 Loaded API module. BACKEND_URL =", BACKEND_URL);
export async function uploadResumeExtract(sessionId: string, file: File) {
	console.log("Calling backend:", BACKEND_URL);
	const form = new FormData();
	form.append("session_id", sessionId);
	form.append("file", file);
	const resp = await fetch(`${BACKEND_URL}/extract`, {
		method: "POST",
		body: form
	});
	if (!resp.ok) throw new Error(`Extract failed: ${resp.status}`);
	return resp.json();
}

export async function fetchTechnicalQuestions(sessionId: string, role: string, countRole = 7, countResume = 8) {
	const resp = await fetch(`${BACKEND_URL}/questions/technical`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ session_id: sessionId, role, count_role: countRole, count_resume: countResume })
	});
	if (!resp.ok) throw new Error(`Technical questions failed: ${resp.status}`);
	return resp.json();
}

export async function fetchHrQuestions(sessionId: string) {
	const resp = await fetch(`${BACKEND_URL}/questions/hr`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ session_id: sessionId })
	});
	if (!resp.ok) throw new Error(`HR questions failed: ${resp.status}`);
	return resp.json();
}

export async function evaluateInterview(sessionId: string, role: string, technicalAnswers: Array<{question: string; answer: string}>, hrAnswers: Array<{question: string; answer: string}>) {
	const resp = await fetch(`${BACKEND_URL}/evaluate`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ session_id: sessionId, role, technical_answers: technicalAnswers, hr_answers: hrAnswers })
	});
	if (!resp.ok) throw new Error(`Evaluation failed: ${resp.status}`);
	return resp.json();
}


