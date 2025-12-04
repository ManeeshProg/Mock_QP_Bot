export const BACKEND_URL = "https://symposium-intent-lie-projector.trycloudflare.com";
console.log("🔵 Loaded API module. BACKEND_URL =", BACKEND_URL);
export async function uploadResumeExtract(sessionId: string, file: File) {
	console.log("[api] uploadResumeExtract called. BACKEND_URL=", BACKEND_URL, "sessionId=", sessionId, "file=", file?.name);
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
	console.log('[api] fetchTechnicalQuestions', { sessionId, role, countRole, countResume });
	const resp = await fetch(`${BACKEND_URL}/questions/technical`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ session_id: sessionId, role, count_role: countRole, count_resume: countResume })
	});
	if (!resp.ok) throw new Error(`Technical questions failed: ${resp.status}`);
	return resp.json();
}

export async function fetchHrQuestions(sessionId: string) {
	console.log('[api] fetchHrQuestions', { sessionId });
	const resp = await fetch(`${BACKEND_URL}/questions/hr`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ session_id: sessionId })
	});
	if (!resp.ok) throw new Error(`HR questions failed: ${resp.status}`);
	return resp.json();
}

export async function evaluateInterview(sessionId: string, role: string, technicalAnswers: Array<{question: string; answer: string}>, hrAnswers: Array<{question: string; answer: string}>) {
	console.log('[api] evaluateInterview', { sessionId, role, technicalAnswersCount: technicalAnswers?.length, hrAnswersCount: hrAnswers?.length });
	const resp = await fetch(`${BACKEND_URL}/evaluate`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ session_id: sessionId, role, technical_answers: technicalAnswers, hr_answers: hrAnswers })
	});
	if (!resp.ok) throw new Error(`Evaluation failed: ${resp.status}`);
	return resp.json();
}


