import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Brain, Mic, MicOff, Volume2, ArrowLeft, ArrowRight } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface TechnicalInterviewProps {
  candidateName: string;
  role: string;
  resumeContent: string;
  sessionId: string;
  remainingSeconds: number;
  onNext: (answers: Array<{ question: string; answer: string }>) => void;
  onBack: () => void;
}

export const TechnicalInterview = ({ candidateName, role, resumeContent, sessionId, remainingSeconds, onNext, onBack }: TechnicalInterviewProps) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<Array<{ question: string; answer: string }>>([]);
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [answeredOnce, setAnsweredOnce] = useState<Set<number>>(new Set());
  const [transcript, setTranscript] = useState("");
  const [interim, setInterim] = useState("");
  const [questions, setQuestions] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();

  // Fetch questions from backend using RAG + Gemini
  useEffect(() => {
    let cancelled = false;
    async function fetchQs() {
      try {
        const resp = await (await import("@/lib/api")).fetchTechnicalQuestions(sessionId, role, 7, 8);
        if (!cancelled) {
          setQuestions(resp.questions || []);
        }
      } catch (e) {
        if (!cancelled) {
          setQuestions([]);
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    fetchQs();
    return () => { cancelled = true; };
  }, [role, resumeContent, sessionId]);

  const COUNT_ROLE = 7; // keep in sync with fetchTechnicalQuestions

  const recognitionCtor = (typeof window !== 'undefined' ? ( (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition ) : null);
  const recognitionInstanceRef = useRef<any>(null);
  const isSecureContextOk = typeof window !== 'undefined' && (window.location.protocol === 'https:' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

  const startSTT = () => {
    if (!recognitionCtor) {
      toast({ title: "Speech recognition unavailable", description: "Use Chrome/Edge on desktop.", variant: "destructive" });
      return;
    }
    if (!isSecureContextOk) {
      toast({ title: "Insecure context", description: "Speech recognition requires HTTPS or localhost.", variant: "destructive" });
      return;
    }
    if (recognitionInstanceRef.current) {
      try { recognitionInstanceRef.current.stop(); } catch {}
      recognitionInstanceRef.current = null;
    }
    const rec = new recognitionCtor();
    recognitionInstanceRef.current = rec;
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-US";
    setTranscript("");
    setInterim("");
    rec.onresult = (event: any) => {
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const res = event.results[i];
        if (res.isFinal) finalText += res[0].transcript;
        else interimText += res[0].transcript;
      }
      if (finalText) setTranscript(prev => (prev ? prev + " " : "") + finalText.trim());
      setInterim(interimText.trim());
    };
    rec.onerror = (event: any) => {
      const err: string = event?.error || "unknown";
      let description = "";
      if (err === "not-allowed") description = "Microphone access denied. Please allow mic permissions.";
      else if (err === "no-speech") description = "No speech detected. Try speaking closer to the mic.";
      else if (err === "audio-capture") description = "No microphone found. Check your input device.";
      else description = "Please retry or refresh the page.";
      toast({ title: `Speech recognition error: ${err}`, description, variant: "destructive" });
      setIsRecording(false);
    };
    rec.onend = () => {
      setIsRecording(false);
    };
    try {
      rec.start();
      setIsRecording(true);
      toast({ title: "Recording started", description: "Speak your answer" });
    } catch (e) {
      toast({ title: "Failed to start STT", description: "Try again or refresh.", variant: "destructive" });
    }
  };

  const stopSTT = () => {
    try { recognitionInstanceRef.current?.stop(); } catch {}
    setIsRecording(false);
    setAnsweredOnce(prev => new Set(prev).add(currentQuestionIndex));
    toast({ title: "Recording stopped" });
  };

  const handleNext = () => {
    const finalAnswer = (transcript + (interim ? (transcript ? " " : "") + interim : "")).trim();
    const newAnswer = {
      question: questions[currentQuestionIndex],
      answer: finalAnswer || currentAnswer,
      type: currentQuestionIndex < COUNT_ROLE ? 'role' : 'resume'
    };

    const updatedAnswers = [...answers, newAnswer];
    setAnswers(updatedAnswers);

    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
      setIsRecording(false);
      setTranscript("");
      setInterim("");
      setAnsweredOnce(prev => new Set(prev).add(currentQuestionIndex));
    } else {
      onNext(updatedAnswers);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prev => prev - 1);
      setCurrentAnswer(answers[currentQuestionIndex - 1]?.answer || "");
    }
  };

  const toggleRecording = () => {
    if (answeredOnce.has(currentQuestionIndex)) return;
    if (isRecording) stopSTT(); else startSTT();
  };

  const speakQuestion = () => {
    if ('speechSynthesis' in window && questions[currentQuestionIndex]) {
      const utterance = new SpeechSynthesisUtterance(questions[currentQuestionIndex]);
      speechSynthesis.speak(utterance);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/10 flex items-center justify-center">
        <Card className="p-8 bg-card/30 backdrop-blur-xl border border-primary/20 shadow-2xl">
          <div className="text-center">
            <Brain className="w-16 h-16 text-primary mx-auto mb-4 animate-pulse" />
            <h3 className="text-xl font-semibold text-foreground mb-2">
              Generating Questions for {candidateName}
            </h3>
            <p className="text-foreground/70 mb-2">
              Creating 7 <span className="text-accent font-semibold">{role}</span> technical questions
            </p>
            <p className="text-foreground/70">
              and 8 personalized questions from your resume...
            </p>
          </div>
        </Card>
      </div>
    );
  }

  const progress = ((currentQuestionIndex + 1) / questions.length) * 100;

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/10 p-6">
      <div className="max-w-4xl mx-auto">
        {!recognitionCtor && (
          <Card className="p-3 mb-4 bg-card/30 border border-destructive/30">
            <div className="text-sm text-destructive">
              Speech recognition is not supported in this browser. Use Chrome or Edge.
            </div>
          </Card>
        )}
        {!isSecureContextOk && (
          <Card className="p-3 mb-4 bg-card/30 border border-yellow-400/30">
            <div className="text-sm text-yellow-400">
              For speech recognition, open this app via HTTPS or run on localhost.
            </div>
          </Card>
        )}
        <div className="flex justify-between items-center mb-6">
          <Button 
            onClick={onBack}
            variant="ghost"
            className="text-foreground/80 hover:text-foreground hover:bg-primary/10"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <div className="text-center">
            <div className="text-foreground/80 text-sm">
              {candidateName} • {role}
            </div>
            <div className="text-foreground font-medium">
              Question {currentQuestionIndex + 1} of {questions.length}
            </div>
          </div>
          <div className="text-right min-w-[120px]">
            <span className="px-3 py-1 text-sm rounded-full bg-primary/20 text-primary font-semibold">
              {Math.floor(remainingSeconds / 60)}:{String(remainingSeconds % 60).padStart(2, '0')}
            </span>
          </div>
        </div>

        <div className="mb-6">
          <Progress value={progress} className="h-2" />
        </div>

        <Card className="p-8 bg-card/30 backdrop-blur-xl border border-primary/20 shadow-2xl mb-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-foreground bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Technical Interview
            </h2>
            <Button
              onClick={speakQuestion}
              variant="outline"
              size="sm"
              className="border-primary/40 hover:bg-primary/10 text-foreground"
            >
              <Volume2 className="w-4 h-4 mr-2" />
              Listen
            </Button>
          </div>

          <div className="mb-8">
            <div className="flex items-center gap-2 mb-4">
              <h3 className="text-lg font-semibold text-foreground">
                Question {currentQuestionIndex + 1}:
              </h3>
              <span className="px-2 py-1 bg-primary/20 text-primary text-xs font-semibold rounded-full">
                {currentQuestionIndex < 7 ? "Role-based" : "Resume-based"}
              </span>
            </div>
            <p className="text-xl text-foreground/90 leading-relaxed">
              {questions[currentQuestionIndex]}
            </p>
          </div>

          <div className="space-y-6">
            <div className="text-center">
              <p className="text-foreground/70 mb-6 font-medium">
                Please record your audio response for this question
              </p>
              <Button
                onClick={toggleRecording}
                variant={isRecording ? "destructive" : "default"}
                size="lg"
                className={`${isRecording ? "bg-destructive hover:bg-destructive/90" : "bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90"} text-primary-foreground shadow-xl transition-all duration-300 hover:scale-105 px-8 py-4`}
                disabled={answeredOnce.has(currentQuestionIndex)}
              >
                {isRecording ? (
                  <>
                    <MicOff className="w-6 h-6 mr-3" />
                    Stop Recording
                  </>
                ) : (
                  <>
                    <Mic className="w-6 h-6 mr-3" />
                    {answeredOnce.has(currentQuestionIndex) ? "Recording Disabled" : "Start Recording"}
                  </>
                )}
              </Button>
              {isRecording && (
                <div className="mt-6 flex items-center justify-center">
                  <div className="w-4 h-4 bg-destructive rounded-full animate-pulse mr-3"></div>
                  <span className="text-destructive font-semibold">Recording in progress...</span>
                </div>
              )}
              <div className="mt-6">
                <Textarea
                  value={(transcript + (interim ? (transcript ? " " : "") + interim : "")).trim() || currentAnswer}
                  readOnly
                  placeholder="Your answer will appear here as you speak..."
                  className="w-full h-28"
                />
              </div>
            </div>
          </div>
        </Card>

        <div className="flex justify-between">
          <Button
            onClick={handlePrevious}
            disabled={currentQuestionIndex === 0}
            variant="outline"
            className="border-primary/40 hover:bg-primary/10 text-foreground"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Previous
          </Button>
          
          <Button
            onClick={handleNext}
            className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-primary-foreground shadow-xl transition-all duration-300 hover:scale-105"
          >
            {currentQuestionIndex < questions.length - 1 ? (
              <>
                Next Question
                <ArrowRight className="w-4 h-4 ml-2" />
              </>
            ) : (
              "Complete Technical Interview"
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};