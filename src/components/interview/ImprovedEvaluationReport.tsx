import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { 
  TrendingUp, 
  Award, 
  Brain, 
  Users, 
  Download, 
  RotateCcw,
  CheckCircle,
  AlertCircle,
  Target,
  Star,
  MessageSquare
} from "lucide-react";
import { InterviewData } from "@/pages/Index";
import { evaluateInterview, BACKEND_URL } from "@/lib/api";

interface EvaluationReportProps {
  interviewData: InterviewData;
  onRestart: () => void;
}

export const ImprovedEvaluationReport = ({ interviewData, onRestart }: EvaluationReportProps) => {
  const [isGenerating, setIsGenerating] = useState(true);
  const [scores, setScores] = useState({
    technical: 0,
    hr: 0,
    overall: 0
  });
  const [evaluationResult, setEvaluationResult] = useState<any>(null);

  useEffect(() => {
    const runEvaluation = async () => {
      setIsGenerating(true);
      try {
        const sessionId = interviewData.sessionId || "";
        const role = interviewData.role || "";
        const technicalAnswers = (interviewData.technicalAnswers || []).map(a => ({ question: a.question, answer: a.answer, type: a.type }));
        const hrAnswers = (interviewData.hrAnswers || []).map(a => ({ question: a.question, answer: a.answer, type: a.type }));
        const result = await evaluateInterview(sessionId, role, technicalAnswers, hrAnswers);
        setEvaluationResult(result);
        const technicalPercent = result.technical?.technical_percent ?? (result.technical?.score ? Math.round((result.technical.score / 75) * 100) : 0);
        const hrPercent = result.hr?.hr_percent ?? (result.hr?.score ? Math.round((result.hr.score / 25) * 100) : 0);
        const overallPercent = result.overall_percent ?? (result.overall ? Math.round((result.overall / 100) * 100) : 0);
        setScores({ technical: technicalPercent, hr: hrPercent, overall: overallPercent });
      } catch (e) {
        // Show error state instead of generating random scores
        console.error('Evaluation failed:', e);
        setScores({ technical: 0, hr: 0, overall: 0 });
        setEvaluationResult({ error: 'Evaluation could not be completed. Please ensure all answers were recorded.' });
      } finally {
        setIsGenerating(false);
      }
    };
    runEvaluation();
  }, [interviewData]);

  const getScoreColor = (score: number) => {
    if (score >= 90) return "text-primary";
    if (score >= 80) return "text-accent";
    if (score >= 70) return "text-yellow-400";
    return "text-destructive";
  };

  const getScoreBadge = (score: number) => {
    if (score >= 90) return { label: "Exceptional", variant: "default" as const };
    if (score >= 80) return { label: "Strong", variant: "secondary" as const };
    if (score >= 70) return { label: "Competent", variant: "outline" as const };
    return { label: "Developing", variant: "destructive" as const };
  };

  const generateDetailedInsights = () => {
    const insights = [
      {
        type: "strength",
        icon: <Star className="w-5 h-5 text-primary" />,
        category: "Technical Excellence",
        text: "Demonstrated strong understanding of software architecture and design patterns"
      },
      {
        type: "strength",
        icon: <MessageSquare className="w-5 h-5 text-primary" />,
        category: "Communication",
        text: "Clear and articulate responses with good use of technical terminology"
      },
      {
        type: "improvement",
        icon: <AlertCircle className="w-5 h-5 text-yellow-400" />,
        category: "Deep Dive",
        text: "Could provide more specific examples from recent projects to illustrate points"
      },
      {
        type: "strength",
        icon: <CheckCircle className="w-5 h-5 text-primary" />,
        category: "Problem Solving",
        text: "Shows systematic approach to breaking down complex problems"
      },
      {
        type: "improvement",
        icon: <AlertCircle className="w-5 h-5 text-yellow-400" />,
        category: "Leadership",
        text: "Consider sharing more examples of team collaboration and mentoring experiences"
      },
      {
        type: "strength",
        icon: <Star className="w-5 h-5 text-primary" />,
        category: "Cultural Fit",
        text: "Values align well with company culture and shows growth mindset"
      }
    ];
    return insights;
  };

  const getRoleScore = () => {
    // Return percent value for role section (0-100)
    if (evaluationResult?.technical?.role_percent !== undefined) return evaluationResult.technical.role_percent;
    // derive from answers if present
    const answers = evaluationResult?.technical?.answers || [];
    const roleMarks = answers.filter((a: any) => a.type === 'role').map((a: any) => a.marks ?? Math.round((a.score ?? 0) / 20));
    if (roleMarks.length) {
      const sumMarks = roleMarks.reduce((s: number, v: number) => s + v, 0);
      const percent = Math.round((sumMarks / (7 * 5)) * 100);
      return percent;
    }
    return 0;
  };

  const getResumeScore = () => {
    // Return percent value for resume section (0-100)
    if (evaluationResult?.technical?.resume_percent !== undefined) return evaluationResult.technical.resume_percent;
    const answers = evaluationResult?.technical?.answers || [];
    const resumeMarks = answers.filter((a: any) => a.type === 'resume').map((a: any) => a.marks ?? Math.round((a.score ?? 0) / 20));
    if (resumeMarks.length) {
      const sumMarks = resumeMarks.reduce((s: number, v: number) => s + v, 0);
      const percent = Math.round((sumMarks / (8 * 5)) * 100);
      return percent;
    }
    return 0;
  };

  // Helper to get marks (integers) for payload
  const getRoleMarks = () => {
    if (evaluationResult?.technical?.role_score !== undefined) return evaluationResult.technical.role_score;
    // fallback derive from percent
    const p = getRoleScore();
    return Math.round((p / 100) * 35);
  };

  const getResumeMarks = () => {
    if (evaluationResult?.technical?.resume_score !== undefined) return evaluationResult.technical.resume_score;
    const p = getResumeScore();
    return Math.round((p / 100) * 40);
  };

  const getHrMarks = () => {
    if (evaluationResult?.hr?.score !== undefined) return evaluationResult.hr.score;
    const p = scores.hr;
    return Math.round((p / 100) * 25);
  };

  if (isGenerating) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <Card className="p-8 bg-black border-black max-w-md w-full">
          <div className="text-center">
            <Brain className="w-16 h-16 text-yellow-400 mx-auto mb-4 animate-pulse" />
            <h3 className="text-xl font-semibold text-yellow-400 mb-2">
              Advanced AI Analysis in Progress
            </h3>
            <p className="text-yellow-300 mb-4">
              Our enhanced evaluation system is analyzing your audio responses, body language, and communication patterns...
            </p>
            <Progress value={75} className="h-2 bg-yellow-400/20" />
            <p className="text-yellow-300 text-sm mt-2">
              Using multiple evaluation models for comprehensive assessment
            </p>
          </div>
        </Card>
      </div>
    );
  }

  // Show error state if evaluation failed
  if (evaluationResult?.error) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <Card className="p-8 bg-black border-destructive max-w-md w-full">
          <div className="text-center">
            <AlertCircle className="w-16 h-16 text-destructive mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-destructive mb-2">
              Evaluation Could Not Be Completed
            </h3>
            <p className="text-card-foreground/80 mb-6">
              {evaluationResult.error}
            </p>
            <Button onClick={() => location.reload()} className="w-full">
              Reload and Try Again
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  const insights = (() => {
    if (evaluationResult) {
      const tech = evaluationResult.technical?.answers || [];
      const hr = evaluationResult.hr?.answers || [];
      const all = [...tech, ...hr];
      const strengths = all.filter((a: any) => (a.score ?? 0) >= 80).map((a: any) => ({ type: 'strength', icon: <Star className="w-5 h-5 text-primary" />, category: '', text: a.feedback || 'Strong performance' }));
      const improvements = all.filter((a: any) => (a.score ?? 100) < 70).map((a: any) => ({ type: 'improvement', icon: <AlertCircle className="w-5 h-5 text-yellow-400" />, category: '', text: a.feedback || 'Needs improvement' }));
      return [...strengths, ...improvements].slice(0,6);
    }
    return generateDetailedInsights();
  })();

  const downloadReport = async (format: 'pdf' | 'docx') => {
    try {
      // Prepare concise performance-only payload: scores + strengths + improvements
      let strengths: string[] = [];
      let improvements: string[] = [];

      // Prefer LLM-generated consolidated feedback (server-side) when available
      if (evaluationResult?.feedback_summary) {
        const fb = evaluationResult.feedback_summary;
        const techFb: string[] = Array.isArray(fb.technical_feedback) ? fb.technical_feedback : [];
        const commFb: string[] = Array.isArray(fb.communication_feedback) ? fb.communication_feedback : [];
        const hrFb: string[] = Array.isArray(fb.hr_feedback) ? fb.hr_feedback : [];
        const tipsFb: string[] = Array.isArray(fb.tips_to_improve) ? fb.tips_to_improve : [];
        // strengths: use technical + communication points
        strengths = [...techFb, ...commFb].filter(Boolean).slice(0,5);
        // improvements/tips: prefer actionable tips and HR feedback
        improvements = [...tipsFb, ...hrFb].filter(Boolean).slice(0,5);
      } else if (evaluationResult) {
        try {
          const techAnswers = evaluationResult.technical?.answers || [];
          const hrAnswers = evaluationResult.hr?.answers || [];
          const all = [...techAnswers, ...hrAnswers];
          // strengths: feedback from answers with high score
          strengths = all.filter((a: any) => (a.score ?? 0) >= 80).map((a: any) => a.feedback).filter(Boolean).slice(0,5);
          // improvements: feedback from answers with low score
          improvements = all.filter((a: any) => (a.score ?? 100) < 70).map((a: any) => a.feedback).filter(Boolean).slice(0,5);
        } catch (e) {
          // fallback
          strengths = generateDetailedInsights().filter(i => i.type === 'strength').map(s => s.text).slice(0,5);
          improvements = generateDetailedInsights().filter(i => i.type === 'improvement').map(s => s.text).slice(0,5);
        }
      } else {
        strengths = generateDetailedInsights().filter(i => i.type === 'strength').map(s => s.text).slice(0,5);
        improvements = generateDetailedInsights().filter(i => i.type === 'improvement').map(s => s.text).slice(0,5);
      }

      const roleMarks = getRoleMarks();
      const resumeMarks = getResumeMarks();
      const hrMarks = getHrMarks();
      const technicalMarks = roleMarks + resumeMarks;
      const totalMarks = technicalMarks + hrMarks;

      const payload = {
        session_id: interviewData.sessionId || '',
        candidate_name: interviewData.candidateName || 'Candidate',
        role: interviewData.role || '',
        overall: scores.overall,
        technical_score: scores.technical,
        hr_score: scores.hr,
        role_score: getRoleScore(),
        resume_score: getResumeScore(),
        // Marks (explicit)
        role_marks: roleMarks,
        resume_marks: resumeMarks,
        technical_marks: technicalMarks,
        hr_marks: hrMarks,
        total_marks: totalMarks,
        strengths,
        improvements,
        // Include detailed LLM-generated feedback arrays when available
        technical_feedback: evaluationResult?.feedback_summary?.technical_feedback || [],
        hr_feedback: evaluationResult?.feedback_summary?.hr_feedback || [],
        communication_feedback: evaluationResult?.feedback_summary?.communication_feedback || [],
        tips_to_improve: evaluationResult?.feedback_summary?.tips_to_improve || []
      };

      const resp = await fetch(`${BACKEND_URL}/report?format=${format}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        console.error('Failed to generate report', resp.status);
        return;
      }
      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `interview-report-${interviewData.sessionId || 'report'}.${format === 'docx' ? 'docx' : 'pdf'}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      console.error('Download failed', e);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-evaluation p-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="p-4 bg-primary/20 rounded-full shadow-glow">
              <Award className="w-12 h-12 text-primary" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-card-foreground mb-2 bg-gradient-primary bg-clip-text text-transparent">
            Advanced AI Interview Analysis
          </h1>
          <p className="text-card-foreground/80 text-lg">
            Comprehensive multi-model evaluation of your interview performance
          </p>
        </div>

        {/* Overall Score */}
        <Card className="p-8 bg-card/10 backdrop-blur-sm border-primary/20 mb-8 shadow-glow">
          <div className="text-center">
            <h2 className="text-2xl font-semibold text-card-foreground mb-6">
              Overall Performance Score
            </h2>
            <div className="flex items-center justify-center mb-6">
              <div className="relative">
                <div className="text-7xl font-bold text-primary">
                {scores.overall}
                <span className="text-2xl font-semibold text-card-foreground/70">/100</span>
                </div>
              </div>
            </div>
            <Badge 
              {...getScoreBadge(scores.overall)} 
              className="text-lg px-4 py-2 mb-4"
            >
              {getScoreBadge(scores.overall).label}
            </Badge>
            <Progress value={scores.overall} className="h-4 max-w-md mx-auto" />
            <p className="text-card-foreground/60 mt-4">
              Based on advanced AI analysis of {interviewData.technicalAnswers.length + interviewData.hrAnswers.length} responses
            </p>
          </div>
        </Card>

        {/* Section Scores: Role-based, Resume-based, HR */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <Card className="p-6 bg-card/10 backdrop-blur-sm border-primary/20 shadow-tech">
            <div className="flex items-center mb-4">
              <Brain className="w-8 h-8 text-primary mr-3" />
              <h3 className="text-xl font-semibold text-card-foreground">Role-based</h3>
            </div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-4xl font-bold text-primary">{getRoleScore()}%</span>
              <Badge {...getScoreBadge(getRoleScore())}>{getScoreBadge(getRoleScore()).label}</Badge>
            </div>
            <Progress value={getRoleScore()} className="h-3 mb-4" />
            <p className="text-card-foreground/70 text-sm">Out of 35 marks (7 questions × 5)</p>
          </Card>

          <Card className="p-6 bg-card/10 backdrop-blur-sm border-primary/20 shadow-tech">
            <div className="flex items-center mb-4">
              <TrendingUp className="text-4xl font-bold text-primary" />
              <h3 className="text-xl font-semibold text-card-foreground">Resume-based</h3>
            </div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-4xl font-bold text-primary">{getResumeScore()}%</span>
              <Badge {...getScoreBadge(getResumeScore())}>{getScoreBadge(getResumeScore()).label}</Badge>
            </div>
            <Progress value={getResumeScore()} className="h-3 mb-4" />
            <p className="text-card-foreground/70 text-sm">Out of 40 marks (8 questions × 5)</p>
          </Card>

          <Card className="p-6 bg-card/10 backdrop-blur-sm border-primary/20 shadow-tech">
            <div className="flex items-center mb-4">
              <Users className="w-8 h-8 text-accent mr-3" />
              <h3 className="text-xl font-semibold text-card-foreground">HR</h3>
            </div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-4xl font-bold text-accent">{scores.hr}%</span>
              <Badge {...getScoreBadge(scores.hr)}>{getScoreBadge(scores.hr).label}</Badge>
            </div>
            <Progress value={scores.hr} className="h-3 mb-4" />
            <p className="text-card-foreground/70 text-sm">Out of 25 marks (5 questions × 5)</p>
          </Card>
        </div>

        {/* Detailed Insights */}
        <Card className="p-6 bg-card/10 backdrop-blur-sm border-primary/20 mb-8 shadow-glow">
          <div className="flex items-center mb-6">
            <Target className="w-6 h-6 text-primary mr-3" />
            <h3 className="text-xl font-semibold text-card-foreground">
              Detailed Analysis & Recommendations
            </h3>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {insights.map((insight, index) => (
              <div 
                key={index} 
                className={`p-4 rounded-lg border ${
                  insight.type === 'strength' 
                    ? 'bg-primary/5 border-primary/20' 
                    : 'bg-black border-black'
                }`}
              >
                <div className="flex items-start space-x-3">
                  {insight.icon}
                  <div>
                    <h4 className={`${insight.type === 'strength' ? 'font-medium text-card-foreground' : 'font-medium text-yellow-400'} mb-1`}>
                      {insight.category}
                    </h4>
                    <p className={`${insight.type === 'strength' ? 'text-card-foreground/80' : 'text-yellow-300 text-sm'}`}>
                      {insight.text}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Action Buttons */}
        <div className="flex justify-center space-x-4">
          <Button
            onClick={onRestart}
            variant="outline"
            className="border-primary/50 hover:bg-primary/10"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Take Another Interview
          </Button>
          <Button onClick={() => downloadReport('docx')} className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow">
            <Download className="w-4 h-4 mr-2" />
            Download Detailed Report
          </Button>
        </div>
      </div>
    </div>
  );
};