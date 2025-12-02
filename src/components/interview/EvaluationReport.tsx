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
  Target
} from "lucide-react";
import { InterviewData } from "@/pages/Index";

interface EvaluationReportProps {
  interviewData: InterviewData;
  onRestart: () => void;
}

export const EvaluationReport = ({ interviewData, onRestart }: EvaluationReportProps) => {
  const [isGenerating, setIsGenerating] = useState(true);
  const [scores, setScores] = useState({
    technical: 0,
    hr: 0,
    overall: 0
  });

  useEffect(() => {
    const runEvaluation = async () => {
      setIsGenerating(true);
      try {
        const sessionId = interviewData.sessionId || "";
        const role = interviewData.role || "";
        const technicalAnswers = (interviewData.technicalAnswers || []).map(a => ({ question: a.question, answer: a.answer, type: a.type }));
        const hrAnswers = (interviewData.hrAnswers || []).map(a => ({ question: a.question, answer: a.answer, type: a.type }));
        const api = await import("@/lib/api");
        const result = await api.evaluateInterview(sessionId, role, technicalAnswers, hrAnswers);
        // Expecting result.technical.score (percentage), result.hr.score (percentage), result.overall (0-100)
        const technicalScore = result.technical?.score ?? 0;
        const hrScore = result.hr?.score ?? 0;
        const overallScore = result.overall ?? 0;
        setScores({ technical: technicalScore, hr: hrScore, overall: overallScore });
      } catch (e) {
        console.error('Evaluation failed:', e);
        setScores({ technical: 0, hr: 0, overall: 0 });
      } finally {
        setIsGenerating(false);
      }
    };

    runEvaluation();
  }, [interviewData]);

  const getScoreColor = (score: number) => {
    if (score >= 90) return "text-green-500";
    if (score >= 80) return "text-blue-500";
    if (score >= 70) return "text-yellow-500";
    return "text-red-500";
  };

  const getScoreBadge = (score: number) => {
    if (score >= 90) return { label: "Excellent", variant: "default" as const };
    if (score >= 80) return { label: "Good", variant: "secondary" as const };
    if (score >= 70) return { label: "Fair", variant: "outline" as const };
    return { label: "Needs Improvement", variant: "destructive" as const };
  };

  const generateInsights = () => {
    const insights = [
      {
        type: "strength",
        icon: <CheckCircle className="w-5 h-5 text-green-500" />,
        text: "Strong technical knowledge demonstrated in React and Node.js"
      },
      {
        type: "improvement",
        icon: <AlertCircle className="w-5 h-5 text-yellow-500" />,
        text: "Could improve communication clarity in complex technical explanations"
      },
      {
        type: "strength",
        icon: <CheckCircle className="w-5 h-5 text-green-500" />,
        text: "Excellent problem-solving approach and analytical thinking"
      },
      {
        type: "improvement",
        icon: <AlertCircle className="w-5 h-5 text-yellow-500" />,
        text: "Consider providing more specific examples from your experience"
      }
    ];
    return insights;
  };

  if (isGenerating) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <Card className="p-8 bg-black border-black max-w-md w-full">
          <div className="text-center">
            <Brain className="w-16 h-16 text-yellow-400 mx-auto mb-4 animate-pulse" />
            <h3 className="text-xl font-semibold text-yellow-400 mb-2">
              Evaluating Your Performance
            </h3>
            <p className="text-yellow-300 mb-4">
              Our AI is analyzing your responses and generating your comprehensive report...
            </p>
            <Progress value={66} className="h-2 bg-yellow-400/20" />
          </div>
        </Card>
      </div>
    );
  }

  const insights = generateInsights();

  return (
    <div className="min-h-screen bg-gradient-evaluation p-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="p-4 bg-primary/20 rounded-full">
              <Award className="w-12 h-12 text-primary" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-primary-foreground mb-2">
            Interview Evaluation Report
          </h1>
          <p className="text-primary-foreground/80 text-lg">
            Comprehensive analysis of your technical and HR interview performance
          </p>
        </div>

        {/* Overall Score */}
        <Card className="p-8 bg-card/10 backdrop-blur-sm border-primary/20 mb-8">
          <div className="text-center">
            <h2 className="text-2xl font-semibold text-card-foreground mb-4">
              Overall Performance
            </h2>
            <div className="flex items-center justify-center mb-4">
              <div className="text-6xl font-bold text-primary mr-4">
                {scores.overall}
              </div>
              <div className="text-left">
                <div className="text-2xl text-card-foreground/80"></div>
                <Badge {...getScoreBadge(scores.overall)}>
                  {getScoreBadge(scores.overall).label}
                </Badge>
              </div>
            </div>
            <Progress value={scores.overall} className="h-3 max-w-md mx-auto" />
          </div>
        </Card>

        {/* Detailed Scores */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card className="p-6 bg-card/10 backdrop-blur-sm border-primary/20">
            <div className="flex items-center mb-4">
              <Brain className="w-8 h-8 text-secondary mr-3" />
              <h3 className="text-xl font-semibold text-card-foreground">
                Technical Interview
              </h3>
            </div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-3xl font-bold text-secondary">
                {scores.technical}%
              </span>
              <Badge {...getScoreBadge(scores.technical)}>
                {getScoreBadge(scores.technical).label}
              </Badge>
            </div>
            <Progress value={scores.technical} className="h-2 mb-3" />
            <p className="text-card-foreground/70 text-sm">
              Based on {interviewData.technicalAnswers.length} technical questions
            </p>
          </Card>

          <Card className="p-6 bg-card/10 backdrop-blur-sm border-primary/20">
            <div className="flex items-center mb-4">
              <Users className="w-8 h-8 text-accent mr-3" />
              <h3 className="text-xl font-semibold text-card-foreground">
                HR Interview
              </h3>
            </div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-3xl font-bold text-accent">
                {scores.hr}%
              </span>
              <Badge {...getScoreBadge(scores.hr)}>
                {getScoreBadge(scores.hr).label}
              </Badge>
            </div>
            <Progress value={scores.hr} className="h-2 mb-3" />
            <p className="text-card-foreground/70 text-sm">
              Based on {interviewData.hrAnswers.length} behavioral questions
            </p>
          </Card>
        </div>

        {/* Key Insights */}
        <Card className="p-6 bg-card/10 backdrop-blur-sm border-primary/20 mb-8">
          <div className="flex items-center mb-4">
            <Target className="w-6 h-6 text-primary mr-3" />
            <h3 className="text-xl font-semibold text-card-foreground">
              Key Insights & Recommendations
            </h3>
          </div>
          <div className="space-y-3">
            {insights.map((insight, index) => (
              <div key={index} className={`flex items-start space-x-3 p-3 rounded-lg ${insight.type === 'improvement' ? 'bg-black border-black' : 'bg-muted/20'}`}>
                {insight.icon}
                <p className={`${insight.type === 'improvement' ? 'text-yellow-300' : 'text-card-foreground/90'}`}>{insight.text}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Performance Breakdown */}
        <Card className="p-6 bg-card/10 backdrop-blur-sm border-primary/20 mb-8">
          <h3 className="text-xl font-semibold text-card-foreground mb-4">
            Performance Breakdown
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-card-foreground mb-3">Technical Skills</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-card-foreground/70">Problem Solving</span>
                  <span className="text-secondary font-medium">85%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-card-foreground/70">Technical Knowledge</span>
                  <span className="text-secondary font-medium">78%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-card-foreground/70">Code Quality</span>
                  <span className="text-secondary font-medium">82%</span>
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-medium text-card-foreground mb-3">Soft Skills</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-card-foreground/70">Communication</span>
                  <span className="text-accent font-medium">88%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-card-foreground/70">Leadership</span>
                  <span className="text-accent font-medium">75%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-card-foreground/70">Cultural Fit</span>
                  <span className="text-accent font-medium">90%</span>
                </div>
              </div>
            </div>
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
          <Button className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-ai">
            <Download className="w-4 h-4 mr-2" />
            Download Report
          </Button>
        </div>
      </div>
    </div>
  );
};