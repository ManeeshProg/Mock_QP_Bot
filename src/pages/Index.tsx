import { useEffect, useRef, useState } from "react";
import { WelcomeScreen } from "@/components/interview/WelcomeScreen";
import { CandidateInfo } from "@/components/interview/CandidateInfo";
import { ResumeUpload } from "@/components/interview/ResumeUpload";
import { TechnicalInterview } from "@/components/interview/TechnicalInterview";
import { HRInterview } from "@/components/interview/HRInterview";
import { ImprovedEvaluationReport } from "@/components/interview/ImprovedEvaluationReport";

export type InterviewStage = 'welcome' | 'candidate-info' | 'resume' | 'technical' | 'hr' | 'evaluation';

export interface InterviewData {
  candidateName?: string;
  role?: string;
  resumeContent?: string;
  sessionId?: string;
  technicalAnswers: Array<{ question: string; answer: string; score?: number; type?: string }>;
  hrAnswers: Array<{ question: string; answer: string; score?: number; type?: string }>;
  overallScore?: {
    technical: number;
    hr: number;
    overall: number;
  };
}

const Index = () => {
  const [currentStage, setCurrentStage] = useState<InterviewStage>('welcome');
  const [interviewData, setInterviewData] = useState<InterviewData>({
    technicalAnswers: [],
    hrAnswers: []
  });
  const [deadlineMs, setDeadlineMs] = useState<number | null>(null);
  const [remainingSeconds, setRemainingSeconds] = useState<number>(0);
  const timerRef = useRef<number | null>(null);

  const handleStageChange = (stage: InterviewStage, data?: Partial<InterviewData>) => {
    if (data) {
      setInterviewData(prev => ({ ...prev, ...data }));
    }
    setCurrentStage(stage);

    // Start timer when entering technical stage
    if (stage === 'technical' && !deadlineMs) {
      const dl = Date.now() + 25 * 60 * 1000; // 25 minutes
      setDeadlineMs(dl);
    }
  };

  // Timer effect
  useEffect(() => {
    if (!deadlineMs || (currentStage !== 'technical' && currentStage !== 'hr')) {
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = window.setInterval(() => {
      const secs = Math.max(0, Math.floor((deadlineMs - Date.now()) / 1000));
      setRemainingSeconds(secs);
      if (secs <= 0) {
        window.clearInterval(timerRef.current!);
        timerRef.current = null;
        // Auto end test to evaluation with whatever answers we have
        setCurrentStage('evaluation');
      }
    }, 500);
    return () => {
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [deadlineMs, currentStage]);

  const renderCurrentStage = () => {
    switch (currentStage) {
      case 'welcome':
        return <WelcomeScreen onNext={() => handleStageChange('candidate-info')} />;
      case 'candidate-info':
        return (
          <CandidateInfo 
            onNext={(data) => handleStageChange('resume', data)} 
            onBack={() => handleStageChange('welcome')}
          />
        );
      case 'resume':
        return (
          <ResumeUpload 
            candidateName={interviewData.candidateName || ''}
            role={interviewData.role || ''}
            onNext={(data) => handleStageChange('technical', data)} 
            onBack={() => handleStageChange('candidate-info')}
          />
        );
      case 'technical':
        return (
          <TechnicalInterview 
            candidateName={interviewData.candidateName || ''}
            role={interviewData.role || ''}
            resumeContent={interviewData.resumeContent || ''}
            sessionId={interviewData.sessionId || ''}
            remainingSeconds={remainingSeconds}
            onNext={(answers) => handleStageChange('hr', { technicalAnswers: answers })}
            onBack={() => handleStageChange('resume')}
          />
        );
      case 'hr':
        return (
          <HRInterview 
            sessionId={interviewData.sessionId || ''}
            remainingSeconds={remainingSeconds}
            onNext={(answers) => handleStageChange('evaluation', { hrAnswers: answers })}
            onBack={() => handleStageChange('technical')}
          />
        );
      case 'evaluation':
        return (
          <ImprovedEvaluationReport 
            interviewData={interviewData}
            onRestart={() => {
              setInterviewData({ technicalAnswers: [], hrAnswers: [] });
              handleStageChange('welcome');
            }}
          />
        );
      default:
        return <WelcomeScreen onNext={() => handleStageChange('candidate-info')} />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {renderCurrentStage()}
    </div>
  );
};

export default Index;