import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Bot, Brain, Users, FileText, Mic, Volume2 } from "lucide-react";

interface WelcomeScreenProps {
  onNext: () => void;
}

export const WelcomeScreen = ({ onNext }: WelcomeScreenProps) => {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="max-w-2xl w-full text-center">
        <div className="flex justify-center mb-8">
          <div className="relative">
            <Bot className="w-24 h-24 text-primary animate-pulse" />
            <div className="absolute -top-2 -right-2 w-8 h-8 bg-accent rounded-full flex items-center justify-center shadow-glow">
              <Brain className="w-5 h-5 text-accent-foreground" />
            </div>
          </div>
        </div>
        
        <h1 className="text-6xl font-bold text-foreground mb-12 bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">
          AI Interview Assistant
        </h1>
        
        <Button 
          onClick={onNext}
          size="lg"
          className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow text-xl px-12 py-8 transition-all duration-300 hover:scale-105 border border-primary/20"
        >
          Start Interview Session
        </Button>
      </div>
    </div>
  );
};