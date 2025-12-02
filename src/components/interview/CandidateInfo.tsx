import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { User, Briefcase, ArrowLeft, ArrowRight } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface CandidateInfoProps {
  onNext: (data: { candidateName: string; role: string }) => void;
  onBack: () => void;
}

const ROLES = [
  "Frontend Developer",
  "Backend Developer", 
  "Full Stack Developer",
  "DevOps Engineer",
  "Data Scientist",
  "Machine Learning Engineer",
  "Product Manager",
  "UI/UX Designer",
  "QA Engineer",
  "Mobile Developer",
  "Cloud Architect",
  "Cybersecurity Specialist"
];

export const CandidateInfo = ({ onNext, onBack }: CandidateInfoProps) => {
  const [candidateName, setCandidateName] = useState("");
  const [selectedRole, setSelectedRole] = useState("");
  const { toast } = useToast();

  const handleSubmit = () => {
    if (!candidateName.trim()) {
      toast({
        title: "Name required",
        description: "Please enter your full name",
        variant: "destructive"
      });
      return;
    }

    if (!selectedRole) {
      toast({
        title: "Role required", 
        description: "Please select the role you're applying for",
        variant: "destructive"
      });
      return;
    }

    onNext({ candidateName, role: selectedRole });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5 flex items-center justify-center p-6">
      <div className="max-w-lg w-full">
        <Button 
          onClick={onBack}
          variant="ghost"
          className="mb-6 text-foreground/80 hover:text-foreground hover:bg-primary/10"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>

        <Card className="p-8 bg-card/30 backdrop-blur-xl border border-primary/20 shadow-2xl">
          <div className="text-center mb-8">
            <div className="flex justify-center mb-6">
              <div className="relative">
                <div className="p-4 bg-primary/20 rounded-full border border-primary/30">
                  <User className="w-12 h-12 text-primary" />
                </div>
                <div className="absolute -bottom-1 -right-1 p-2 bg-accent/20 rounded-full border border-accent/30">
                  <Briefcase className="w-4 h-4 text-accent" />
                </div>
              </div>
            </div>
            <h2 className="text-3xl font-bold text-foreground mb-3 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Tell Us About Yourself
            </h2>
            <p className="text-foreground/70 leading-relaxed">
              Let's get to know you better to create a personalized interview experience
            </p>
          </div>

          <div className="space-y-6">
            <div className="space-y-2">
              <label className="text-sm font-semibold text-foreground/90 uppercase tracking-wide">
                Full Name
              </label>
              <div className="relative">
                <User className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-primary/60" />
                <Input
                  type="text"
                  placeholder="Enter your full name"
                  value={candidateName}
                  onChange={(e) => setCandidateName(e.target.value)}
                  className="pl-12 h-12 bg-background/50 border border-primary/20 focus:border-primary/50 text-foreground placeholder:text-muted-foreground"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-semibold text-foreground/90 uppercase tracking-wide">
                Role Applying For
              </label>
              <div className="relative">
                <Briefcase className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-primary/60 z-10" />
                <Select value={selectedRole} onValueChange={setSelectedRole}>
                  <SelectTrigger className="pl-12 h-12 bg-background/50 border border-primary/20 focus:border-primary/50 text-foreground">
                    <SelectValue placeholder="Select your target role" />
                  </SelectTrigger>
                  <SelectContent className="bg-card border border-primary/20">
                    {ROLES.map((role) => (
                      <SelectItem 
                        key={role} 
                        value={role}
                        className="text-foreground hover:bg-primary/10 focus:bg-primary/10"
                      >
                        {role}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="pt-4">
              <Button
                onClick={handleSubmit}
                disabled={!candidateName.trim() || !selectedRole}
                className="w-full h-12 bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-primary-foreground font-semibold shadow-xl transition-all duration-300 hover:scale-[1.02] disabled:opacity-50 disabled:hover:scale-100"
                size="lg"
              >
                Continue to Resume Upload
                <ArrowRight className="w-5 h-5 ml-2" />
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};