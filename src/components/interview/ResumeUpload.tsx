import React, { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Upload, FileText, ArrowLeft, CheckCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { uploadResumeExtract } from "@/lib/api";

interface ResumeUploadProps {
  candidateName: string;
  role: string;
  onNext: (data: { resumeContent: string; sessionId: string }) => void;
  onBack: () => void;
}

export const ResumeUpload = ({ candidateName, role, onNext, onBack }: ResumeUploadProps) => {
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [resumeContent, setResumeContent] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleFileUpload = async (file: File) => {
    if (file.type !== "application/pdf") {
      toast({
        title: "Invalid file type",
        description: "Please upload a PDF file",
        variant: "destructive"
      });
      return;
    }

    setResumeFile(file);
    setIsProcessing(true);

    try {
      const id = crypto.randomUUID();
      setSessionId(id);
      const result = await uploadResumeExtract(id, file);
      // Backend indexes chunks; we don't need full text in FE any more
      setResumeContent(`Indexed ${result.chunks_indexed} chunks for session ${id}`);
      toast({
        title: "Resume processed successfully",
        description: `Analyzed and indexed (${result.chunks_indexed} chunks)`
      });
    } catch (error) {
      toast({
        title: "Error processing resume",
        description: "Please try uploading again",
        variant: "destructive"
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSubmit = () => {
    if (!resumeContent) {
      toast({
        title: "Resume required",
        description: "Please upload your resume",
        variant: "destructive"
      });
      return;
    }

    onNext({ resumeContent, sessionId });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5 flex items-center justify-center p-6">
      <div className="max-w-2xl w-full">
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
              <div className="p-4 bg-primary/20 rounded-full border border-primary/30">
                <FileText className="w-12 h-12 text-primary" />
              </div>
            </div>
            <h2 className="text-3xl font-bold text-foreground mb-3 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Upload Your Resume
            </h2>
            <p className="text-foreground/70 leading-relaxed">
              Hello <span className="text-primary font-semibold">{candidateName}</span>! 
              Upload your resume for the <span className="text-accent font-semibold">{role}</span> position.
              Our AI will analyze it to create personalized interview questions.
            </p>
          </div>

          <div className="space-y-6">
            <div>
              <label className="text-sm font-semibold text-foreground/90 uppercase tracking-wide">
                Resume (PDF)
              </label>
              <div 
                className="border-2 border-dashed border-primary/30 rounded-xl p-8 text-center hover:border-primary/60 hover:bg-primary/5 transition-all duration-300 cursor-pointer group"
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileUpload(file);
                  }}
                  className="hidden"
                />
                
                {resumeFile ? (
                  <div className="flex items-center justify-center space-x-3">
                    <CheckCircle className="w-10 h-10 text-primary" />
                    <div className="text-left">
                      <span className="text-foreground font-semibold block">
                        {resumeFile.name}
                      </span>
                      <span className="text-accent text-sm">
                        Ready for analysis
                      </span>
                    </div>
                  </div>
                ) : (
                  <div>
                    <Upload className="w-16 h-16 text-primary/60 mx-auto mb-4 group-hover:text-primary transition-colors" />
                    <p className="text-foreground/80 mb-2 font-medium">
                      Click to upload your resume
                    </p>
                    <p className="text-sm text-muted-foreground">
                      PDF files only, max 10MB
                    </p>
                  </div>
                )}
              </div>
            </div>

            <Button
              onClick={handleSubmit}
              disabled={isProcessing || !resumeContent}
              className="w-full h-12 bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-primary-foreground font-semibold shadow-xl transition-all duration-300 hover:scale-[1.02] disabled:opacity-50 disabled:hover:scale-100"
              size="lg"
            >
              {isProcessing ? "Processing Resume..." : "Start Technical Interview"}
              {!isProcessing && <ArrowLeft className="w-5 h-5 ml-2 rotate-180" />}
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
};