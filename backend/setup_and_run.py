#!/usr/bin/env python3
"""
Setup and run script for Resume Savvy RAG API
"""
import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_env():
    """Check environment variables"""
    print("🔍 Checking environment...")
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ GEMINI_API_KEY found in environment")
    else:
        print("⚠️ GEMINI_API_KEY not found - will use fallback questions")
    
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    print(f"🤖 Using Gemini model: {model}")

def run_server():
    """Run the FastAPI server"""
    print("🚀 Starting server...")
    try:
        import uvicorn
        from app import app
        # In production, avoid `reload=True`. Enable dev reload by setting DEV_MODE=1 in env.
        dev_mode = os.getenv("DEV_MODE", "0")
        reload_flag = True if dev_mode.lower() in ("1", "true", "yes") else False
        uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), reload=reload_flag)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies first: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    print("🎯 Resume Savvy RAG API Setup")
    print("=" * 40)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    check_env()
    
    # Ask user if they want to install dependencies
    install = input("\n📦 Install dependencies? (y/n): ").lower().strip()
    if install == 'y':
        if install_dependencies():
            print("\n🎉 Setup complete! Starting server...")
            run_server()
        else:
            print("\n❌ Setup failed. Please install dependencies manually.")
    else:
        print("\n🚀 Starting server with existing dependencies...")
        run_server()
