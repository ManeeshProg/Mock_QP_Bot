import requests
import json

# Test the API endpoints
BASE_URL = "http://localhost:8000"

def test_health():
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_root():
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Root endpoint: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Root endpoint failed: {e}")
        return False

def test_technical_questions():
    try:
        data = {
            "session_id": "test-session-123",
            "role": "Frontend Developer",
            "count_role": 3,
            "count_resume": 2
        }
        response = requests.post(f"{BASE_URL}/questions/technical", json=data)
        print(f"Technical questions: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Technical questions failed: {e}")
        return False

def test_hr_questions():
    try:
        data = {
            "session_id": "test-session-123"
        }
        response = requests.post(f"{BASE_URL}/questions/hr", json=data)
        print(f"HR questions: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"HR questions failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Resume Savvy RAG API...")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health()
    root_ok = test_root()
    
    if health_ok and root_ok:
        print("\n✅ Basic API is working!")
        
        # Test question generation (these might fail if no session exists)
        print("\nTesting question generation...")
        test_technical_questions()
        test_hr_questions()
    else:
        print("\n❌ API is not responding. Check if server is running on port 8000")
