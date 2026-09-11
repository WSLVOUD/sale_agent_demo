#!/usr/bin/env python
"""Test first contact workflow."""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_first_contact():
    """Test the first contact flow with a greeting."""
    session_id = f"test-fc-{int(time.time())}"
    
    print(f"Testing first contact flow with session_id: {session_id}")
    print("-" * 50)
    
    # Test greeting
    payload = {
        "session_id": session_id,
        "question": "Hello"
    }
    
    try:
        print("Sending greeting message...")
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=180  # 3 minutes timeout
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        
        print("\n=== Response ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Check if first contact was triggered
        perf = result.get("_perf", {})
        if perf.get("first_contact_intro"):
            print("\n=== First Contact Intro ===")
            print(perf.get("first_contact_intro"))
        
        print("\n=== Test PASSED ===")
        
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_first_contact()
