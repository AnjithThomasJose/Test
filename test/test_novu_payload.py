#!/usr/bin/env python3
"""
Test script for Novu integration payload
Tests the /analyze-resume-callback endpoint with the new name field
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_novu_payload():
    """Test the payload with name field for Novu integration"""
    
    # Use a whitelisted callback URL or set CALLBACK_WHITELIST_DOMAINS env var
    # For testing, you can use: https://apis-buh3qzwapq-uc.a.run.app/webhook
    payload = {
        "uid": "695cbd86f9d367c59649f3ea",
        "callback_url": "https://apis-buh3qzwapq-uc.a.run.app/webhook",  # Whitelisted domain
        "tenant_id": "test-tenant-123",
        "name": "Anjith Thomas Jose",
        "email": "anjithtj@theknowledgeartisans.com",
        "resume_url": "https://storage.googleapis.com/test-bucket/sample.pdf"  # Replace with actual test resume URL
    }
    
    print("=" * 80)
    print("Testing Novu Integration Payload")
    print("=" * 80)
    print(f"\n📤 Sending request to: {BASE_URL}/analyze-resume-callback")
    print(f"\n📋 Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.patch(
            f"{BASE_URL}/analyze-resume-callback",
            json=payload,
            headers={
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        print(f"\n📥 Response Status: {response.status_code}")
        print(f"📥 Response Headers: {dict(response.headers)}")
        
        try:
            response_data = response.json()
            print(f"\n📥 Response Body:")
            print(json.dumps(response_data, indent=2))
        except:
            print(f"\n📥 Response Text: {response.text}")
        
        if response.status_code == 200:
            print("\n✅ Request successful!")
            print(f"   Status: {response_data.get('status', 'unknown')}")
            print(f"   UID: {response_data.get('uid', 'unknown')}")
            print(f"   Session ID: {response_data.get('session_id', 'unknown')}")
            print("\n💡 Check your server logs to see if Novu trigger was called")
            print("💡 The resume parsing will happen in the background")
        else:
            print(f"\n❌ Request failed with status {response.status_code}")
            if response_data:
                print(f"   Error: {response_data.get('detail', 'Unknown error')}")
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to server at {BASE_URL}")
        print("   Make sure the server is running: uvicorn app:app --reload --port 8000")
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    test_novu_payload()
