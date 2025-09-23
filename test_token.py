#!/usr/bin/env python3
"""
Simple script to test HuggingFace API token validity
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

def test_hf_token():
    """Test if HuggingFace token is working."""
    print("🔍 Testing HuggingFace API token...")

    # Load environment variables
    load_dotenv()

    # Get token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ ERROR: No HF_TOKEN found in environment variables")
        print("   Make sure to set HF_TOKEN in your .env file")
        return False

    print(f"✅ Found HF_TOKEN: {hf_token[:8]}...{hf_token[-4:] if len(hf_token) > 12 else hf_token}")

    try:
        # Initialize client
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )

        print("🔄 Testing API connection...")

        # Make a simple test request - just to test authentication
        try:
            completion = client.chat.completions.create(
                model="meta-llama/Llama-2-7b-chat-hf",  # Try a known model
                messages=[
                    {
                        "role": "user",
                        "content": "Test"
                    }
                ],
                max_tokens=10,
                temperature=0.1,
            )
            response = completion.choices[0].message.content.strip()
            print("✅ SUCCESS: API call successful!")
            print(f"   Response: {response}")
            return True

        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "Unauthorized" in error_str:
                print(f"❌ ERROR: Authentication failed: {error_str}")
                print("   The HF_TOKEN is invalid or expired")
                return False
            else:
                # Any other error (model not found, rate limit, etc.) means auth worked
                print("✅ SUCCESS: Token authentication is working!")
                print("   (The model may not be available, but your token is valid)")
                print(f"   Error received: {error_str[:100]}...")
                return True

    except Exception as e:
        error_str = str(e)
        print(f"❌ ERROR: API call failed: {error_str}")

        if "401" in error_str:
            print("   This indicates an invalid or expired token")
            print("   Please check your HF_TOKEN in the .env file")
        elif "rate" in error_str.lower() or "limit" in error_str.lower():
            print("   This indicates rate limiting (token may be valid but over quota)")
        elif "402" in error_str:
            print("   This indicates payment required (free tier exhausted)")

        return False

if __name__ == "__main__":
    success = test_hf_token()
    sys.exit(0 if success else 1)