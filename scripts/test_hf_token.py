#!/usr/bin/env python3
"""
API Token Test Script for HuggingFace Router
Tests if the HF_TOKEN works with different model providers
"""

import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

def test_hf_token():
    """Test HuggingFace API token with different models."""
    print("🔍 Testing HuggingFace API Token")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Get token
    hf_token = os.getenv("HF_TOKEN") or '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'
    print(f"📋 Using token: {hf_token[:10]}...{hf_token[-5:] if len(hf_token) > 15 else hf_token}")

    # Initialize client
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )
        print("✅ Client initialized successfully")
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return False

    # Test models from the benchmark (corrected format)
    test_models = [
        "meta-llama/Llama-3.1-8B-Instruct:cerebras",
        "Qwen/Qwen2.5-Coder-32B-Instruct:together",
        "deepseek-ai/DeepSeek-R1:novita",
        "microsoft/codebert-base",  # Corrected format
        "openai/gpt-oss-20b:nebius"
    ]

    print(f"\n🧪 Testing {len(test_models)} models...")
    print("-" * 50)

    results = {}

    for model_id in test_models:
        print(f"\n🔬 Testing model: {model_id}")
        try:
            # Simple test prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Respond with just 'OK' if you can see this message."
                },
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]

            completion = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=10,
                temperature=0.1,
                timeout=30
            )

            response = completion.choices[0].message.content
            if response is None:
                print(f"❌ Failed: API returned None content")
                results[model_id] = {"status": "failed", "error": "API returned None content"}
                continue
                
            response = response.strip()
            print(f"✅ Success: {response[:50]}...")
            results[model_id] = {"status": "success", "response": response}

        except Exception as e:
            error_str = str(e)
            print(f"❌ Failed: {error_str}")
            results[model_id] = {"status": "failed", "error": error_str}

    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 50)

    successful = 0
    failed = 0

    for model_id, result in results.items():
        status = result["status"]
        if status == "success":
            successful += 1
            print(f"✅ {model_id}: Working")
        else:
            failed += 1
            print(f"❌ {model_id}: Failed - {result.get('error', 'Unknown error')}")

    print(f"\n🎯 Results: {successful} working, {failed} failed")

    if successful == 0:
        print("🚨 No models are working! Check your HF_TOKEN.")
        return False
    elif successful < len(test_models):
        print("⚠️  Some models failed. This might explain benchmark issues.")
        return False
    else:
        print("🎉 All models working! Token is valid.")
        return True

def test_token_basic():
    """Basic token validation without API calls."""
    print("🔍 Basic Token Validation")
    print("=" * 30)

    hf_token = os.getenv("HF_TOKEN") or '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'

    if not hf_token:
        print("❌ No token found!")
        return False

    if not hf_token.startswith("hf_"):
        print("❌ Token doesn't start with 'hf_'")
        return False

    if len(hf_token) < 20:
        print("❌ Token seems too short")
        return False

    print("✅ Token format looks valid")
    return True

if __name__ == "__main__":
    print("🚀 HuggingFace API Token Test Script")
    print("====================================\n")

    # Basic validation first
    if not test_token_basic():
        sys.exit(1)

    print()

    # Full API test
    success = test_hf_token()

    if success:
        print("\n🎉 Token test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Token test found issues!")
        sys.exit(1)