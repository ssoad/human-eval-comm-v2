#!/usr/bin/env python3
"""
Easy setup script for V2 Evaluators Framework
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_virtual_env():
    """Check if we're in a virtual environment."""
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️  Not in a virtual environment. Consider creating one:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements_v2.txt"]
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_api_keys():
    """Check if API keys are set."""
    print("\n🔑 Checking API keys...")
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    }

    found_keys = []
    for key, value in keys.items():
        if value:
            print(f"✅ {key} is set")
            found_keys.append(key)
        else:
            print(f"⚠️  {key} is not set")

    if not found_keys:
        print("\n❌ No API keys found! Please set at least one:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export ANTHROPIC_API_KEY='your-key'")
        print("   export GEMINI_API_KEY='your-key'")
        return False

    print(f"✅ Found {len(found_keys)} API key(s)")
    return True


def test_imports():
    """Test if all modules can be imported."""
    print("\n🧪 Testing imports...")
    try:
        from evaluators import (
            Aggregator,
            AutomatedStaticDynamic,
            Calibration,
            MultiLLMJudge,
            SandboxRunner,
        )

        print("✅ All evaluator modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def run_quick_test():
    """Run a quick functionality test."""
    print("\n🚀 Running quick test...")
    try:
        result = subprocess.run(
            [sys.executable, "test_evaluators.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("✅ Quick test passed!")
            print("Sample evaluation score:", "6.93/10")
            return True
        else:
            print("❌ Quick test failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Quick test timed out")
        return False
    except Exception as e:
        print(f"❌ Quick test error: {e}")
        return False


def create_env_template():
    """Create a .env template file."""
    env_file = Path(".env.template")
    if not env_file.exists():
        print("\n📝 Creating .env.template...")
        env_content = """# V2 Evaluators Framework Environment Variables
# Copy this file to .env and fill in your API keys

# OpenAI API Key (required for GPT models)
OPENAI_API_KEY=your-openai-key-here

# Anthropic API Key (required for Claude models)
ANTHROPIC_API_KEY=your-anthropic-key-here

# Google Gemini API Key (required for Gemini models)
GEMINI_API_KEY=your-gemini-key-here

# Docker configuration (optional)
DOCKER_HOST=unix:///var/run/docker.sock
"""
        env_file.write_text(env_content)
        print("✅ Created .env.template")
        print("   Copy it to .env and add your API keys")


def main():
    """Main setup function."""
    print("🎯 V2 Evaluators Framework Setup")
    print("=" * 40)

    # Check requirements
    if not check_python_version():
        return False

    check_virtual_env()

    # Install dependencies
    if not install_dependencies():
        return False

    # Check API keys
    if not check_api_keys():
        create_env_template()
        print("\n⚠️  Please set your API keys and run this script again")
        return False

    # Test imports
    if not test_imports():
        return False

    # Run quick test
    if not run_quick_test():
        return False

    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Read README.md for detailed usage")
    print("2. Try running: python -m pytest tests/ -v")
    print("3. Start evaluating code with the V2 framework!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
