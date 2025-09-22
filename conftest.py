"""
Pytest configuration and shared fixtures.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the test session."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_config_file():
    """Create a mock configuration file for testing."""
    config_data = {
        "judge_models": [
            {
                "name": "test-model-1",
                "model": "gpt-3.5-turbo",
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "api_key": "test-key-1",
            },
            {
                "name": "test-model-2",
                "model": "claude-3-sonnet",
                "endpoint": "https://api.anthropic.com/v1/messages",
                "api_key": "test-key-2",
            },
        ],
        "evaluation_prompt": "Test evaluation prompt",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        import yaml

        yaml.dump(config_data, f)
        config_path = f.name

    yield config_path

    # Cleanup
    try:
        os.unlink(config_path)
    except FileNotFoundError:
        pass
