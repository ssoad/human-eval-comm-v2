"""
Unit tests for MultiLLMJudge module.
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

from evaluators.multi_llm_judge import JudgeResponse, MultiLLMJudge, NormalizedScores


class TestMultiLLMJudge:
    """Test cases for MultiLLMJudge."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = {
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
            ]
        }
        return config

    @pytest.fixture
    def judge_instance(self, mock_config):
        """Create MultiLLMJudge instance with mock config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(mock_config, f)
            config_path = f.name

        judge = MultiLLMJudge(config_path)
        os.unlink(config_path)
        judge.config = mock_config
        return judge

    def test_init_with_config_file(self):
        """Test initialization with config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            config = {"judge_models": []}
            yaml.dump(config, f)
            config_path = f.name

        judge = MultiLLMJudge(config_path)
        assert judge.config == config
        os.unlink(config_path)

    def test_init_without_config_file(self):
        """Test initialization without config file."""
        judge = MultiLLMJudge("nonexistent.yaml")
        assert judge.config == {}
        assert judge.judge_models == []

    def test_parse_judge_response_valid_json(self, judge_instance):
        """Test parsing valid JSON response."""
        response_text = """
        {
            "score": 8.5,
            "confidence": 0.9,
            "rationale": "Good code quality with minor issues"
        }
        """

        result = judge_instance._parse_judge_response(response_text, "test-model")

        assert result is not None
        assert result.score == 8.5
        assert result.confidence == 0.9
        assert result.rationale == "Good code quality with minor issues"
        assert result.model_name == "test-model"

    def test_parse_judge_response_json_with_markdown(self, judge_instance):
        """Test parsing JSON response wrapped in markdown."""
        response_text = """
        ```json
        {
            "score": 7.0,
            "confidence": 0.8,
            "rationale": "Decent implementation"
        }
        ```
        """

        result = judge_instance._parse_judge_response(response_text, "test-model")

        assert result is not None
        assert result.score == 7.0
        assert result.confidence == 0.8

    def test_parse_judge_response_invalid_json(self, judge_instance):
        """Test parsing invalid JSON response."""
        response_text = "This is not valid JSON"

        result = judge_instance._parse_judge_response(response_text, "test-model")

        # Should fallback to text extraction
        assert result is None or isinstance(result, JudgeResponse)

    def test_extract_scores_from_text(self, judge_instance):
        """Test extracting scores from unstructured text."""
        text = "I would rate this code 8.5/10 with 90% confidence."

        result = judge_instance._extract_scores_from_text(text, "test-model")

        assert result is not None
        assert result.score == 8.5
        assert result.confidence == 0.9

    def test_extract_scores_from_text_with_percentage(self, judge_instance):
        """Test extracting scores with percentage confidence."""
        text = "Score: 7.0, Confidence: 85%"

        result = judge_instance._extract_scores_from_text(text, "test-model")

        assert result is not None
        assert result.score == 7.0
        assert result.confidence == 0.85

    def test_normalize_scores(self, judge_instance):
        """Test score normalization."""
        responses = [
            JudgeResponse(
                score=8.0, confidence=0.9, rationale="Good", model_name="model1"
            ),
            JudgeResponse(
                score=7.0, confidence=0.7, rationale="Fair", model_name="model2"
            ),
            JudgeResponse(
                score=9.0, confidence=0.8, rationale="Excellent", model_name="model3"
            ),
        ]

        result = judge_instance._normalize_scores(responses)

        assert result.mean_score == 8.0
        assert result.mean_confidence == 0.8
        assert result.consensus_score > 0
        assert len(result.judge_responses) == 3

    def test_empty_normalized_scores(self, judge_instance):
        """Test empty normalized scores."""
        result = judge_instance._empty_normalized_scores()

        assert result.mean_score == 0.0
        assert result.mean_confidence == 0.0
        assert result.consensus_score == 0.0
        assert len(result.judge_responses) == 0

    @pytest.mark.asyncio
    async def test_evaluate_code_no_models(self):
        """Test evaluation with no models configured."""
        judge = MultiLLMJudge()
        judge.judge_models = []

        result = await judge.evaluate_code("print('hello')", "test problem")

        assert result.mean_score == 0.0
        assert result.mean_confidence == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_code_with_mock_models(self, judge_instance):
        """Test evaluation with mock model responses."""
        # Mock the HTTP session and responses
        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "content": '{"score": 8.0, "confidence": 0.9, "rationale": "Good code"}'
                    }
                }
            ]
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = (
                mock_response
            )

            result = await judge_instance.evaluate_code(
                "print('hello')", "test problem"
            )

            assert result.mean_score > 0
            assert result.mean_confidence > 0

    def test_evaluate_questions(self, judge_instance):
        """Test question evaluation."""
        questions = "What are the input constraints?"
        problem = "Write a function to sort a list"
        original = "Write a function to sort a list of integers"

        # Mock the async evaluation
        with patch.object(judge_instance, "_evaluate_with_prompt") as mock_eval:
            mock_eval.return_value = judge_instance._empty_normalized_scores()

            result = judge_instance.evaluate_questions(questions, problem, original)

            assert result is not None
            mock_eval.assert_called_once()

    def test_default_evaluation_prompt(self, judge_instance):
        """Test default evaluation prompt format."""
        prompt = judge_instance._default_evaluation_prompt()

        assert "{code}" in prompt
        assert "{problem}" in prompt
        assert "{expected}" in prompt
        assert "score" in prompt
        assert "confidence" in prompt
        assert "rationale" in prompt


class TestJudgeResponse:
    """Test cases for JudgeResponse dataclass."""

    def test_judge_response_creation(self):
        """Test JudgeResponse creation."""
        response = JudgeResponse(
            score=8.5,
            confidence=0.9,
            rationale="Good implementation",
            model_name="test-model",
        )

        assert response.score == 8.5
        assert response.confidence == 0.9
        assert response.rationale == "Good implementation"
        assert response.model_name == "test-model"


class TestNormalizedScores:
    """Test cases for NormalizedScores dataclass."""

    def test_normalized_scores_creation(self):
        """Test NormalizedScores creation."""
        responses = [
            JudgeResponse(
                score=8.0, confidence=0.9, rationale="Good", model_name="model1"
            ),
            JudgeResponse(
                score=7.0, confidence=0.7, rationale="Fair", model_name="model2"
            ),
        ]

        scores = NormalizedScores(
            mean_score=7.5,
            mean_confidence=0.8,
            score_std=0.5,
            confidence_std=0.1,
            judge_responses=responses,
            consensus_score=7.7,
        )

        assert scores.mean_score == 7.5
        assert scores.mean_confidence == 0.8
        assert len(scores.judge_responses) == 2
        assert scores.consensus_score == 7.7
