"""
Multi-LLM Judge Module

Calls multiple LLM judges to score outputs/questions/explanations with structured
JSON responses (score, confidence, short rationale). Normalizes outputs to numeric
scores and confidences.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import yaml
from dotenv import load_dotenv

# Load .env variables at startup
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class JudgeResponse:
    """Structured response from an LLM judge."""

    score: float
    confidence: float
    rationale: str
    model_name: str


@dataclass
class NormalizedScores:
    """Normalized scores from multiple judges."""

    mean_score: float
    mean_confidence: float
    score_std: float
    confidence_std: float
    judge_responses: List[JudgeResponse]
    consensus_score: float  # weighted by confidence


class MultiLLMJudge:
    """Manages multiple LLM judges for scoring code outputs."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        logger.info(f"Loading config from: {config_path}")
        self.config = self._load_config(config_path)
        self.judge_models = self.config.get("judge_models", [])
        self.evaluation_prompt = self.config.get(
            "evaluation_prompt", self._default_evaluation_prompt()
        )
        logger.info(f"Initialized with {len(self.judge_models)} judge models")

    def _create_ssl_session(self) -> aiohttp.ClientSession:
        """Create aiohttp session with SSL bypass for development."""
        connector = aiohttp.TCPConnector(verify_ssl=False)
        return aiohttp.ClientSession(connector=connector)

    def _detect_api_provider(self, model_config: Dict[str, Any]) -> str:
        """Detect which API provider to use based on endpoint."""
        endpoint = model_config.get("endpoint", "")
        if "generativelanguage.googleapis.com" in endpoint:
            return "gemini"
        elif "anthropic.com" in endpoint:
            return "anthropic"
        elif "openai.com" in endpoint:
            return "openai"
        else:
            return "unknown"

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file with env var substitution."""
        try:
            with open(config_path, "r") as f:
                # Substitute environment variables like ${OPENAI_API_KEY}
                config_text = os.path.expandvars(f.read())
                return yaml.safe_load(config_text)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}

    def _default_evaluation_prompt(self) -> str:
        """Default evaluation prompt for LLM judges."""
        return """
        You are an expert code reviewer. Please evaluate the following code submission and provide a structured response.

        Code to evaluate:
        ```python
        {code}
        ```

        Problem description:
        {problem}

        Expected behavior:
        {expected}

        Please provide your evaluation in the following JSON format:
        {{
            "score": <float between 0.0 and 10.0>,
            "confidence": <float between 0.0 and 1.0>,
            "rationale": "<brief explanation of your scoring>"
        }}

        Consider:
        - Correctness and functionality
        - Code quality and readability
        - Efficiency and best practices
        - Edge case handling
        """

    async def _call_llm_judge(
        self, session: aiohttp.ClientSession, model_config: Dict[str, Any], prompt: str
    ) -> Optional[JudgeResponse]:
        """Call a single LLM judge asynchronously."""
        try:
            provider = self._detect_api_provider(model_config)
            logger.debug(f"Calling {provider} API for model {model_config.get('name')}")

            if provider == "gemini":
                return await self._call_gemini_api(model_config, prompt)
            elif provider == "anthropic":
                return await self._call_anthropic_api(model_config, prompt)
            elif provider == "openai":
                return await self._call_openai_api(model_config, prompt)
            else:
                logger.error(f"Unknown API provider for model {model_config.get('name')}")
                return None

        except Exception as e:
            logger.error(f"Exception calling {model_config.get('name')}: {e}")
            return None

    async def _call_gemini_api(
        self, model_config: Dict[str, Any], prompt: str
    ) -> Optional[JudgeResponse]:
        """Call Google Gemini API."""
        endpoint = model_config.get("endpoint")
        if not endpoint.endswith(":generateContent"):
            endpoint = endpoint.rstrip("/") + f"/models/{model_config['model']}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": model_config.get('api_key', ''),
        }
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        async with self._create_ssl_session() as session:
            async with session.post(
                endpoint, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    logger.debug(f"Gemini response received for {model_config['name']}: {content[:200]}...")
                    return self._parse_judge_response(content, model_config["name"])
                else:
                    logger.error(f"Gemini API error for {model_config['name']}: {response.status}")
                    return None

    async def _call_anthropic_api(
        self, model_config: Dict[str, Any], prompt: str
    ) -> Optional[JudgeResponse]:
        """Call Anthropic Claude API."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": model_config.get('api_key', ''),
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
        }

        async with self._create_ssl_session() as session:
            async with session.post(
                model_config["endpoint"], json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["content"][0]["text"]
                    logger.debug(f"Anthropic response received for {model_config['name']}: {content[:200]}...")
                    return self._parse_judge_response(content, model_config["name"])
                else:
                    logger.error(f"Anthropic API error for {model_config['name']}: {response.status}")
                    return None

    async def _call_openai_api(
        self, model_config: Dict[str, Any], prompt: str
    ) -> Optional[JudgeResponse]:
        """Call OpenAI API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model_config.get('api_key', '')}",
        }
        payload = {
            "model": model_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
        }

        async with self._create_ssl_session() as session:
            async with session.post(
                model_config["endpoint"], json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    logger.debug(f"OpenAI response received for {model_config['name']}: {content[:200]}...")
                    return self._parse_judge_response(content, model_config["name"])
                else:
                    logger.error(f"OpenAI API error for {model_config['name']}: {response.status}")
                    return None

    def _sanitize_json_content(self, content: str) -> str:
        """Clean and sanitize content before JSON parsing."""
        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        # Strip whitespace
        content = content.strip()

        # Remove control characters that can break JSON parsing
        import re

        # Remove unescaped control characters (except \n, \r, \t which are valid in JSON strings)
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)

        # Fix common JSON issues
        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',(\s*[}\]])', r'\1', content)

        return content

    def _parse_judge_response(
        self, content: str, model_name: str
    ) -> Optional[JudgeResponse]:
        """Parse LLM response into structured format."""
        try:
            # Clean and sanitize the content
            cleaned_content = self._sanitize_json_content(content)

            # Try to parse JSON
            data = json.loads(cleaned_content)

            return JudgeResponse(
                score=float(data.get("score", 0)),
                confidence=float(data.get("confidence", 0)),
                rationale=data.get("rationale", ""),
                model_name=model_name,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse response from {model_name}: {e}")
            logger.debug(f"Raw response content: {repr(content[:500])}")
            # Fallback: try to extract numbers from text
            return self._extract_scores_from_text(content, model_name)

    def _extract_scores_from_text(
        self, text: str, model_name: str
    ) -> Optional[JudgeResponse]:
        """Fallback method to extract scores from unstructured text."""
        import re

        # Look for score patterns
        score_patterns = [
            r'score["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"(\d+(?:\.\d+)?)\s*out\s*of\s*10",
        ]

        confidence_patterns = [
            r'confidence["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
            r"(\d+(?:\.\d+)?)\s*%?\s*confident",
        ]

        score = None
        confidence = None

        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                break

        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                confidence = float(match.group(1))
                if confidence > 1:
                    confidence = confidence / 100
                break

        if score is not None:
            return JudgeResponse(
                score=score,
                confidence=confidence or 0.5,
                rationale=text[:200] + "..." if len(text) > 200 else text,
                model_name=model_name,
            )

        return None

    async def evaluate_code(
        self, code: str, problem: str, expected: str = ""
    ) -> NormalizedScores:
        """Evaluate code using multiple LLM judges."""
        logger.info(f"Evaluating code (length: {len(code)}) with {len(self.judge_models)} judges")

        if not self.judge_models:
            logger.warning("No judge models configured")
            return self._empty_normalized_scores()

        prompt = self.evaluation_prompt.format(
            code=code, problem=problem, expected=expected
        )
        logger.debug(f"Evaluation prompt prepared (length: {len(prompt)})")

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._call_llm_judge(session, model_config, prompt)
                for model_config in self.judge_models
            ]
            logger.debug(f"Created {len(tasks)} evaluation tasks")

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug("All evaluation tasks completed")

        # Filter out None and exception responses
        valid_responses = [
            r for r in responses if isinstance(r, JudgeResponse) and r is not None
        ]

        logger.info(f"Received {len(valid_responses)} valid responses from {len(self.judge_models)} judges")

        if not valid_responses:
            logger.error("No valid responses from any judges")
            return self._empty_normalized_scores()

        return self._normalize_scores(valid_responses)

    def _normalize_scores(self, responses: List[JudgeResponse]) -> NormalizedScores:
        """Normalize scores from multiple judges."""
        scores = [r.score for r in responses]
        confidences = [r.confidence for r in responses]

        mean_score = sum(scores) / len(scores)
        mean_confidence = sum(confidences) / len(confidences)

        score_std = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
        confidence_std = (
            sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        ) ** 0.5

        # Weighted consensus score by confidence
        total_weight = sum(confidences)
        if total_weight > 0:
            consensus_score = (
                sum(r.score * r.confidence for r in responses) / total_weight
            )
        else:
            consensus_score = mean_score

        return NormalizedScores(
            mean_score=mean_score,
            mean_confidence=mean_confidence,
            score_std=score_std,
            confidence_std=confidence_std,
            judge_responses=responses,
            consensus_score=consensus_score,
        )

    def _empty_normalized_scores(self) -> NormalizedScores:
        """Return empty scores when evaluation fails."""
        return NormalizedScores(
            mean_score=0.0,
            mean_confidence=0.0,
            score_std=0.0,
            confidence_std=0.0,
            judge_responses=[],
            consensus_score=0.0,
        )

    def evaluate_questions(
        self, questions: str, problem: str, original_problem: str = ""
    ) -> NormalizedScores:
        """Evaluate clarifying questions quality."""
        questions_prompt = f"""
        You are an expert at evaluating the quality of clarifying questions in software development.

        Original Problem:
        {original_problem}

        Modified Problem:
        {problem}

        Clarifying Questions:
        {questions}

        Please evaluate the quality of these questions on a scale of 0-10, considering:
        - How well they identify ambiguities or missing information
        - Whether they help clarify the requirements
        - Their specificity and relevance

        Provide your evaluation in JSON format:
        {{
            "score": <float between 0.0 and 10.0>,
            "confidence": <float between 0.0 and 1.0>,
            "rationale": "<brief explanation of your scoring>"
        }}
        """

        # Use synchronous evaluation for questions
        return asyncio.run(self._evaluate_with_prompt(questions_prompt))

    async def _evaluate_with_prompt(self, prompt: str) -> NormalizedScores:
        """Helper method to evaluate with a custom prompt."""
        if not self.judge_models:
            return self._empty_normalized_scores()

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._call_llm_judge(session, model_config, prompt)
                for model_config in self.judge_models
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = [
            r for r in responses if isinstance(r, JudgeResponse) and r is not None
        ]

        if not valid_responses:
            return self._empty_normalized_scores()

        return self._normalize_scores(valid_responses)
