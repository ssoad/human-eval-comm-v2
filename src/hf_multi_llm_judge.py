"""
HFMultiLLMJudge: Multi-LLM Judge for HuggingFace models

This class supports evaluation using HuggingFace-hosted models.
All models used are consistent and available everywhere in the pipeline.
"""

import asyncio
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class JudgeResponse:
    score: float
    confidence: float
    rationale: str
    model_name: str


@dataclass
class NormalizedScores:
    mean_score: float
    mean_confidence: float
    score_std: float
    confidence_std: float
    judge_responses: List[JudgeResponse]
    consensus_score: float


class HFMultiLLMJudge:
    """
    Manages multiple HuggingFace LLM judges for scoring code outputs.
    All models must be available and consistent across the pipeline.
    """

    def __init__(self, model_names: List[str], evaluation_prompt: str):
        self.model_names = model_names
        self.evaluation_prompt = evaluation_prompt
        logger.info(
            "Initialized HFMultiLLMJudge with models: %s", self.model_names
        )

    async def _call_hf_model(
        self, model_name: str, prompt: str
    ) -> Optional[JudgeResponse]:
        """
        Call a HuggingFace model via Inference API (stub: replace with actual API call).
        """
        # TODO: Implement actual HuggingFace Inference API call here
        # For now, return a dummy response
        logger.info(
            "Calling HuggingFace model: %s", model_name
        )
        return JudgeResponse(
            score=8.0,
            confidence=0.9,
            rationale=f"Dummy rationale for {model_name}",
            model_name=model_name,
        )

    async def evaluate_code(
        self, code: str, problem: str, expected: str = ""
    ) -> NormalizedScores:
        prompt = self.evaluation_prompt.format(
            code=code, problem=problem, expected=expected
        )
        tasks = [
            self._call_hf_model(model_name, prompt)
            for model_name in self.model_names
        ]
        responses = await asyncio.gather(*tasks)
        valid_responses = [
            r for r in responses
            if isinstance(r, JudgeResponse) and r is not None
        ]
        if not valid_responses:
            return self._empty_normalized_scores()
        return self._normalize_scores(valid_responses)

    def _normalize_scores(
        self, responses: List[JudgeResponse]
    ) -> NormalizedScores:
        scores = [r.score for r in responses]
        confidences = [r.confidence for r in responses]
        mean_score = sum(scores) / len(scores)
        mean_confidence = sum(confidences) / len(confidences)
        score_std = (
            sum((s - mean_score) ** 2 for s in scores) / len(scores)
        ) ** 0.5
        confidence_std = (
            sum((c - mean_confidence) ** 2 for c in confidences)
            / len(confidences)
        ) ** 0.5
        total_weight = sum(confidences)
        if total_weight > 0:
            consensus_score = sum(
                r.score * r.confidence for r in responses
            ) / total_weight
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
        # Use the same models for question evaluation
        return asyncio.run(
            self.evaluate_code(questions, problem, original_problem)
        )
