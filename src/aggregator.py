"""
Aggregator Module

Combines multiple evaluation metrics into a single composite score:
- Test pass rate
- Multi-LLM consensus score (calibrated)
- Static/security score
- Readability/complexity metrics
- Resource efficiency

Outputs per-problem JSON + CSV summarization with configurable weights.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .automated_static_dynamic import DynamicTestResults, StaticAnalysisResults
from .calibration import Calibration
from .multi_llm_judge import NormalizedScores
from .sandbox_runner import ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationWeights:
    """Configurable weights for different evaluation metrics."""

    test_pass_rate: float = 0.25
    llm_consensus: float = 0.20
    static_analysis: float = 0.15
    security_score: float = 0.15
    readability: float = 0.10
    resource_efficiency: float = 0.10
    complexity_penalty: float = 0.05


@dataclass
class ProblemEvaluation:
    """Complete evaluation results for a single problem."""

    problem_id: str
    timestamp: str

    # Raw metrics
    test_results: Optional[DynamicTestResults]
    static_results: Optional[StaticAnalysisResults]
    llm_scores: Optional[NormalizedScores]
    sandbox_results: Optional[ExecutionResult]

    # Composite scores
    test_pass_rate_score: float
    llm_consensus_score: float
    static_analysis_score: float
    security_score: float
    readability_score: float
    resource_efficiency_score: float
    complexity_penalty: float

    # Final scores
    composite_score: float
    weighted_composite_score: float

    # Metadata
    evaluation_time: float
    weights_used: Dict[str, float]
    confidence_interval: Tuple[float, float]


class Aggregator:
    """Combines multiple evaluation metrics into composite scores."""

    def __init__(
        self,
        weights: Optional[EvaluationWeights] = None,
        calibration: Optional[Calibration] = None,
    ):
        """Initialize aggregator with evaluation weights."""
        self.weights = weights or EvaluationWeights()
        self.calibration = calibration
        self.evaluation_history: List[ProblemEvaluation] = []

        # Validate weights sum to 1.0
        total_weight = sum(asdict(self.weights).values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to sum to 1.0."""
        weight_dict = asdict(self.weights)
        total_weight = sum(weight_dict.values())

        for key, value in weight_dict.items():
            setattr(self.weights, key, value / total_weight)

    def evaluate_problem(
        self,
        problem_id: str,
        test_results: Optional[DynamicTestResults] = None,
        static_results: Optional[StaticAnalysisResults] = None,
        llm_scores: Optional[NormalizedScores] = None,
        sandbox_results: Optional[ExecutionResult] = None,
    ) -> ProblemEvaluation:
        """Evaluate a single problem and compute composite scores."""
        start_time = datetime.now()

        # Calculate individual metric scores
        test_pass_rate_score = self._calculate_test_pass_rate_score(test_results)
        llm_consensus_score = self._calculate_llm_consensus_score(llm_scores)
        static_analysis_score = self._calculate_static_analysis_score(static_results)
        security_score = self._calculate_security_score(static_results)
        readability_score = self._calculate_readability_score(static_results)
        resource_efficiency_score = self._calculate_resource_efficiency_score(
            sandbox_results
        )
        complexity_penalty = self._calculate_complexity_penalty(static_results)

        # Calculate composite scores
        composite_score = self._calculate_composite_score(
            test_pass_rate_score,
            llm_consensus_score,
            static_analysis_score,
            security_score,
            readability_score,
            resource_efficiency_score,
            complexity_penalty,
        )

        weighted_composite_score = self._calculate_weighted_composite_score(
            test_pass_rate_score,
            llm_consensus_score,
            static_analysis_score,
            security_score,
            readability_score,
            resource_efficiency_score,
            complexity_penalty,
        )

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            llm_scores, test_results, static_results
        )

        evaluation_time = (datetime.now() - start_time).total_seconds()

        evaluation = ProblemEvaluation(
            problem_id=problem_id,
            timestamp=start_time.isoformat(),
            test_results=test_results,
            static_results=static_results,
            llm_scores=llm_scores,
            sandbox_results=sandbox_results,
            test_pass_rate_score=test_pass_rate_score,
            llm_consensus_score=llm_consensus_score,
            static_analysis_score=static_analysis_score,
            security_score=security_score,
            readability_score=readability_score,
            resource_efficiency_score=resource_efficiency_score,
            complexity_penalty=complexity_penalty,
            composite_score=composite_score,
            weighted_composite_score=weighted_composite_score,
            evaluation_time=evaluation_time,
            weights_used=asdict(self.weights),
            confidence_interval=confidence_interval,
        )

        self.evaluation_history.append(evaluation)
        return evaluation

    def _calculate_test_pass_rate_score(
        self, test_results: Optional[DynamicTestResults]
    ) -> float:
        """Calculate score based on test pass rate."""
        if test_results is None:
            return 0.0

        total_tests = (
            test_results.test_passes
            + test_results.test_failures
            + test_results.test_errors
        )
        if total_tests == 0:
            return 0.0

        pass_rate = test_results.test_passes / total_tests

        # Bonus for high coverage
        coverage_bonus = min(
            0.2, test_results.coverage_percentage / 500
        )  # Up to 0.2 bonus for 100% coverage

        # Penalty for hypothesis failures
        hypothesis_penalty = 0.0
        if test_results.hypothesis_tests > 0:
            hypothesis_failure_rate = (
                test_results.hypothesis_failures / test_results.hypothesis_tests
            )
            hypothesis_penalty = hypothesis_failure_rate * 0.3

        score = pass_rate + coverage_bonus - hypothesis_penalty
        return max(0.0, min(10.0, score * 10))

    def _calculate_llm_consensus_score(
        self, llm_scores: Optional[NormalizedScores]
    ) -> float:
        """Calculate score based on LLM consensus."""
        if llm_scores is None or not llm_scores.judge_responses:
            return 0.0

        # Use consensus score as base
        base_score = llm_scores.consensus_score

        # Apply calibration if available
        if self.calibration and llm_scores.judge_responses:
            calibrated_scores = []
            for response in llm_scores.judge_responses:
                calibrated_confidence = self.calibration.calibrate_confidence(
                    response.model_name, response.confidence
                )
                # Weight score by calibrated confidence
                weighted_score = response.score * calibrated_confidence
                calibrated_scores.append(weighted_score)

            if calibrated_scores:
                base_score = sum(calibrated_scores) / len(calibrated_scores)

        # Bonus for high confidence
        confidence_bonus = min(0.5, llm_scores.mean_confidence * 0.5)

        # Penalty for high score variance (low consensus)
        variance_penalty = min(1.0, llm_scores.score_std / 2.0)

        score = base_score + confidence_bonus - variance_penalty
        return max(0.0, min(10.0, score))

    def _calculate_static_analysis_score(
        self, static_results: Optional[StaticAnalysisResults]
    ) -> float:
        """Calculate score based on static analysis."""
        if static_results is None:
            return 0.0

        # Base score from pylint
        base_score = static_results.pylint_score

        # Bonus for type coverage
        type_bonus = min(1.0, static_results.type_coverage / 100)

        # Penalty for mypy errors
        mypy_penalty = min(2.0, len(static_results.mypy_errors) * 0.2)

        score = base_score + type_bonus - mypy_penalty
        return max(0.0, min(10.0, score))

    def _calculate_security_score(
        self, static_results: Optional[StaticAnalysisResults]
    ) -> float:
        """Calculate score based on security analysis."""
        if static_results is None:
            return 0.0

        # Base security score from bandit
        base_score = static_results.security_score

        # Penalty for high-severity security issues
        high_severity_issues = sum(
            1
            for issue in static_results.bandit_issues
            if issue.get("severity") == "HIGH"
        )
        security_penalty = min(3.0, high_severity_issues * 0.5)

        score = base_score - security_penalty
        return max(0.0, min(10.0, score))

    def _calculate_readability_score(
        self, static_results: Optional[StaticAnalysisResults]
    ) -> float:
        """Calculate score based on code readability and maintainability."""
        if static_results is None:
            return 0.0

        # Base score from maintainability index
        mi_score = static_results.complexity_metrics.get("maintainability_index", 0)

        # Convert MI to 0-10 scale (MI typically ranges from 0-100)
        base_score = (mi_score / 100) * 10 if mi_score > 0 else 5.0

        # Penalty for very long files
        loc = static_results.complexity_metrics.get("lines_of_code", 0)
        if loc > 100:
            length_penalty = min(2.0, (loc - 100) / 50)
            base_score -= length_penalty

        return max(0.0, min(10.0, base_score))

    def _calculate_resource_efficiency_score(
        self, sandbox_results: Optional[ExecutionResult]
    ) -> float:
        """Calculate score based on resource efficiency."""
        if sandbox_results is None:
            return 5.0  # Neutral score if no sandbox results

        base_score = 10.0

        # Penalty for timeout
        if sandbox_results.timeout:
            base_score -= 3.0

        # Penalty for being killed
        if sandbox_results.killed:
            base_score -= 2.0

        # Penalty for high memory usage
        if sandbox_results.memory_used > 50:  # MB
            memory_penalty = min(2.0, (sandbox_results.memory_used - 50) / 25)
            base_score -= memory_penalty

        # Penalty for long execution time
        if sandbox_results.execution_time > 5.0:  # seconds
            time_penalty = min(2.0, (sandbox_results.execution_time - 5) / 10)
            base_score -= time_penalty

        return max(0.0, min(10.0, base_score))

    def _calculate_complexity_penalty(
        self, static_results: Optional[StaticAnalysisResults]
    ) -> float:
        """Calculate penalty for high complexity."""
        if static_results is None:
            return 0.0

        complexity = static_results.complexity_metrics.get("cyclomatic_complexity", 0)

        if complexity <= 5:
            return 0.0
        elif complexity <= 10:
            return (complexity - 5) * 0.2
        else:
            return 1.0 + (complexity - 10) * 0.3

    def _calculate_composite_score(
        self,
        test_pass_rate_score: float,
        llm_consensus_score: float,
        static_analysis_score: float,
        security_score: float,
        readability_score: float,
        resource_efficiency_score: float,
        complexity_penalty: float,
    ) -> float:
        """Calculate unweighted composite score."""
        scores = [
            test_pass_rate_score,
            llm_consensus_score,
            static_analysis_score,
            security_score,
            readability_score,
            resource_efficiency_score,
        ]

        # Simple average with complexity penalty
        composite = sum(scores) / len(scores) - complexity_penalty
        return max(0.0, min(10.0, composite))

    def _calculate_weighted_composite_score(
        self,
        test_pass_rate_score: float,
        llm_consensus_score: float,
        static_analysis_score: float,
        security_score: float,
        readability_score: float,
        resource_efficiency_score: float,
        complexity_penalty: float,
    ) -> float:
        """Calculate weighted composite score."""
        weighted_score = (
            self.weights.test_pass_rate * test_pass_rate_score
            + self.weights.llm_consensus * llm_consensus_score
            + self.weights.static_analysis * static_analysis_score
            + self.weights.security_score * security_score
            + self.weights.readability * readability_score
            + self.weights.resource_efficiency * resource_efficiency_score
            - self.weights.complexity_penalty * complexity_penalty
        )

        return max(0.0, min(10.0, weighted_score))

    def _calculate_confidence_interval(
        self,
        llm_scores: Optional[NormalizedScores],
        test_results: Optional[DynamicTestResults],
        static_results: Optional[StaticAnalysisResults],
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the evaluation."""
        confidence_factors = []

        # LLM confidence
        if llm_scores and llm_scores.judge_responses:
            confidence_factors.append(llm_scores.mean_confidence)

        # Test coverage confidence
        if test_results and test_results.test_passes > 0:
            total_tests = (
                test_results.test_passes
                + test_results.test_failures
                + test_results.test_errors
            )
            test_confidence = test_results.test_passes / total_tests
            coverage_confidence = test_results.coverage_percentage / 100
            confidence_factors.append((test_confidence + coverage_confidence) / 2)

        # Static analysis confidence (based on completeness)
        if static_results:
            completeness_score = 0.0
            if static_results.pylint_score > 0:
                completeness_score += 0.25
            if static_results.security_score > 0:
                completeness_score += 0.25
            if static_results.complexity_metrics:
                completeness_score += 0.25
            if static_results.type_coverage > 0:
                completeness_score += 0.25
            confidence_factors.append(completeness_score)

        if not confidence_factors:
            return (0.0, 0.0)

        mean_confidence = sum(confidence_factors) / len(confidence_factors)
        std_confidence = (
            np.std(confidence_factors) if len(confidence_factors) > 1 else 0.0
        )

        # 95% confidence interval
        margin = 1.96 * std_confidence
        lower = max(0.0, mean_confidence - margin)
        upper = min(1.0, mean_confidence + margin)

        return (lower, upper)

    def export_evaluation_json(self, evaluation: ProblemEvaluation, output_path: str):
        """Export single evaluation to JSON."""
        try:
            with open(output_path, "w") as f:
                json.dump(asdict(evaluation), f, indent=2, default=str)
            logger.info(f"Evaluation exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export evaluation: {e}")

    def export_summary_csv(self, output_path: str):
        """Export evaluation summary to CSV."""
        try:
            summary_data = []
            for evaluation in self.evaluation_history:
                summary_data.append(
                    {
                        "problem_id": evaluation.problem_id,
                        "timestamp": evaluation.timestamp,
                        "test_pass_rate_score": evaluation.test_pass_rate_score,
                        "llm_consensus_score": evaluation.llm_consensus_score,
                        "static_analysis_score": evaluation.static_analysis_score,
                        "security_score": evaluation.security_score,
                        "readability_score": evaluation.readability_score,
                        "resource_efficiency_score": evaluation.resource_efficiency_score,
                        "complexity_penalty": evaluation.complexity_penalty,
                        "composite_score": evaluation.composite_score,
                        "weighted_composite_score": evaluation.weighted_composite_score,
                        "confidence_lower": evaluation.confidence_interval[0],
                        "confidence_upper": evaluation.confidence_interval[1],
                        "evaluation_time": evaluation.evaluation_time,
                    }
                )

            df = pd.DataFrame(summary_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Summary exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export summary: {e}")

    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not self.evaluation_history:
            return {"error": "No evaluation history available"}

        # Calculate statistics
        composite_scores = [e.composite_score for e in self.evaluation_history]
        weighted_scores = [e.weighted_composite_score for e in self.evaluation_history]

        report = {
            "summary": {
                "total_problems": len(self.evaluation_history),
                "mean_composite_score": np.mean(composite_scores),
                "std_composite_score": np.std(composite_scores),
                "mean_weighted_score": np.mean(weighted_scores),
                "std_weighted_score": np.std(weighted_scores),
                "score_range": (np.min(composite_scores), np.max(composite_scores)),
            },
            "weights_used": asdict(self.weights),
            "score_distribution": {
                "excellent": len([s for s in composite_scores if s >= 8.0]),
                "good": len([s for s in composite_scores if 6.0 <= s < 8.0]),
                "fair": len([s for s in composite_scores if 4.0 <= s < 6.0]),
                "poor": len([s for s in composite_scores if s < 4.0]),
            },
            "recent_evaluations": [
                {
                    "problem_id": e.problem_id,
                    "composite_score": e.composite_score,
                    "weighted_score": e.weighted_composite_score,
                    "timestamp": e.timestamp,
                }
                for e in self.evaluation_history[-10:]  # Last 10 evaluations
            ],
        }

        return report

    def update_weights(self, new_weights: EvaluationWeights):
        """Update evaluation weights."""
        self.weights = new_weights
        self._normalize_weights()
        logger.info("Evaluation weights updated")

    def get_evaluation_history(self) -> List[ProblemEvaluation]:
        """Get all evaluation history."""
        return self.evaluation_history.copy()

    def clear_history(self):
        """Clear evaluation history."""
        self.evaluation_history.clear()
        logger.info("Evaluation history cleared")
