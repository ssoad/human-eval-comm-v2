"""
Unit tests for Aggregator module.
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.aggregator import Aggregator, EvaluationWeights, ProblemEvaluation
from src.automated_static_dynamic import (
    DynamicTestResults,
    StaticAnalysisResults,
)
from src.multi_llm_judge import JudgeResponse, NormalizedScores
from src.sandbox_runner import ExecutionResult


class TestEvaluationWeights:
    """Test cases for EvaluationWeights dataclass."""

    def test_evaluation_weights_default(self):
        """Test default evaluation weights."""
        weights = EvaluationWeights()

        assert weights.test_pass_rate == 0.25
        assert weights.llm_consensus == 0.20
        assert weights.static_analysis == 0.15
        assert weights.security_score == 0.15
        assert weights.readability == 0.10
        assert weights.resource_efficiency == 0.10
        assert weights.complexity_penalty == 0.05

    def test_evaluation_weights_custom(self):
        """Test custom evaluation weights."""
        weights = EvaluationWeights(
            test_pass_rate=0.5, llm_consensus=0.3, static_analysis=0.2
        )

        assert weights.test_pass_rate == 0.5
        assert weights.llm_consensus == 0.3
        assert weights.static_analysis == 0.2


class TestAggregator:
    """Test cases for Aggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create Aggregator instance."""
        return Aggregator()

    @pytest.fixture
    def sample_test_results(self):
        """Sample dynamic test results."""
        return DynamicTestResults(
            test_passes=8,
            test_failures=2,
            test_errors=0,
            coverage_percentage=85.0,
            execution_time=3.5,
            memory_usage=50.0,
            hypothesis_tests=5,
            hypothesis_failures=0,
        )

    @pytest.fixture
    def sample_static_results(self):
        """Sample static analysis results."""
        return StaticAnalysisResults(
            pylint_score=8.5,
            pylint_issues=[{"type": "warning", "message": "Unused variable"}],
            bandit_issues=[],
            security_score=9.0,
            complexity_metrics={
                "cyclomatic_complexity": 3.0,
                "maintainability_index": 85.0,
                "lines_of_code": 50,
            },
            mypy_errors=[],
            type_coverage=90.0,
        )

    @pytest.fixture
    def sample_llm_scores(self):
        """Sample LLM normalized scores."""
        responses = [
            JudgeResponse(
                score=8.0, confidence=0.9, rationale="Good code", model_name="model1"
            ),
            JudgeResponse(
                score=7.5, confidence=0.8, rationale="Fair code", model_name="model2"
            ),
        ]

        return NormalizedScores(
            mean_score=7.75,
            mean_confidence=0.85,
            score_std=0.25,
            confidence_std=0.05,
            judge_responses=responses,
            consensus_score=7.8,
        )

    @pytest.fixture
    def sample_sandbox_results(self):
        """Sample sandbox execution results."""
        return ExecutionResult(
            success=True,
            stdout="Tests passed",
            stderr="",
            exit_code=0,
            execution_time=2.5,
            memory_used=45.0,
            cpu_time_used=2.0,
            timeout=False,
            killed=False,
        )

    def test_init_default_weights(self, aggregator):
        """Test initialization with default weights."""
        assert aggregator.weights.test_pass_rate == 0.25
        assert aggregator.weights.llm_consensus == 0.20
        assert len(aggregator.evaluation_history) == 0
        assert aggregator.calibration is None

    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        weights = EvaluationWeights(test_pass_rate=0.5, llm_consensus=0.5)
        aggregator = Aggregator(weights=weights)

        assert aggregator.weights.test_pass_rate == 0.5
        assert aggregator.weights.llm_consensus == 0.5

    def test_normalize_weights(self, aggregator):
        """Test weight normalization."""
        # Set weights that don't sum to 1.0
        aggregator.weights.test_pass_rate = 0.5
        aggregator.weights.llm_consensus = 0.5
        aggregator.weights.static_analysis = 0.5

        aggregator._normalize_weights()

        total_weight = sum(
            [
                aggregator.weights.test_pass_rate,
                aggregator.weights.llm_consensus,
                aggregator.weights.static_analysis,
                aggregator.weights.security_score,
                aggregator.weights.readability,
                aggregator.weights.resource_efficiency,
                aggregator.weights.complexity_penalty,
            ]
        )

        assert abs(total_weight - 1.0) < 0.01

    def test_calculate_test_pass_rate_score(self, aggregator, sample_test_results):
        """Test test pass rate score calculation."""
        score = aggregator._calculate_test_pass_rate_score(sample_test_results)

        assert 0 <= score <= 10
        assert score > 5.0  # Should be good with 8/10 passes and 85% coverage

    def test_calculate_test_pass_rate_score_no_tests(self, aggregator):
        """Test test pass rate score with no test results."""
        score = aggregator._calculate_test_pass_rate_score(None)

        assert score == 0.0

    def test_calculate_test_pass_rate_score_all_failures(self, aggregator):
        """Test test pass rate score with all test failures."""
        test_results = DynamicTestResults(
            test_passes=0,
            test_failures=10,
            test_errors=0,
            coverage_percentage=0.0,
            execution_time=1.0,
            memory_usage=10.0,
            hypothesis_tests=0,
            hypothesis_failures=0,
        )

        score = aggregator._calculate_test_pass_rate_score(test_results)

        assert score == 0.0

    def test_calculate_llm_consensus_score(self, aggregator, sample_llm_scores):
        """Test LLM consensus score calculation."""
        score = aggregator._calculate_llm_consensus_score(sample_llm_scores)

        assert 0 <= score <= 10
        assert score > 5.0  # Should be good with high consensus

    def test_calculate_llm_consensus_score_no_scores(self, aggregator):
        """Test LLM consensus score with no LLM scores."""
        score = aggregator._calculate_llm_consensus_score(None)

        assert score == 0.0

    def test_calculate_static_analysis_score(self, aggregator, sample_static_results):
        """Test static analysis score calculation."""
        score = aggregator._calculate_static_analysis_score(sample_static_results)

        assert 0 <= score <= 10
        assert score > 5.0  # Should be good with high pylint score and type coverage

    def test_calculate_security_score(self, aggregator, sample_static_results):
        """Test security score calculation."""
        score = aggregator._calculate_security_score(sample_static_results)

        assert 0 <= score <= 10
        assert score == 9.0  # Should match the security score

    def test_calculate_security_score_with_issues(self, aggregator):
        """Test security score calculation with security issues."""
        static_results = StaticAnalysisResults(
            pylint_score=8.0,
            pylint_issues=[],
            bandit_issues=[
                {
                    "severity": "HIGH",
                    "confidence": "HIGH",
                    "description": "SQL injection",
                    "line": 10,
                },
                {
                    "severity": "MEDIUM",
                    "confidence": "MEDIUM",
                    "description": "Weak crypto",
                    "line": 20,
                },
            ],
            security_score=7.0,
            complexity_metrics={},
            mypy_errors=[],
            type_coverage=0.0,
        )

        score = aggregator._calculate_security_score(static_results)

        assert 0 <= score <= 10
        assert score < 7.0  # Should be penalized for security issues

    def test_calculate_readability_score(self, aggregator, sample_static_results):
        """Test readability score calculation."""
        score = aggregator._calculate_readability_score(sample_static_results)

        assert 0 <= score <= 10
        assert score > 5.0  # Should be good with high maintainability index

    def test_calculate_readability_score_long_file(self, aggregator):
        """Test readability score with long file penalty."""
        static_results = StaticAnalysisResults(
            pylint_score=8.0,
            pylint_issues=[],
            bandit_issues=[],
            security_score=9.0,
            complexity_metrics={
                "maintainability_index": 85.0,
                "lines_of_code": 200,  # Long file
            },
            mypy_errors=[],
            type_coverage=90.0,
        )

        score = aggregator._calculate_readability_score(static_results)

        assert 0 <= score <= 10
        assert score < 8.5  # Should be penalized for long file

    def test_calculate_resource_efficiency_score(
        self, aggregator, sample_sandbox_results
    ):
        """Test resource efficiency score calculation."""
        score = aggregator._calculate_resource_efficiency_score(sample_sandbox_results)

        assert 0 <= score <= 10
        assert score > 5.0  # Should be good with successful execution

    def test_calculate_resource_efficiency_score_timeout(self, aggregator):
        """Test resource efficiency score with timeout."""
        sandbox_results = ExecutionResult(
            success=False,
            stdout="",
            stderr="Timeout",
            exit_code=-1,
            execution_time=60.0,
            memory_used=100.0,
            cpu_time_used=60.0,
            timeout=True,
            killed=False,
        )

        score = aggregator._calculate_resource_efficiency_score(sandbox_results)

        assert 0 <= score <= 10
        assert score < 5.0  # Should be penalized for timeout

    def test_calculate_complexity_penalty(self, aggregator, sample_static_results):
        """Test complexity penalty calculation."""
        penalty = aggregator._calculate_complexity_penalty(sample_static_results)

        assert penalty == 0.0  # Should be no penalty for low complexity (3.0)

    def test_calculate_complexity_penalty_high_complexity(self, aggregator):
        """Test complexity penalty with high complexity."""
        static_results = StaticAnalysisResults(
            pylint_score=8.0,
            pylint_issues=[],
            bandit_issues=[],
            security_score=9.0,
            complexity_metrics={"cyclomatic_complexity": 15.0},
            mypy_errors=[],
            type_coverage=90.0,
        )

        penalty = aggregator._calculate_complexity_penalty(static_results)

        assert penalty > 0.0  # Should have penalty for high complexity

    def test_calculate_composite_score(self, aggregator):
        """Test composite score calculation."""
        score = aggregator._calculate_composite_score(
            test_pass_rate_score=8.0,
            llm_consensus_score=7.5,
            static_analysis_score=8.5,
            security_score=9.0,
            readability_score=8.0,
            resource_efficiency_score=7.0,
            complexity_penalty=0.5,
        )

        assert 0 <= score <= 10
        assert score > 5.0

    def test_calculate_weighted_composite_score(self, aggregator):
        """Test weighted composite score calculation."""
        score = aggregator._calculate_weighted_composite_score(
            test_pass_rate_score=8.0,
            llm_consensus_score=7.5,
            static_analysis_score=8.5,
            security_score=9.0,
            readability_score=8.0,
            resource_efficiency_score=7.0,
            complexity_penalty=0.5,
        )

        assert 0 <= score <= 10
        assert score > 5.0

    def test_calculate_confidence_interval(
        self, aggregator, sample_llm_scores, sample_test_results, sample_static_results
    ):
        """Test confidence interval calculation."""
        lower, upper = aggregator._calculate_confidence_interval(
            sample_llm_scores, sample_test_results, sample_static_results
        )

        assert 0 <= lower <= upper <= 1
        assert upper > lower

    def test_calculate_confidence_interval_no_data(self, aggregator):
        """Test confidence interval calculation with no data."""
        lower, upper = aggregator._calculate_confidence_interval(None, None, None)

        assert lower == 0.0
        assert upper == 0.0

    def test_evaluate_problem(
        self,
        aggregator,
        sample_test_results,
        sample_static_results,
        sample_llm_scores,
        sample_sandbox_results,
    ):
        """Test complete problem evaluation."""
        evaluation = aggregator.evaluate_problem(
            problem_id="test_problem",
            test_results=sample_test_results,
            static_results=sample_static_results,
            llm_scores=sample_llm_scores,
            sandbox_results=sample_sandbox_results,
        )

        assert evaluation.problem_id == "test_problem"
        assert evaluation.test_results == sample_test_results
        assert evaluation.static_results == sample_static_results
        assert evaluation.llm_scores == sample_llm_scores
        assert evaluation.sandbox_results == sample_sandbox_results

        assert 0 <= evaluation.composite_score <= 10
        assert 0 <= evaluation.weighted_composite_score <= 10
        assert evaluation.evaluation_time > 0
        assert len(evaluation.weights_used) == 7
        assert len(evaluation.confidence_interval) == 2

        # Check that evaluation was added to history
        assert len(aggregator.evaluation_history) == 1
        assert aggregator.evaluation_history[0] == evaluation

    def test_export_evaluation_json(
        self, aggregator, sample_test_results, sample_static_results
    ):
        """Test exporting evaluation to JSON."""
        evaluation = aggregator.evaluate_problem(
            problem_id="test_problem",
            test_results=sample_test_results,
            static_results=sample_static_results,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            aggregator.export_evaluation_json(evaluation, output_path)

            assert os.path.exists(output_path)

            with open(output_path, "r") as f:
                data = json.load(f)

            assert data["problem_id"] == "test_problem"
            assert "composite_score" in data
            assert "weighted_composite_score" in data

        finally:
            os.unlink(output_path)

    def test_export_summary_csv(
        self, aggregator, sample_test_results, sample_static_results
    ):
        """Test exporting summary to CSV."""
        # Create multiple evaluations
        for i in range(3):
            aggregator.evaluate_problem(
                problem_id=f"test_problem_{i}",
                test_results=sample_test_results,
                static_results=sample_static_results,
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            aggregator.export_summary_csv(output_path)

            assert os.path.exists(output_path)

            # Check that CSV has expected columns
            import pandas as pd

            df = pd.read_csv(output_path)

            expected_columns = [
                "problem_id",
                "composite_score",
                "weighted_composite_score",
                "test_pass_rate_score",
                "llm_consensus_score",
                "static_analysis_score",
            ]

            for col in expected_columns:
                assert col in df.columns

            assert len(df) == 3

        finally:
            os.unlink(output_path)

    def test_generate_evaluation_report(
        self, aggregator, sample_test_results, sample_static_results
    ):
        """Test generating evaluation report."""
        # Create some evaluations
        for i in range(5):
            aggregator.evaluate_problem(
                problem_id=f"test_problem_{i}",
                test_results=sample_test_results,
                static_results=sample_static_results,
            )

        report = aggregator.generate_evaluation_report()

        assert "summary" in report
        assert "weights_used" in report
        assert "score_distribution" in report
        assert "recent_evaluations" in report

        assert report["summary"]["total_problems"] == 5
        assert "mean_composite_score" in report["summary"]
        assert "std_composite_score" in report["summary"]

        score_dist = report["score_distribution"]
        assert "excellent" in score_dist
        assert "good" in score_dist
        assert "fair" in score_dist
        assert "poor" in score_dist

    def test_generate_evaluation_report_no_history(self, aggregator):
        """Test generating report with no evaluation history."""
        report = aggregator.generate_evaluation_report()

        assert "error" in report
        assert report["error"] == "No evaluation history available"

    def test_update_weights(self, aggregator):
        """Test updating evaluation weights."""
        new_weights = EvaluationWeights(test_pass_rate=0.5, llm_consensus=0.5)

        aggregator.update_weights(new_weights)

        assert aggregator.weights.test_pass_rate == 0.5
        assert aggregator.weights.llm_consensus == 0.5

    def test_get_evaluation_history(
        self, aggregator, sample_test_results, sample_static_results
    ):
        """Test getting evaluation history."""
        evaluation = aggregator.evaluate_problem(
            problem_id="test_problem",
            test_results=sample_test_results,
            static_results=sample_static_results,
        )

        history = aggregator.get_evaluation_history()

        assert len(history) == 1
        assert history[0] == evaluation
        assert history is not aggregator.evaluation_history  # Should be a copy

    def test_clear_history(
        self, aggregator, sample_test_results, sample_static_results
    ):
        """Test clearing evaluation history."""
        aggregator.evaluate_problem(
            problem_id="test_problem",
            test_results=sample_test_results,
            static_results=sample_static_results,
        )

        assert len(aggregator.evaluation_history) == 1

        aggregator.clear_history()

        assert len(aggregator.evaluation_history) == 0
