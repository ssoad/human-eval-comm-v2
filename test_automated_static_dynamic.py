"""
Unit tests for AutomatedStaticDynamic module.
"""

import json
import os
import tempfile
from unittest.mock import Mock, mock_open, patch

import pytest

from evaluators.automated_static_dynamic import (
    AutomatedStaticDynamic,
    DynamicTestResults,
    StaticAnalysisResults,
)


class TestAutomatedStaticDynamic:
    """Test cases for AutomatedStaticDynamic."""

    @pytest.fixture
    def analyzer(self):
        """Create AutomatedStaticDynamic instance."""
        return AutomatedStaticDynamic()

    @pytest.fixture
    def sample_code(self):
        """Sample Python code for testing."""
        return """
def fibonacci(n):
    '''Calculate fibonacci number.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
"""

    @pytest.fixture
    def sample_test_code(self):
        """Sample test code."""
        return """
import pytest
from solution import fibonacci

def test_fibonacci_basic():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5

def test_fibonacci_edge_cases():
    assert fibonacci(2) == 1
    assert fibonacci(10) == 55
"""

    def test_init(self):
        """Test initialization."""
        analyzer = AutomatedStaticDynamic()
        assert analyzer.temp_dir == tempfile.gettempdir()
        assert isinstance(analyzer.analysis_cache, dict)

        custom_temp = "/tmp/custom"
        analyzer = AutomatedStaticDynamic(custom_temp)
        assert analyzer.temp_dir == custom_temp

    def test_empty_static_results(self, analyzer):
        """Test empty static analysis results."""
        result = analyzer._empty_static_results()

        assert result.pylint_score == 0.0
        assert len(result.pylint_issues) == 0
        assert len(result.bandit_issues) == 0
        assert result.security_score == 0.0
        assert len(result.complexity_metrics) == 0
        assert len(result.mypy_errors) == 0
        assert result.type_coverage == 0.0

    def test_empty_dynamic_results(self, analyzer):
        """Test empty dynamic test results."""
        result = analyzer._empty_dynamic_results()

        assert result.test_passes == 0
        assert result.test_failures == 0
        assert result.test_errors == 1
        assert result.coverage_percentage == 0.0
        assert result.execution_time == 0.0
        assert result.memory_usage == 0.0
        assert result.hypothesis_tests == 0
        assert result.hypothesis_failures == 0

    @patch("subprocess.run")
    def test_run_pylint_success(self, mock_run, analyzer):
        """Test successful pylint execution."""
        mock_output = json.dumps(
            [{"type": "warning", "message": "Unused variable", "line": 5, "column": 10}]
        )

        mock_run.return_value.stdout = mock_output
        mock_run.return_value.returncode = 0

        result = analyzer._run_pylint("/tmp/test.py")

        assert result["score"] > 0
        assert len(result["issues"]) == 1
        assert result["issues"][0]["type"] == "warning"

    @patch("subprocess.run")
    def test_run_pylint_failure(self, mock_run, analyzer):
        """Test pylint execution failure."""
        mock_run.side_effect = FileNotFoundError("pylint not found")

        result = analyzer._run_pylint("/tmp/test.py")

        assert result["score"] == 5.0
        assert len(result["issues"]) == 0

    @patch("subprocess.run")
    def test_run_bandit_success(self, mock_run, analyzer):
        """Test successful bandit execution."""
        mock_output = json.dumps(
            {
                "results": [
                    {
                        "issue_severity": "MEDIUM",
                        "issue_confidence": "HIGH",
                        "issue_text": "Potential security issue",
                        "line_number": 10,
                    }
                ]
            }
        )

        mock_run.return_value.stdout = mock_output
        mock_run.return_value.returncode = 0

        result = analyzer._run_bandit("/tmp/test.py")

        assert result["security_score"] < 10.0
        assert len(result["issues"]) == 1
        assert result["issues"][0]["severity"] == "MEDIUM"

    @patch("subprocess.run")
    def test_run_bandit_failure(self, mock_run, analyzer):
        """Test bandit execution failure."""
        mock_run.side_effect = FileNotFoundError("bandit not found")

        result = analyzer._run_bandit("/tmp/test.py")

        assert result["security_score"] == 5.0
        assert len(result["issues"]) == 0

    @patch("subprocess.run")
    def test_run_radon_success(self, mock_run, analyzer):
        """Test successful radon execution."""
        # Mock cyclomatic complexity output
        cc_output = json.dumps({"/tmp/test.py": [{"complexity": 3}, {"complexity": 5}]})

        # Mock maintainability index output
        mi_output = json.dumps({"/tmp/test.py": [{"mi": 85.5}, {"mi": 72.3}]})

        mock_run.side_effect = [
            Mock(stdout=cc_output, returncode=0),
            Mock(stdout=mi_output, returncode=0),
        ]

        with patch("builtins.open", mock_open(read_data="def test():\n    pass\n")):
            result = analyzer._run_radon("/tmp/test.py")

        assert result["cyclomatic_complexity"] == 4.0
        assert result["maintainability_index"] == 78.9
        assert result["lines_of_code"] == 2

    @patch("subprocess.run")
    def test_run_radon_failure(self, mock_run, analyzer):
        """Test radon execution failure."""
        mock_run.side_effect = FileNotFoundError("radon not found")

        result = analyzer._run_radon("/tmp/test.py")

        assert result["cyclomatic_complexity"] == 0.0
        assert result["maintainability_index"] == 0.0
        assert result["lines_of_code"] == 0

    @patch("subprocess.run")
    def test_run_mypy_success(self, mock_run, analyzer):
        """Test successful mypy execution."""
        mock_run.return_value.stdout = "error: No return type annotation"
        mock_run.return_value.returncode = 0

        # Mock JSON report file
        mock_report = {"summary": {"total_lines": 100, "typed_lines": 80}}

        with patch("builtins.open", mock_open(read_data=json.dumps(mock_report))):
            result = analyzer._run_mypy("/tmp/test.py")

        assert len(result["errors"]) == 1
        assert result["coverage"] == 80.0

    @patch("subprocess.run")
    def test_run_mypy_failure(self, mock_run, analyzer):
        """Test mypy execution failure."""
        mock_run.side_effect = FileNotFoundError("mypy not found")

        result = analyzer._run_mypy("/tmp/test.py")

        assert len(result["errors"]) == 0
        assert result["coverage"] == 0.0

    def test_analyze_code_no_test_code(self, analyzer, sample_code):
        """Test code analysis without test code."""
        with patch.object(analyzer, "_run_static_analysis") as mock_static:
            mock_static.return_value = analyzer._empty_static_results()

            static_result, dynamic_result = analyzer.analyze_code(
                sample_code, "", "test_problem"
            )

            assert isinstance(static_result, StaticAnalysisResults)
            assert isinstance(dynamic_result, DynamicTestResults)
            assert dynamic_result.test_errors == 1  # Empty dynamic result

    def test_analyze_code_with_test_code(self, analyzer, sample_code, sample_test_code):
        """Test code analysis with test code."""
        with patch.object(
            analyzer, "_run_static_analysis"
        ) as mock_static, patch.object(
            analyzer, "_run_dynamic_testing"
        ) as mock_dynamic:

            mock_static.return_value = analyzer._empty_static_results()
            mock_dynamic.return_value = analyzer._empty_dynamic_results()

            static_result, dynamic_result = analyzer.analyze_code(
                sample_code, sample_test_code, "test_problem"
            )

            assert isinstance(static_result, StaticAnalysisResults)
            assert isinstance(dynamic_result, DynamicTestResults)
            mock_dynamic.assert_called_once()

    def test_calculate_overall_score(self, analyzer):
        """Test overall score calculation."""
        static_results = StaticAnalysisResults(
            pylint_score=8.0,
            pylint_issues=[],
            bandit_issues=[],
            security_score=9.0,
            complexity_metrics={
                "cyclomatic_complexity": 3.0,
                "maintainability_index": 85.0,
            },
            mypy_errors=[],
            type_coverage=90.0,
        )

        dynamic_results = DynamicTestResults(
            test_passes=8,
            test_failures=2,
            test_errors=0,
            coverage_percentage=85.0,
            execution_time=2.5,
            memory_usage=50.0,
            hypothesis_tests=5,
            hypothesis_failures=0,
        )

        score = analyzer._calculate_overall_score(static_results, dynamic_results)

        assert 0 <= score <= 10
        assert score > 5.0  # Should be a decent score

    def test_generate_report(self, analyzer):
        """Test report generation."""
        static_results = StaticAnalysisResults(
            pylint_score=8.0,
            pylint_issues=[{"type": "warning", "message": "test"}],
            bandit_issues=[],
            security_score=9.0,
            complexity_metrics={"cyclomatic_complexity": 3.0},
            mypy_errors=[],
            type_coverage=90.0,
        )

        dynamic_results = DynamicTestResults(
            test_passes=8,
            test_failures=2,
            test_errors=0,
            coverage_percentage=85.0,
            execution_time=2.5,
            memory_usage=50.0,
            hypothesis_tests=5,
            hypothesis_failures=0,
        )

        report = analyzer.generate_report(static_results, dynamic_results)

        assert "static_analysis" in report
        assert "dynamic_testing" in report
        assert "overall_score" in report
        assert report["static_analysis"]["pylint_score"] == 8.0
        assert report["dynamic_testing"]["test_passes"] == 8
        assert 0 <= report["overall_score"] <= 10


class TestStaticAnalysisResults:
    """Test cases for StaticAnalysisResults dataclass."""

    def test_static_analysis_results_creation(self):
        """Test StaticAnalysisResults creation."""
        result = StaticAnalysisResults(
            pylint_score=8.5,
            pylint_issues=[{"type": "warning"}],
            bandit_issues=[],
            security_score=9.0,
            complexity_metrics={"cyclomatic_complexity": 3.0},
            mypy_errors=[],
            type_coverage=85.0,
        )

        assert result.pylint_score == 8.5
        assert len(result.pylint_issues) == 1
        assert result.security_score == 9.0
        assert result.complexity_metrics["cyclomatic_complexity"] == 3.0
        assert result.type_coverage == 85.0


class TestDynamicTestResults:
    """Test cases for DynamicTestResults dataclass."""

    def test_dynamic_test_results_creation(self):
        """Test DynamicTestResults creation."""
        result = DynamicTestResults(
            test_passes=10,
            test_failures=2,
            test_errors=0,
            coverage_percentage=85.0,
            execution_time=3.5,
            memory_usage=45.0,
            hypothesis_tests=5,
            hypothesis_failures=1,
        )

        assert result.test_passes == 10
        assert result.test_failures == 2
        assert result.test_errors == 0
        assert result.coverage_percentage == 85.0
        assert result.execution_time == 3.5
        assert result.memory_usage == 45.0
        assert result.hypothesis_tests == 5
        assert result.hypothesis_failures == 1
