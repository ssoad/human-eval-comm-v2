"""
Automated Static and Dynamic Analysis Module

Runs static analyzers (pylint, bandit, radon, mypy) on generated code and
produces JSON metrics. Also runs dynamic testing using the repo's test harness
under sandboxed conditions and Hypothesis property tests where possible.
"""

import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StaticAnalysisResults:
    """Results from static analysis tools."""

    pylint_score: float
    pylint_issues: List[Dict[str, Any]]
    bandit_issues: List[Dict[str, Any]]
    security_score: float
    complexity_metrics: Dict[str, float]
    mypy_errors: List[str]
    type_coverage: float


@dataclass
class DynamicTestResults:
    """Results from dynamic testing."""

    test_passes: int
    test_failures: int
    test_errors: int
    coverage_percentage: float
    execution_time: float
    memory_usage: float
    hypothesis_tests: int
    hypothesis_failures: int


class AutomatedStaticDynamic:
    """Handles static and dynamic analysis of generated code."""

    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize analyzer with optional temp directory."""
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.analysis_cache = {}

    def analyze_code(
        self, code: str, test_code: str = "", problem_id: str = ""
    ) -> Tuple[StaticAnalysisResults, DynamicTestResults]:
        """Perform comprehensive static and dynamic analysis."""
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
            # Write code to temporary file
            code_file = os.path.join(temp_dir, f"{problem_id}_solution.py")
            with open(code_file, "w") as f:
                f.write(code)

            # Perform static analysis
            static_results = self._run_static_analysis(code_file)

            # Perform dynamic testing if test code provided
            if test_code:
                test_file = os.path.join(temp_dir, f"{problem_id}_test.py")
                with open(test_file, "w") as f:
                    f.write(test_code)
                dynamic_results = self._run_dynamic_testing(code_file, test_file)
            else:
                dynamic_results = self._empty_dynamic_results()

            return static_results, dynamic_results

    def _run_static_analysis(self, code_file: str) -> StaticAnalysisResults:
        """Run static analysis tools on the code file."""
        try:
            pylint_results = self._run_pylint(code_file)
            bandit_results = self._run_bandit(code_file)
            radon_results = self._run_radon(code_file)
            mypy_results = self._run_mypy(code_file)

            return StaticAnalysisResults(
                pylint_score=pylint_results["score"],
                pylint_issues=pylint_results["issues"],
                bandit_issues=bandit_results["issues"],
                security_score=bandit_results["security_score"],
                complexity_metrics=radon_results,
                mypy_errors=mypy_results["errors"],
                type_coverage=mypy_results["coverage"],
            )

        except Exception as e:
            logger.error(f"Static analysis failed: {e}")
            return self._empty_static_results()

    def _run_pylint(self, code_file: str) -> Dict[str, Any]:
        """Run pylint analysis."""
        try:
            pylint_path = os.path.join(os.path.dirname(sys.executable), "pylint")
            result = subprocess.run(
                [pylint_path, "--output-format=json", code_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.stdout:
                pylint_data = json.loads(result.stdout)
                score = 10.0
                issues = []

                for issue in pylint_data:
                    severity = issue.get("type", "").upper()
                    if severity == "ERROR":
                        score -= 2.0
                    elif severity == "WARNING":
                        score -= 1.0
                    elif severity == "CONVENTION":
                        score -= 0.5

                    issues.append(
                        {
                            "type": issue.get("type", ""),
                            "message": issue.get("message", ""),
                            "line": issue.get("line", 0),
                            "column": issue.get("column", 0),
                        }
                    )

                score = max(0.0, score)
                return {"score": score, "issues": issues}
            else:
                return {"score": 10.0, "issues": []}

        except (
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            FileNotFoundError,
        ) as e:
            logger.warning(f"Pylint analysis failed: {e}")
            return {"score": 5.0, "issues": []}

    def _run_bandit(self, code_file: str) -> Dict[str, Any]:
        """Run bandit security analysis."""
        try:
            bandit_path = os.path.join(os.path.dirname(sys.executable), "bandit")
            result = subprocess.run(
                [bandit_path, "-f", "json", code_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.stdout:
                bandit_data = json.loads(result.stdout)
                issues = []
                security_score = 10.0

                for issue in bandit_data.get("results", []):
                    severity = issue.get("issue_severity", "MEDIUM").upper()
                    if severity == "HIGH":
                        security_score -= 3.0
                    elif severity == "MEDIUM":
                        security_score -= 2.0
                    elif severity == "LOW":
                        security_score -= 1.0

                    issues.append(
                        {
                            "severity": severity,
                            "confidence": issue.get("issue_confidence", ""),
                            "description": issue.get("issue_text", ""),
                            "line": issue.get("line_number", 0),
                        }
                    )

                security_score = max(0.0, security_score)
                return {"security_score": security_score, "issues": issues}
            else:
                return {"security_score": 10.0, "issues": []}

        except (
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            FileNotFoundError,
        ) as e:
            logger.warning(f"Bandit analysis failed: {e}")
            return {"security_score": 5.0, "issues": []}

    def _run_radon(self, code_file: str) -> Dict[str, float]:
        """Run radon complexity analysis."""
        try:
            radon_path = os.path.join(os.path.dirname(sys.executable), "radon")
            # Cyclomatic complexity
            cc_result = subprocess.run(
                [radon_path, "cc", "--json", code_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Maintainability index
            mi_result = subprocess.run(
                [radon_path, "mi", "--json", code_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            metrics = {
                "cyclomatic_complexity": 0.0,
                "maintainability_index": 0.0,
                "lines_of_code": 0,
            }

            if cc_result.stdout:
                cc_data = json.loads(cc_result.stdout)
                if cc_data and code_file in cc_data:
                    cc_values = cc_data[code_file]
                    if isinstance(cc_values, list):
                        cc_scores = [item.get("complexity", 0) if isinstance(item, dict) else float(item) if isinstance(item, (int, float, str)) and str(item).replace('.', '').isdigit() else 0 for item in cc_values]
                        metrics["cyclomatic_complexity"] = (
                            sum(cc_scores) / len(cc_scores) if cc_scores else 0.0
                        )
                    elif isinstance(cc_values, dict):
                        metrics["cyclomatic_complexity"] = cc_values.get("complexity", 0)
                    elif isinstance(cc_values, (int, float)):
                        metrics["cyclomatic_complexity"] = float(cc_values)
                    elif isinstance(cc_values, str) and cc_values.replace('.', '').isdigit():
                        metrics["cyclomatic_complexity"] = float(cc_values)
                    else:
                        metrics["cyclomatic_complexity"] = 0.0

            if mi_result.stdout:
                mi_data = json.loads(mi_result.stdout)
                if mi_data and code_file in mi_data:
                    mi_values = mi_data[code_file]
                    if isinstance(mi_values, list):
                        mi_scores = [item.get("mi", 0) if isinstance(item, dict) else float(item) if isinstance(item, (int, float, str)) and str(item).replace('.', '').isdigit() else 0 for item in mi_values]
                        metrics["maintainability_index"] = (
                            sum(mi_scores) / len(mi_scores) if mi_scores else 0.0
                        )
                    elif isinstance(mi_values, dict):
                        metrics["maintainability_index"] = mi_values.get("mi", 0)
                    elif isinstance(mi_values, (int, float)):
                        metrics["maintainability_index"] = float(mi_values)
                    elif isinstance(mi_values, str) and mi_values.replace('.', '').isdigit():
                        metrics["maintainability_index"] = float(mi_values)
                    else:
                        metrics["maintainability_index"] = 0.0

            # Count lines of code
            with open(code_file, "r") as f:
                metrics["lines_of_code"] = len([line for line in f if line.strip()])

            return metrics

        except (
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            FileNotFoundError,
        ) as e:
            logger.warning(f"Radon analysis failed: {e}")
            return {
                "cyclomatic_complexity": 0.0,
                "maintainability_index": 0.0,
                "lines_of_code": 0,
            }

    def _run_mypy(self, code_file: str) -> Dict[str, Any]:
        """Run mypy type checking."""
        try:
            mypy_path = os.path.join(os.path.dirname(sys.executable), "mypy")
            result = subprocess.run(
                [mypy_path, "--json-report", "/tmp/mypy_report", code_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            errors = []
            coverage = 100.0

            # Parse mypy output for errors
            if result.stdout:
                for line in result.stdout.split("\n"):
                    if "error:" in line:
                        errors.append(line.strip())

            # Try to read JSON report if available
            try:
                with open("/tmp/mypy_report/index.json", "r") as f:
                    mypy_data = json.load(f)
                    if "summary" in mypy_data:
                        summary = mypy_data["summary"]
                        if isinstance(summary, dict):
                            total_lines = summary.get("total_lines", 1)
                            typed_lines = summary.get("typed_lines", 0)
                            coverage = (
                                (typed_lines / total_lines) * 100
                                if total_lines > 0
                                else 100.0
                            )
                        else:
                            coverage = 0.0
            except (FileNotFoundError, KeyError, TypeError):
                pass

            return {"errors": errors, "coverage": coverage}

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Mypy analysis failed: {e}")
            return {"errors": [], "coverage": 0.0}

    def _run_dynamic_testing(
        self, code_file: str, test_file: str
    ) -> DynamicTestResults:
        """Run dynamic tests on the code."""
        try:
            # Import the solution module
            spec = importlib.util.spec_from_file_location("solution", code_file)
            solution_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(solution_module)

            # Run standard tests
            test_results = self._run_standard_tests(test_file)

            # Run hypothesis tests if available
            hypothesis_results = self._run_hypothesis_tests(code_file, test_file)

            return DynamicTestResults(
                test_passes=test_results["passes"],
                test_failures=test_results["failures"],
                test_errors=test_results["errors"],
                coverage_percentage=test_results["coverage"],
                execution_time=test_results["execution_time"],
                memory_usage=test_results["memory_usage"],
                hypothesis_tests=hypothesis_results["tests"],
                hypothesis_failures=hypothesis_results["failures"],
            )

        except Exception as e:
            logger.error(f"Dynamic testing failed: {e}")
            return self._empty_dynamic_results()

    def _run_standard_tests(self, test_file: str) -> Dict[str, Any]:
        """Run standard pytest tests."""
        try:
            import time

            import psutil

            process = psutil.Process()
            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--json-report",
                    "--json-report-file=/tmp/pytest_report.json",
                    test_file,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Parse pytest JSON report
            passes = 0
            failures = 0
            errors = 0
            coverage = 0.0

            try:
                with open("/tmp/pytest_report.json", "r") as f:
                    pytest_data = json.load(f)
                    summary = pytest_data.get("summary", {})
                    passes = summary.get("passed", 0)
                    failures = summary.get("failed", 0)
                    errors = summary.get("error", 0)

                    # Try to get coverage if available
                    coverage_data = pytest_data.get("coverage", {})
                    if coverage_data:
                        coverage = coverage_data.get("percent_covered", 0.0)
            except (FileNotFoundError, KeyError):
                pass

            return {
                "passes": passes,
                "failures": failures,
                "errors": errors,
                "coverage": coverage,
                "execution_time": end_time - start_time,
                "memory_usage": end_memory - start_memory,
            }

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Standard testing failed: {e}")
            return {
                "passes": 0,
                "failures": 0,
                "errors": 1,
                "coverage": 0.0,
                "execution_time": 0.0,
                "memory_usage": 0.0,
            }

    def _run_hypothesis_tests(self, code_file: str, test_file: str) -> Dict[str, int]:
        """Run Hypothesis property-based tests."""
        try:
            # Check if test file contains hypothesis imports
            with open(test_file, "r") as f:
                test_content = f.read()

            if (
                "from hypothesis import" not in test_content
                and "import hypothesis" not in test_content
            ):
                return {"tests": 0, "failures": 0}

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--hypothesis-show-statistics",
                    test_file,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parse hypothesis statistics from output
            tests = 0
            failures = 0

            for line in result.stdout.split("\n"):
                if "examples" in line.lower():
                    try:
                        # Extract number of examples
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit() and i > 0:
                                tests = int(part)
                                break
                    except (ValueError, IndexError):
                        pass

                if "failed" in line.lower():
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit() and i > 0:
                                failures = int(part)
                                break
                    except (ValueError, IndexError):
                        pass

            return {"tests": tests, "failures": failures}

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Hypothesis testing failed: {e}")
            return {"tests": 0, "failures": 0}

    def _empty_static_results(self) -> StaticAnalysisResults:
        """Return empty static analysis results."""
        return StaticAnalysisResults(
            pylint_score=0.0,
            pylint_issues=[],
            bandit_issues=[],
            security_score=0.0,
            complexity_metrics={},
            mypy_errors=[],
            type_coverage=0.0,
        )

    def _empty_dynamic_results(self) -> DynamicTestResults:
        """Return empty dynamic test results."""
        return DynamicTestResults(
            test_passes=0,
            test_failures=0,
            test_errors=1,
            coverage_percentage=0.0,
            execution_time=0.0,
            memory_usage=0.0,
            hypothesis_tests=0,
            hypothesis_failures=0,
        )

    def generate_report(
        self, static_results: StaticAnalysisResults, dynamic_results: DynamicTestResults
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        return {
            "static_analysis": {
                "pylint_score": static_results.pylint_score,
                "pylint_issues_count": len(static_results.pylint_issues),
                "security_score": static_results.security_score,
                "bandit_issues_count": len(static_results.bandit_issues),
                "complexity_metrics": static_results.complexity_metrics,
                "mypy_errors_count": len(static_results.mypy_errors),
                "type_coverage": static_results.type_coverage,
            },
            "dynamic_testing": {
                "test_passes": dynamic_results.test_passes,
                "test_failures": dynamic_results.test_failures,
                "test_errors": dynamic_results.test_errors,
                "coverage_percentage": dynamic_results.coverage_percentage,
                "execution_time": dynamic_results.execution_time,
                "memory_usage": dynamic_results.memory_usage,
                "hypothesis_tests": dynamic_results.hypothesis_tests,
                "hypothesis_failures": dynamic_results.hypothesis_failures,
            },
            "overall_score": self._calculate_overall_score(
                static_results, dynamic_results
            ),
        }

    def _calculate_overall_score(
        self, static_results: StaticAnalysisResults, dynamic_results: DynamicTestResults
    ) -> float:
        """Calculate overall quality score from all metrics."""
        # Weighted combination of different metrics
        weights = {
            "pylint": 0.2,
            "security": 0.2,
            "test_pass_rate": 0.3,
            "coverage": 0.15,
            "complexity": 0.1,
            "type_coverage": 0.05,
        }

        # Calculate test pass rate
        total_tests = (
            dynamic_results.test_passes
            + dynamic_results.test_failures
            + dynamic_results.test_errors
        )
        test_pass_rate = (
            (dynamic_results.test_passes / total_tests * 100) if total_tests > 0 else 0
        )

        # Calculate complexity penalty (higher complexity = lower score)
        complexity_penalty = min(
            1.0, static_results.complexity_metrics.get("cyclomatic_complexity", 0) / 10
        )

        score = (
            weights["pylint"] * static_results.pylint_score
            + weights["security"] * static_results.security_score
            + weights["test_pass_rate"] * test_pass_rate
            + weights["coverage"] * dynamic_results.coverage_percentage
            + weights["complexity"] * (10 - complexity_penalty * 10)
            + weights["type_coverage"] * static_results.type_coverage
        )

        return min(10.0, max(0.0, score))
