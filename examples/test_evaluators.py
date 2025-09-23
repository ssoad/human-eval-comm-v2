#!/usr/bin/env python3
"""
Simple test runner for evaluators modules.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    try:
        # Test imports without using the classes
        import src.aggregator  # noqa: F401
        import src.automated_static_dynamic  # noqa: F401
        import src.calibration  # noqa: F401
        import src.multi_llm_judge  # noqa: F401
        import src.sandbox_runner  # noqa: F401

        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of each module."""
    try:
        # Import modules in function scope
        from src.aggregator import Aggregator
        from src.automated_static_dynamic import AutomatedStaticDynamic
        from src.calibration import Calibration
        from src.multi_llm_judge import MultiLLMJudge
        from src.sandbox_runner import SandboxRunner

        # Test MultiLLMJudge
        MultiLLMJudge()
        print("✓ MultiLLMJudge initialized")

        # Test AutomatedStaticDynamic
        AutomatedStaticDynamic()
        print("✓ AutomatedStaticDynamic initialized")

        # Test SandboxRunner
        SandboxRunner(use_docker=False)
        print("✓ SandboxRunner initialized")

        # Test Calibration
        Calibration()
        print("✓ Calibration initialized")

        # Test Aggregator
        Aggregator()
        print("✓ Aggregator initialized")

        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_sample_evaluation():
    """Test a sample evaluation workflow."""
    try:
        from src.aggregator import Aggregator
        from src.automated_static_dynamic import (
            DynamicTestResults,
            StaticAnalysisResults,
        )

        # Create sample data
        static_results = StaticAnalysisResults(
            pylint_score=8.5,
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
            execution_time=3.5,
            memory_usage=50.0,
            hypothesis_tests=5,
            hypothesis_failures=0,
        )

        # Test aggregation
        aggregator = Aggregator()
        evaluation = aggregator.evaluate_problem(
            problem_id="test_problem",
            test_results=dynamic_results,
            static_results=static_results,
        )

        print(
            f"✓ Sample evaluation completed: "
            f"score = {evaluation.composite_score:.2f}"
        )
        return True

    except Exception as e:
        print(f"✗ Sample evaluation failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing V2 Evaluators Modules...")
    print("=" * 40)

    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Sample Evaluation", test_sample_evaluation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1

    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
