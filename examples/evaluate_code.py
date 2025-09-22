#!/usr/bin/env python3
"""
Simple command-line interface for evaluating code with V2 Evaluators Framework
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evaluators import Aggregator, AutomatedStaticDynamic, MultiLLMJudge, SandboxRunner


def load_code_from_file(file_path):
    """Load code from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print(f"⚠️  Warning: File {file_path} is empty")
            return content
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    except PermissionError:
        print(f"❌ Error: Permission denied reading file: {file_path}")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"❌ Error: Cannot decode file as UTF-8: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        sys.exit(1)


def evaluate_single_code(
    code, test_code="", problem_description="", output_format="json"
):
    """Evaluate a single piece of code."""
    print("🔍 Starting evaluation...")

    # Initialize evaluators
    analyzer = AutomatedStaticDynamic()
    runner = SandboxRunner(use_docker=False)  # Use subprocess for simplicity
    aggregator = Aggregator()

    # Run static and dynamic analysis
    print("📊 Running static and dynamic analysis...")
    try:
        static_results, dynamic_results = analyzer.analyze_code(code, test_code)
        print("   ✅ Static and dynamic analysis completed")
    except Exception as e:
        print(f"   ⚠️  Static/dynamic analysis failed: {e}")
        # Create dummy results for failed analysis
        from evaluators.automated_static_dynamic import (
            DynamicTestResults,
            StaticAnalysisResults,
        )

        static_results = StaticAnalysisResults(
            pylint_score=0.0,
            pylint_issues=[],
            bandit_issues=[],
            security_score=0.0,
            complexity_metrics={
                "cyclomatic_complexity": 0.0,
                "maintainability_index": 0.0,
            },
            mypy_errors=[],
            type_coverage=0.0,
        )
        dynamic_results = DynamicTestResults(
            test_passes=0,
            test_failures=0,
            test_errors=0,
            coverage_percentage=0.0,
            execution_time=0.0,
            memory_usage=0.0,
            hypothesis_tests=0,
            hypothesis_failures=0,
        )

    # Run in sandbox
    print("🏃 Running code in sandbox...")
    try:
        execution_result = runner.run_code(code, test_code)
        print("   ✅ Sandbox execution completed")
    except Exception as e:
        print(f"   ⚠️  Sandbox execution failed: {e}")
        # Create dummy result for failed execution
        from evaluators.sandbox_runner import ExecutionResult

        execution_result = ExecutionResult(
            success=False,
            stdout="",
            stderr=str(e),
            exit_code=1,
            execution_time=0.0,
            memory_used=0.0,
            cpu_time_used=0.0,
            timeout=False,
            killed=False,
            error_message=str(e),
        )

    # Aggregate results
    print("📈 Calculating composite score...")
    evaluation = aggregator.evaluate_problem(
        problem_id="evaluation",
        test_results=dynamic_results,
        static_results=static_results,
        sandbox_results=execution_result,
    )

    # Output results
    if output_format == "json":
        result = {
            "composite_score": evaluation.composite_score,
            "test_pass_rate": evaluation.test_pass_rate_score,
            "static_analysis_score": evaluation.static_analysis_score,
            "security_score": evaluation.security_score,
            "readability_score": evaluation.readability_score,
            "resource_efficiency_score": evaluation.resource_efficiency_score,
            "execution_success": execution_result.success,
            "execution_time": execution_result.execution_time,
            "memory_used": execution_result.memory_used,
        }
        # Print only JSON, no other output
        print(json.dumps(result))
    else:
        print(f"\n📊 Evaluation Results:")
        print(f"Composite Score: {evaluation.composite_score:.2f}/10")
        print(f"Test Pass Rate: {evaluation.test_pass_rate_score:.2f}/10")
        print(f"Static Analysis: {evaluation.static_analysis_score:.2f}/10")
        print(f"Security Score: {evaluation.security_score:.2f}/10")
        print(f"Readability: {evaluation.readability_score:.2f}/10")
        print(f"Resource Efficiency: {evaluation.resource_efficiency_score:.2f}/10")
        print(f"Execution Success: {execution_result.success}")
        print(f"Execution Time: {execution_result.execution_time:.2f}s")
        print(f"Memory Used: {execution_result.memory_used:.2f}MB")


async def evaluate_with_llm(code, problem_description=""):
    """Evaluate code with LLM judges."""
    print("🤖 Running LLM evaluation...")

    try:
        judge = MultiLLMJudge()
        llm_scores = await judge.evaluate_code(code, problem_description)

        print(f"   LLM Consensus Score: {llm_scores.consensus_score:.2f}/10")
        print(f"   Mean Confidence: {llm_scores.mean_confidence:.2f}")
        print(f"   Score Standard Deviation: {llm_scores.score_std:.2f}")
        print("   ✅ LLM evaluation completed")

        return llm_scores
    except Exception as e:
        print(f"   ⚠️  LLM evaluation failed: {e}")
        print("   Make sure your API keys are set correctly")
        print("   Check config.yaml for LLM model configurations")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate code with V2 Evaluators Framework"
    )
    parser.add_argument("code", help="Code to evaluate (file path or direct code)")
    parser.add_argument("--test", "-t", help="Test code (file path or direct code)")
    parser.add_argument("--problem", "-p", help="Problem description")
    parser.add_argument(
        "--output",
        "-o",
        choices=["json", "human"],
        default="human",
        help="Output format",
    )
    parser.add_argument("--llm", action="store_true", help="Include LLM evaluation")
    parser.add_argument(
        "--file", "-f", action="store_true", help="Treat code as file path"
    )

    args = parser.parse_args()

    # Load code
    if args.file:
        # Force file mode - treat code argument as file path
        code = load_code_from_file(args.code)
        print(f"📁 Loaded code from: {args.code}")
    elif Path(args.code).exists():
        # Auto-detect if it's a file path
        code = load_code_from_file(args.code)
        print(f"📁 Loaded code from: {args.code}")
    else:
        # Treat as direct code
        code = args.code
        print("📝 Using provided code")

    # Load test code
    test_code = ""
    if args.test:
        if Path(args.test).exists():
            test_code = load_code_from_file(args.test)
            print(f"🧪 Loaded tests from: {args.test}")
        else:
            test_code = args.test
            print("🧪 Using provided test code")

    # Run basic evaluation
    evaluate_single_code(code, test_code, args.problem or "", args.output)

    # Run LLM evaluation if requested
    if args.llm:
        print("\n" + "=" * 50)
        asyncio.run(evaluate_with_llm(code, args.problem or ""))


if __name__ == "__main__":
    main()
