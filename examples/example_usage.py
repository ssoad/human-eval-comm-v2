
# Ensure .env is loaded for API keys
import os

from dotenv import load_dotenv
import asyncio

from evaluators import (Aggregator, AutomatedStaticDynamic, MultiLLMJudge,
                        SandboxRunner)

load_dotenv()

# Check for API keys and provide guidance
print("🔍 Checking for API keys...")
keys_found = []
if os.getenv("GEMINI_API_KEY"):
    keys_found.append("Gemini")
if os.getenv("OPENAI_API_KEY"):
    keys_found.append("OpenAI")
if os.getenv("ANTHROPIC_API_KEY"):
    keys_found.append("Anthropic")

if not keys_found:
    print("   ⚠️ No API keys found in .env file. LLM evaluation will be skipped.")
    print("      Please create a .env file and add your API keys.")
else:
    print(f"   ✅ API keys found for: {', '.join(keys_found)}")

#!/usr/bin/env python3
"""
Simple example showing how to use the V2 Evaluators Framework
"""


async def main():
    """Example usage of V2 Evaluators Framework."""

    # Sample code to evaluate
    code = """
def fibonacci(n):
    '''Calculate the nth Fibonacci number.'''
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    test_code = """
import sys
import os

# Add the current directory to Python path so tests can import solution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fibonacci():
    '''Test basic fibonacci functionality.'''
    # Import here to ensure it's available
    try:
        from solution import fibonacci
    except ImportError:
        import solution
        fibonacci = solution.fibonacci

    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55

def test_fibonacci_edge_cases():
    '''Test edge cases for fibonacci.'''
    # Import here to ensure it's available
    try:
        from solution import fibonacci
    except ImportError:
        import solution
        fibonacci = solution.fibonacci

    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    # Test negative input (should raise ValueError)
    try:
        fibonacci(-1)
        assert False, "Should have raised ValueError for negative input"
    except ValueError:
        pass  # Expected behavior

if __name__ == "__main__":
    test_fibonacci()
    test_fibonacci_edge_cases()
    print("All tests passed!")
"""

    problem_description = (
        "Implement a function to calculate Fibonacci numbers with proper error handling"
    )

    print("🎯 V2 Evaluators Framework Example")
    print("=" * 50)

    # Initialize evaluators
    print("🔧 Initializing evaluators...")
    analyzer = AutomatedStaticDynamic()
    runner = SandboxRunner(use_docker=False)  # Use subprocess for now
    aggregator = Aggregator()

    # Run static and dynamic analysis
    print("\n📊 Running static and dynamic analysis...")
    static_results, dynamic_results = analyzer.analyze_code(
        code, test_code, "fibonacci_example"
    )

    print(f"   Pylint Score: {static_results.pylint_score:.2f}/10")
    print(f"   Security Score: {static_results.security_score:.2f}/10")
    print(f"   Test Passes: {dynamic_results.test_passes}")
    print(f"   Test Failures: {dynamic_results.test_failures}")

    # Run in sandbox
    print("\n🏃 Running code in sandbox...")
    execution_result = runner.run_code(code, test_code)

    print(f"   Execution Success: {execution_result.success}")
    print(f"   Execution Time: {execution_result.execution_time:.2f}s")
    memory_display = f"{execution_result.memory_used:+.2f}MB"
    memory_desc = "(consumed)" if execution_result.memory_used >= 0 else "(freed)"
    print(f"   Memory Change: {memory_display} {memory_desc}")

    # Run LLM evaluation (if API keys are available)
    print("\n🤖 Running LLM evaluation...")
    llm_scores = None
    try:
        judge = MultiLLMJudge()
        llm_scores = await judge.evaluate_code(code, problem_description)
        print(f"   LLM Consensus Score: {llm_scores.consensus_score:.2f}/10")
        print(f"   Mean Confidence: {llm_scores.mean_confidence:.2f}")
    except Exception as e:
        print(f"   ⚠️  LLM evaluation skipped: {e}")
        print("   (Set API keys to enable LLM evaluation)")

    # Aggregate results
    print("\n📈 Calculating composite score...")
    evaluation = aggregator.evaluate_problem(
        problem_id="fibonacci_example",
        test_results=dynamic_results,
        static_results=static_results,
        sandbox_results=execution_result,
        llm_scores=llm_scores,
    )

    # Display results
    print("\n🎉 Final Evaluation Results:")
    print("=" * 50)
    print(f"Composite Score: {evaluation.composite_score:.2f}/10")
    print(f"Weighted Score: {evaluation.weighted_composite_score:.2f}/10")
    print()
    print("Individual Scores:")
    print(f"  Test Pass Rate: {evaluation.test_pass_rate_score:.2f}/10")
    print(f"  LLM Consensus: {evaluation.llm_consensus_score:.2f}/10")
    print(f"  Static Analysis: {evaluation.static_analysis_score:.2f}/10")
    print(f"  Security: {evaluation.security_score:.2f}/10")
    print(f"  Readability: {evaluation.readability_score:.2f}/10")
    print(f"  Resource Efficiency: {evaluation.resource_efficiency_score:.2f}/10")
    print(f"  Complexity Penalty: {evaluation.complexity_penalty:.2f}/10")

    print(
        f"\nConfidence Interval: [{evaluation.confidence_interval[0]:.2f}, {evaluation.confidence_interval[1]:.2f}]"
    )

    # Export results
    print("\n💾 Exporting results...")
    aggregator.export_evaluation_json(evaluation, "fibonacci_evaluation.json")
    print("   Results saved to: fibonacci_evaluation.json")

    # Generate summary
    aggregator.export_summary_csv("evaluation_summary.csv")
    print("   Summary saved to: evaluation_summary.csv")

    print("\n✅ Example completed successfully!")
    print("\n📚 Next steps:")
    print("1. Check the generated JSON and CSV files")
    print("2. Try modifying the code and running again")
    print("3. Experiment with different evaluation weights")
    print("4. Read README.md for advanced usage")


if __name__ == "__main__":
    asyncio.run(main())
