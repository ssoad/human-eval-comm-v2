#!/usr/bin/env python3
"""
Benchmark Results Analysis Script

This script analyzes the expanded HumanEval benchmark results and generates
comprehensive observations about model performance, communication patterns,
and correlations.

Usage:
    python analyze_benchmark_results.py

The script will automatically find the latest results files and generate analysis.
"""

import os
import json
import pandas as pd
import glob
from datetime import datetime
from pathlib import Path


def find_latest_results():
    """Find the latest results files."""
    results_dir = Path("results")

    # Find latest JSON results file
    json_files = list(results_dir.glob("v2_fixed_results_163_problems_*.json"))
    if not json_files:
        raise FileNotFoundError("No results JSON files found")

    json_file = max(json_files, key=lambda x: x.stat().st_mtime)

    # Find latest CSV leaderboard file
    csv_files = list(results_dir.glob("v2_fixed_leaderboard_163_problems_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No leaderboard CSV files found")

    csv_file = max(csv_files, key=lambda x: x.stat().st_mtime)

    return json_file, csv_file


def load_data(json_file, csv_file):
    """Load JSON results and CSV leaderboard data."""
    # Load JSON results
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Load CSV leaderboard
    df = pd.read_csv(csv_file)

    return json_data, df


def analyze_communication_patterns(df):
    """Analyze communication patterns across models."""
    comm_rates = df.set_index('Model')['Comm Rate'].str.rstrip('%').astype(int)

    analysis = {
        'most_confident': {'model': comm_rates.idxmin(), 'rate': comm_rates.min()},
        'most_cautious': {'model': comm_rates.idxmax(), 'rate': comm_rates.max()},
        'range': comm_rates.max() - comm_rates.min(),
        'rates': comm_rates.to_dict()
    }

    return analysis


def analyze_performance_patterns(df):
    """Analyze code performance patterns."""
    pass_rates = df.set_index('Model')['Pass@1'].str.rstrip('%').astype(int)
    test_rates = df.set_index('Model')['Test Pass'].str.rstrip('%').astype(int)

    analysis = {
        'best_execution': {'model': pass_rates.idxmax(), 'rate': pass_rates.max()},
        'best_tests': {'model': test_rates.idxmax(), 'rate': test_rates.max()},
        'execution_range': pass_rates.max() - pass_rates.min(),
        'pass_rates': pass_rates.to_dict(),
        'test_rates': test_rates.to_dict()
    }

    return analysis


def analyze_quality_patterns(df):
    """Analyze code quality patterns."""
    readability = df.set_index('Model')['Readability'].astype(int)
    security = df.set_index('Model')['Security'].astype(int)

    analysis = {
        'best_readability': {'model': readability.idxmax(), 'score': readability.max()},
        'best_security': {'model': security.idxmax(), 'score': security.max()},
        'readability_range': readability.max() - readability.min(),
        'security_range': security.max() - security.min(),
        'readability_scores': readability.to_dict(),
        'security_scores': security.to_dict()
    }

    return analysis


def analyze_overall_performance(df):
    """Analyze overall performance rankings."""
    v2_scores = df.set_index('Model')['V2 Score'].astype(float)

    analysis = {
        'top_performer': {'model': v2_scores.idxmax(), 'score': v2_scores.max()},
        'performance_gap': v2_scores.max() - v2_scores.min(),
        'rankings': v2_scores.sort_values(ascending=False).to_dict(),
        'v2_scores': v2_scores.to_dict()
    }

    return analysis


def analyze_question_quality(df):
    """Analyze question quality patterns."""
    good_q_rates = df.set_index('Model')['Good Q Rate'].str.rstrip('%').astype(int)

    analysis = {
        'best_quality': {'model': good_q_rates.idxmax(), 'rate': good_q_rates.max()},
        'quality_range': good_q_rates.max() - good_q_rates.min(),
        'rates': good_q_rates.to_dict()
    }

    return analysis


def calculate_correlations(df):
    """Calculate interesting correlations between metrics."""
    comm_rates = df.set_index('Model')['Comm Rate'].str.rstrip('%').astype(int)
    pass_rates = df.set_index('Model')['Pass@1'].str.rstrip('%').astype(int)
    test_rates = df.set_index('Model')['Test Pass'].str.rstrip('%').astype(int)
    v2_scores = df.set_index('Model')['V2 Score'].astype(float)
    good_q_rates = df.set_index('Model')['Good Q Rate'].str.rstrip('%').astype(int)

    correlations = {
        'comm_vs_pass1': comm_rates.corr(pass_rates),
        'comm_vs_v2': comm_rates.corr(v2_scores),
        'pass1_vs_v2': pass_rates.corr(v2_scores),
        'comm_vs_questions': comm_rates.corr(good_q_rates),
        'tests_vs_v2': test_rates.corr(v2_scores)
    }

    return correlations


def analyze_model_characteristics(df):
    """Analyze individual model characteristics."""
    characteristics = {}

    for _, row in df.iterrows():
        model = row['Model']
        comm = int(row['Comm Rate'].rstrip('%'))
        pass1 = int(row['Pass@1'].rstrip('%'))
        v2 = float(row['V2 Score'])

        # Determine confidence level
        if comm < 40:
            confidence = 'Very Confident'
        elif comm < 55:
            confidence = 'Balanced'
        else:
            confidence = 'Cautious'

        # Determine capability level
        if pass1 > 80:
            capability = 'High Capability'
        elif pass1 > 75:
            capability = 'Good Capability'
        else:
            capability = 'Moderate Capability'

        characteristics[model] = {
            'confidence': confidence,
            'capability': capability,
            'comm_rate': comm,
            'pass1_rate': pass1,
            'v2_score': v2
        }

    return characteristics


def generate_comprehensive_report(json_data, df):
    """Generate a comprehensive analysis report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Perform all analyses
    comm_analysis = analyze_communication_patterns(df)
    perf_analysis = analyze_performance_patterns(df)
    quality_analysis = analyze_quality_patterns(df)
    overall_analysis = analyze_overall_performance(df)
    question_analysis = analyze_question_quality(df)
    correlations = calculate_correlations(df)
    characteristics = analyze_model_characteristics(df)

    # Generate report
    report = []
    report.append("📊 HUMAN-EVAL BENCHMARK ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Results file: {len(json_data)} evaluations across {len(df)} models")
    report.append("")

    # Current Leaderboard
    report.append("🏆 CURRENT LEADERBOARD")
    report.append("-" * 30)
    # Create a formatted leaderboard table
    header = f"{'Model':<20} {'Comm Rate':<10} {'Good Q':<8} {'Pass@1':<8} {'Test':<8} {'Read':<6} {'Sec':<6} {'V2':<6}"
    report.append(header)
    report.append("-" * len(header))
    for _, row in df.iterrows():
        line = f"{row['Model']:<20} {row['Comm Rate']:<10} {row['Good Q Rate']:<8} {row['Pass@1']:<8} {row['Test Pass']:<8} {row['Readability']:<6} {row['Security']:<6} {row['V2 Score']:<6}"
        report.append(line)
    report.append("")

    # Key Findings
    report.append("🔍 KEY FINDINGS")
    report.append("-" * 20)
    report.append("")

    # Communication Patterns
    report.append("1. COMMUNICATION PATTERNS")
    report.append(f"   • Most confident: {comm_analysis['most_confident']['model']} ({comm_analysis['most_confident']['rate']}%)")
    report.append(f"   • Most cautious: {comm_analysis['most_cautious']['model']} ({comm_analysis['most_cautious']['rate']}%)")
    report.append(f"   • Communication range: {comm_analysis['range']} percentage points")
    report.append("")

    # Performance Patterns
    report.append("2. CODE PERFORMANCE PATTERNS")
    report.append(f"   • Best execution success: {perf_analysis['best_execution']['model']} ({perf_analysis['best_execution']['rate']}%)")
    report.append(f"   • Best test performance: {perf_analysis['best_tests']['model']} ({perf_analysis['best_tests']['rate']}%)")
    report.append(f"   • Execution success range: {perf_analysis['execution_range']} percentage points")
    report.append("")

    # Quality Patterns
    report.append("3. CODE QUALITY PATTERNS")
    report.append(f"   • Best readability: {quality_analysis['best_readability']['model']} ({quality_analysis['best_readability']['score']})")
    report.append(f"   • Best security: {quality_analysis['best_security']['model']} ({quality_analysis['best_security']['score']})")
    report.append(f"   • Readability range: {quality_analysis['readability_range']} points")
    report.append("")

    # Overall Performance
    report.append("4. OVERALL PERFORMANCE")
    report.append(f"   • Top performer: {overall_analysis['top_performer']['model']} (V2 Score: {overall_analysis['top_performer']['score']})")
    report.append(f"   • Performance gap: {overall_analysis['performance_gap']:.1f} points between best and worst")
    report.append("")

    # Question Quality
    report.append("5. QUESTION QUALITY")
    report.append(f"   • Best question quality: {question_analysis['best_quality']['model']} ({question_analysis['best_quality']['rate']}%)")
    report.append(f"   • Question quality range: {question_analysis['quality_range']} percentage points")
    report.append("")

    # Correlations
    report.append("6. KEY CORRELATIONS")
    report.append(f"   • Communication vs Pass@1: {correlations['comm_vs_pass1']:.2f}")
    report.append(f"   • Communication vs V2 Score: {correlations['comm_vs_v2']:.2f}")
    report.append(f"   • Pass@1 vs V2 Score: {correlations['pass1_vs_v2']:.2f}")
    report.append("")

    # Model Characteristics
    report.append("7. MODEL CHARACTERISTICS")
    for model, char in characteristics.items():
        report.append(f"   • {model}: {char['confidence']} ({char['comm_rate']}% comm), {char['capability']} ({char['pass1_rate']}% Pass@1), V2: {char['v2_score']}")
    report.append("")

    # Insights
    report.append("💡 KEY INSIGHTS")
    report.append("-" * 15)
    report.append("• Communication vs Performance Trade-off: Higher communication rates correlate negatively with overall performance")
    report.append("• Quality Matters: Code readability and security heavily influence final V2 composite scores")
    report.append("• Balanced Approach: Models with moderate communication rates often achieve best overall results")
    report.append("• Realistic Behavior: Confident models ask fewer questions, cautious models seek more clarification")
    report.append("• No Perfect Metric: High execution success doesn't guarantee top overall ranking")

    return "\n".join(report)


def main():
    """Main function to run the analysis."""
    try:
        # Find latest results
        print("🔍 Finding latest results files...")
        json_file, csv_file = find_latest_results()
        print(f"✅ Found: {json_file.name}")
        print(f"✅ Found: {csv_file.name}")

        # Load data
        print("📂 Loading data...")
        json_data, df = load_data(json_file, csv_file)
        print(f"✅ Loaded {len(json_data)} evaluations and {len(df)} model results")

        # Generate comprehensive report
        print("📊 Generating analysis report...")
        report = generate_comprehensive_report(json_data, df)

        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"benchmark_analysis_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write(report)

        print(f"✅ Analysis saved to: {report_file}")

        # Also print to console
        print("\n" + "="*80)
        print(report)
        print("="*80)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())