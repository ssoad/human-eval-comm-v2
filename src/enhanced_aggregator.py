"""
Enhanced Aggregator with Configurable Scoring Formulas

Provides flexible aggregation of evaluation results with customizable scoring formulas,
weights, and normalization strategies. Supports multiple evaluation dimensions.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ScoringFormula:
    """A configurable scoring formula with weights and normalization."""

    name: str
    description: str
    weights: Dict[str, float]
    normalization_strategy: str = "minmax"  # minmax, zscore, robust
    penalty_factors: Dict[str, float] = None
    bonus_factors: Dict[str, float] = None

    def __post_init__(self):
        if self.penalty_factors is None:
            self.penalty_factors = {}
        if self.bonus_factors is None:
            self.bonus_factors = {}


@dataclass
class AggregatedResult:
    """Result of aggregation with detailed breakdown."""

    problem_id: str
    composite_score: float
    weighted_composite_score: float
    individual_scores: Dict[str, float]
    confidence_intervals: Dict[str, tuple]
    formula_used: str
    normalization_applied: str
    penalties_applied: Dict[str, float]
    bonuses_applied: Dict[str, float]
    metadata: Dict[str, Any]


class EnhancedAggregator:
    """Enhanced aggregator with configurable scoring formulas."""

    def __init__(self, config_file: str = "aggregation_config.json"):
        """Initialize the enhanced aggregator."""
        self.formulas = self._load_default_formulas()
        self.current_formula = "balanced_v2"
        self._load_config(config_file)

    def _load_default_formulas(self) -> Dict[str, ScoringFormula]:
        """Load default scoring formulas."""
        return {
            "balanced_v2": ScoringFormula(
                name="Balanced V2",
                description="Balanced formula emphasizing all evaluation dimensions",
                weights={
                    "test_pass_rate": 0.25,
                    "llm_consensus": 0.20,
                    "static_analysis": 0.15,
                    "security_score": 0.15,
                    "readability": 0.10,
                    "resource_efficiency": 0.10,
                    "complexity_penalty": 0.05
                },
                penalty_factors={
                    "high_complexity": 0.1,
                    "security_issues": 0.15,
                    "timeout_penalty": 0.2
                },
                bonus_factors={
                    "perfect_test_coverage": 0.05,
                    "excellent_readability": 0.03
                }
            ),

            "correctness_focused": ScoringFormula(
                name="Correctness Focused",
                description="Heavily weights test passing and LLM consensus",
                weights={
                    "test_pass_rate": 0.40,
                    "llm_consensus": 0.35,
                    "static_analysis": 0.10,
                    "security_score": 0.10,
                    "readability": 0.03,
                    "resource_efficiency": 0.01,
                    "complexity_penalty": 0.01
                }
            ),

            "efficiency_focused": ScoringFormula(
                name="Efficiency Focused",
                description="Emphasizes resource usage and performance",
                weights={
                    "test_pass_rate": 0.20,
                    "llm_consensus": 0.15,
                    "static_analysis": 0.10,
                    "security_score": 0.10,
                    "readability": 0.10,
                    "resource_efficiency": 0.25,
                    "complexity_penalty": 0.10
                }
            ),

            "security_first": ScoringFormula(
                name="Security First",
                description="Prioritizes security and code quality",
                weights={
                    "test_pass_rate": 0.15,
                    "llm_consensus": 0.10,
                    "static_analysis": 0.20,
                    "security_score": 0.35,
                    "readability": 0.10,
                    "resource_efficiency": 0.05,
                    "complexity_penalty": 0.05
                }
            )
        }

    def _load_config(self, config_file: str):
        """Load configuration from file."""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Load custom formulas
                if "custom_formulas" in config:
                    for formula_name, formula_data in config["custom_formulas"].items():
                        self.formulas[formula_name] = ScoringFormula(**formula_data)

                # Set current formula
                if "current_formula" in config:
                    self.current_formula = config["current_formula"]

                logger.info(f"Loaded configuration from {config_file}")

        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    def add_custom_formula(self, formula: ScoringFormula):
        """Add a custom scoring formula."""
        self.formulas[formula.name] = formula
        logger.info(f"Added custom formula: {formula.name}")

    def set_current_formula(self, formula_name: str):
        """Set the current scoring formula."""
        if formula_name not in self.formulas:
            available = list(self.formulas.keys())
            raise ValueError(f"Formula '{formula_name}' not found. Available: {available}")

        self.current_formula = formula_name
        logger.info(f"Set current formula to: {formula_name}")

    def aggregate_results(self, evaluation_data: Dict[str, Any]) -> AggregatedResult:
        """Aggregate evaluation results using current formula."""
        formula = self.formulas[self.current_formula]

        # Extract individual scores
        individual_scores = self._extract_individual_scores(evaluation_data)

        # Apply normalization
        normalized_scores = self._normalize_scores(individual_scores, formula)

        # Calculate penalties and bonuses
        penalties = self._calculate_penalties(evaluation_data, formula)
        bonuses = self._calculate_bonuses(evaluation_data, formula)

        # Calculate composite score
        composite_score = self._calculate_composite_score(normalized_scores, formula)

        # Apply penalties and bonuses
        adjusted_score = composite_score
        for penalty_value in penalties.values():
            adjusted_score -= penalty_value
        for bonus_value in bonuses.values():
            adjusted_score += bonus_value

        # Ensure score is within bounds
        adjusted_score = max(0.0, min(10.0, adjusted_score))

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(evaluation_data)

        return AggregatedResult(
            problem_id=evaluation_data.get('problem_id', 'unknown'),
            composite_score=round(composite_score, 3),
            weighted_composite_score=round(adjusted_score, 3),
            individual_scores=normalized_scores,
            confidence_intervals=confidence_intervals,
            formula_used=self.current_formula,
            normalization_applied=formula.normalization_strategy,
            penalties_applied=penalties,
            bonuses_applied=bonuses,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "raw_scores": individual_scores,
                "evaluation_data": evaluation_data
            }
        )

    def _extract_individual_scores(self, evaluation_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract individual scores from evaluation data."""
        scores = {}

        # Test pass rate
        dynamic_results = evaluation_data.get('dynamic_results', {})
        total_tests = (dynamic_results.get('test_passes', 0) +
                      dynamic_results.get('test_failures', 0) +
                      dynamic_results.get('test_errors', 0))
        if total_tests > 0:
            scores['test_pass_rate'] = (dynamic_results.get('test_passes', 0) / total_tests) * 10
        else:
            scores['test_pass_rate'] = 0.0

        # LLM consensus
        llm_scores = evaluation_data.get('llm_scores', {})
        scores['llm_consensus'] = llm_scores.get('consensus_score', 0.0)

        # Static analysis
        static_results = evaluation_data.get('static_results', {})
        scores['static_analysis'] = static_results.get('pylint_score', 0.0)

        # Security score
        scores['security_score'] = static_results.get('security_score', 0.0)

        # Readability (based on pylint score and complexity)
        complexity_metrics = static_results.get('complexity_metrics', {})
        complexity_penalty = min(1.0, complexity_metrics.get('cyclomatic_complexity', 0) / 10)
        scores['readability'] = max(0.0, static_results.get('pylint_score', 0.0) - complexity_penalty)

        # Resource efficiency
        sandbox_results = evaluation_data.get('sandbox_results', {})
        execution_time = sandbox_results.get('execution_time', 1.0)
        memory_used = sandbox_results.get('memory_used', 0.0)

        # Efficiency score based on resource usage (lower is better)
        time_score = max(0.0, 10.0 - (execution_time * 2))  # Penalize slow execution
        memory_score = max(0.0, 10.0 - abs(memory_used))   # Penalize high memory usage
        scores['resource_efficiency'] = (time_score + memory_score) / 2

        # Complexity penalty (inverse of complexity)
        scores['complexity_penalty'] = max(0.0, 10.0 - complexity_penalty * 10)

        return scores

    def _normalize_scores(self, scores: Dict[str, float], formula: ScoringFormula) -> Dict[str, float]:
        """Normalize scores using the specified strategy."""
        if formula.normalization_strategy == "minmax":
            return self._minmax_normalization(scores)
        elif formula.normalization_strategy == "zscore":
            return self._zscore_normalization(scores)
        elif formula.normalization_strategy == "robust":
            return self._robust_normalization(scores)
        else:
            logger.warning(f"Unknown normalization strategy: {formula.normalization_strategy}")
            return scores

    def _minmax_normalization(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalization to 0-10 scale."""
        normalized = {}
        for key, value in scores.items():
            # Assuming scores are already in 0-10 range, but clamp just in case
            normalized[key] = max(0.0, min(10.0, value))
        return normalized

    def _zscore_normalization(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Z-score normalization."""
        import numpy as np

        values = list(scores.values())
        if len(values) < 2:
            return scores

        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return {k: 5.0 for k in scores.keys()}  # Neutral score if no variance

        normalized = {}
        for key, value in scores.items():
            z_score = (value - mean_val) / std_val
            # Convert to 0-10 scale
            normalized_score = 5.0 + (z_score * 2.0)  # 2 std dev range = 10 points
            normalized[key] = max(0.0, min(10.0, normalized_score))

        return normalized

    def _robust_normalization(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Robust normalization using median and MAD."""
        import numpy as np

        values = list(scores.values())
        if len(values) < 2:
            return scores

        median_val = np.median(values)
        mad = np.median([abs(v - median_val) for v in values])

        if mad == 0:
            return {k: 5.0 for k in scores.keys()}

        normalized = {}
        for key, value in scores.items():
            robust_z = (value - median_val) / mad
            normalized_score = 5.0 + (robust_z * 2.0)
            normalized[key] = max(0.0, min(10.0, normalized_score))

        return normalized

    def _calculate_penalties(self, evaluation_data: Dict[str, Any], formula: ScoringFormula) -> Dict[str, float]:
        """Calculate penalties based on evaluation data."""
        penalties = {}

        # High complexity penalty
        static_results = evaluation_data.get('static_results', {})
        complexity = static_results.get('complexity_metrics', {}).get('cyclomatic_complexity', 0)
        if complexity > 15:
            penalty = formula.penalty_factors.get('high_complexity', 0.1) * (complexity - 15) / 5
            penalties['high_complexity'] = min(penalty, 1.0)

        # Security issues penalty
        security_score = static_results.get('security_score', 10.0)
        if security_score < 7.0:
            penalty = formula.penalty_factors.get('security_issues', 0.15) * (7.0 - security_score)
            penalties['security_issues'] = min(penalty, 1.0)

        # Timeout penalty
        sandbox_results = evaluation_data.get('sandbox_results', {})
        if sandbox_results.get('timeout', False):
            penalties['timeout_penalty'] = formula.penalty_factors.get('timeout_penalty', 0.2)

        return penalties

    def _calculate_bonuses(self, evaluation_data: Dict[str, Any], formula: ScoringFormula) -> Dict[str, float]:
        """Calculate bonuses based on evaluation data."""
        bonuses = {}

        # Perfect test coverage bonus
        dynamic_results = evaluation_data.get('dynamic_results', {})
        coverage = dynamic_results.get('coverage_percentage', 0.0)
        if coverage >= 95.0:
            bonuses['perfect_test_coverage'] = formula.bonus_factors.get('perfect_test_coverage', 0.05)

        # Excellent readability bonus
        static_results = evaluation_data.get('static_results', {})
        readability = static_results.get('pylint_score', 0.0)
        if readability >= 9.0:
            bonuses['excellent_readability'] = formula.bonus_factors.get('excellent_readability', 0.03)

        return bonuses

    def _calculate_composite_score(self, normalized_scores: Dict[str, float], formula: ScoringFormula) -> float:
        """Calculate composite score using formula weights."""
        composite = 0.0
        total_weight = 0.0

        for metric, weight in formula.weights.items():
            if metric in normalized_scores:
                composite += normalized_scores[metric] * weight
                total_weight += weight

        # Normalize by total weight (in case weights don't sum to 1)
        if total_weight > 0:
            composite /= total_weight

        return composite

    def _calculate_confidence_intervals(self, evaluation_data: Dict[str, Any]) -> Dict[str, tuple]:
        """Calculate confidence intervals for scores."""
        intervals = {}

        # LLM consensus confidence interval
        llm_scores = evaluation_data.get('llm_scores', {})
        if 'judge_responses' in llm_scores:
            responses = llm_scores['judge_responses']
            if len(responses) > 1:
                scores = [r.get('score', 0) for r in responses]
                mean_score = sum(scores) / len(scores)
                std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5

                # 95% confidence interval
                margin = 1.96 * std_score / (len(scores) ** 0.5)
                intervals['llm_consensus'] = (
                    max(0.0, mean_score - margin),
                    min(10.0, mean_score + margin)
                )

        return intervals

    def get_available_formulas(self) -> List[str]:
        """Get list of available scoring formulas."""
        return list(self.formulas.keys())

    def get_formula_details(self, formula_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a formula."""
        if formula_name not in self.formulas:
            return None

        formula = self.formulas[formula_name]
        return {
            "name": formula.name,
            "description": formula.description,
            "weights": formula.weights,
            "normalization_strategy": formula.normalization_strategy,
            "penalty_factors": formula.penalty_factors,
            "bonus_factors": formula.bonus_factors
        }

    def save_config(self, config_file: str = "aggregation_config.json"):
        """Save current configuration to file."""
        config = {
            "current_formula": self.current_formula,
            "custom_formulas": {}
        }

        # Save custom formulas only
        for name, formula in self.formulas.items():
            if name not in ["balanced_v2", "correctness_focused", "efficiency_focused", "security_first"]:
                config["custom_formulas"][name] = asdict(formula)

        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def batch_aggregate(self, evaluation_results: List[Dict[str, Any]]) -> List[AggregatedResult]:
        """Aggregate multiple evaluation results."""
        results = []
        for evaluation_data in evaluation_results:
            try:
                result = self.aggregate_results(evaluation_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to aggregate result for {evaluation_data.get('problem_id', 'unknown')}: {e}")

        return results