"""
Human Calibration System

Manages human annotations for calibrating LLM judge reliability.
Learns P(human_correct | judge_confidence) using isotonic/logistic regression.
Provides calibrated confidence scores for improved evaluation accuracy.
"""

import logging
import json
import csv
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSample:
    """A single calibration sample with judge predictions and human labels."""

    problem_id: str
    judge_name: str
    judge_score: float
    judge_confidence: float
    human_correct: bool  # True if human says code is correct
    human_question_quality: int  # 1-5 scale for question quality
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class CalibrationResult:
    """Results from calibration training."""

    judge_name: str
    calibration_method: str  # 'isotonic' or 'logistic'
    reliability_score: float  # Overall reliability (0-1)
    samples_used: int
    calibration_curve: List[Tuple[float, float]]  # (confidence, calibrated_prob)
    validation_metrics: Dict[str, float]


class HumanCalibrationSystem:
    """Manages human calibration for LLM judges."""

    def __init__(self, calibration_dir: str = "calibration_data"):
        """Initialize the calibration system."""
        self.calibration_dir = calibration_dir
        self.samples: List[CalibrationSample] = []
        self.calibration_models: Dict[str, CalibrationResult] = {}

        os.makedirs(calibration_dir, exist_ok=True)
        self._load_existing_samples()

    def _load_existing_samples(self):
        """Load existing calibration samples from disk."""
        samples_file = os.path.join(self.calibration_dir, "calibration_samples.jsonl")

        if os.path.exists(samples_file):
            try:
                with open(samples_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            sample = CalibrationSample(**data)
                            self.samples.append(sample)
                logger.info(f"Loaded {len(self.samples)} calibration samples")
            except Exception as e:
                logger.error(f"Failed to load calibration samples: {e}")

    def add_sample(self, sample: CalibrationSample):
        """Add a new calibration sample."""
        self.samples.append(sample)
        self._save_sample(sample)

    def _save_sample(self, sample: CalibrationSample):
        """Save a sample to disk."""
        samples_file = os.path.join(self.calibration_dir, "calibration_samples.jsonl")

        try:
            with open(samples_file, 'a') as f:
                f.write(json.dumps(asdict(sample)) + '\n')
        except Exception as e:
            logger.error(f"Failed to save calibration sample: {e}")

    def generate_annotation_candidates(self, evaluation_results: List[Dict[str, Any]],
                                    strategy: str = "disagreement") -> List[Dict[str, Any]]:
        """Generate candidates for human annotation based on strategy."""

        candidates = []

        if strategy == "disagreement":
            # Find cases where judges disagree significantly
            for result in evaluation_results:
                judge_scores = [j.get('score', 0) for j in result.get('judge_responses', [])]
                if len(judge_scores) > 1:
                    score_range = max(judge_scores) - min(judge_scores)
                    if score_range > 0.3:  # Significant disagreement
                        candidates.append({
                            'problem_id': result.get('problem_id'),
                            'reason': f'Judge disagreement: {score_range:.2f}',
                            'judge_scores': judge_scores,
                            'priority': 'high'
                        })

        elif strategy == "low_confidence":
            # Find cases with low confidence consensus
            for result in evaluation_results:
                consensus_conf = result.get('consensus_confidence', 1.0)
                if consensus_conf < 0.7:
                    candidates.append({
                        'problem_id': result.get('problem_id'),
                        'reason': f'Low confidence: {consensus_conf:.2f}',
                        'priority': 'medium'
                    })

        elif strategy == "random_sample":
            # Random sampling for baseline
            import random
            sample_size = min(50, len(evaluation_results))
            sampled = random.sample(evaluation_results, sample_size)
            for result in sampled:
                candidates.append({
                    'problem_id': result.get('problem_id'),
                    'reason': 'Random sample',
                    'priority': 'low'
                })

        return candidates

    def export_annotation_csv(self, candidates: List[Dict[str, Any]],
                            output_file: str = "annotation_candidates.csv"):
        """Export candidates to CSV for human annotation."""

        output_path = os.path.join(self.calibration_dir, output_file)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'problem_id', 'reason', 'priority', 'judge_scores',
                'human_correct', 'question_quality', 'notes'
            ])
            writer.writeheader()

            for candidate in candidates:
                row = {
                    'problem_id': candidate['problem_id'],
                    'reason': candidate['reason'],
                    'priority': candidate['priority'],
                    'judge_scores': str(candidate.get('judge_scores', [])),
                    'human_correct': '',  # To be filled by human
                    'question_quality': '',  # To be filled by human
                    'notes': ''  # Additional notes
                }
                writer.writerow(row)

        logger.info(f"Exported {len(candidates)} annotation candidates to {output_path}")
        return output_path

    def import_annotations(self, annotation_file: str):
        """Import human annotations from CSV file."""

        annotation_path = os.path.join(self.calibration_dir, annotation_file)

        if not os.path.exists(annotation_path):
            logger.error(f"Annotation file not found: {annotation_path}")
            return

        new_samples = 0

        with open(annotation_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Skip if not annotated
                if not row.get('human_correct', '').strip():
                    continue

                try:
                    human_correct = row['human_correct'].lower() in ['true', '1', 'yes']
                    question_quality = int(row.get('question_quality', 3))

                    # Create calibration samples for each judge
                    problem_id = row['problem_id']

                    # This would need to be enhanced to get actual judge data
                    # For now, create placeholder samples
                    sample = CalibrationSample(
                        problem_id=problem_id,
                        judge_name="placeholder_judge",
                        judge_score=0.5,  # Would need actual judge score
                        judge_confidence=0.8,  # Would need actual confidence
                        human_correct=human_correct,
                        human_question_quality=question_quality,
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            'source': annotation_file,
                            'notes': row.get('notes', '')
                        }
                    )

                    self.add_sample(sample)
                    new_samples += 1

                except Exception as e:
                    logger.error(f"Failed to process annotation row: {e}")

        logger.info(f"Imported {new_samples} new calibration samples")

    def train_calibration_models(self, method: str = "isotonic"):
        """Train calibration models for each judge."""

        if not self.samples:
            logger.warning("No calibration samples available for training")
            return

        # Group samples by judge
        judge_samples = {}
        for sample in self.samples:
            if sample.judge_name not in judge_samples:
                judge_samples[sample.judge_name] = []
            judge_samples[sample.judge_name].append(sample)

        for judge_name, samples in judge_samples.items():
            if len(samples) < 10:
                logger.warning(f"Insufficient samples for {judge_name}: {len(samples)}")
                continue

            try:
                if method == "isotonic":
                    result = self._train_isotonic_regression(judge_name, samples)
                elif method == "logistic":
                    result = self._train_logistic_regression(judge_name, samples)
                else:
                    logger.error(f"Unknown calibration method: {method}")
                    continue

                self.calibration_models[judge_name] = result
                logger.info(f"Trained {method} calibration for {judge_name}")

            except Exception as e:
                logger.error(f"Failed to train calibration for {judge_name}: {e}")

    def _train_isotonic_regression(self, judge_name: str, samples: List[CalibrationSample]) -> CalibrationResult:
        """Train isotonic regression calibration model."""
        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.metrics import brier_score_loss
            import numpy as np

            # Prepare data
            confidences = np.array([s.judge_confidence for s in samples])
            labels = np.array([1.0 if s.human_correct else 0.0 for s in samples])

            # Train isotonic regression
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(confidences, labels)

            # Create calibration curve
            conf_range = np.linspace(0, 1, 100)
            calibrated_probs = iso_reg.predict(conf_range)
            calibration_curve = list(zip(conf_range, calibrated_probs))

            # Calculate reliability score (1 - Brier score)
            predictions = iso_reg.predict(confidences)
            brier_score = brier_score_loss(labels, predictions)
            reliability_score = 1.0 - brier_score

            return CalibrationResult(
                judge_name=judge_name,
                calibration_method="isotonic",
                reliability_score=max(0.0, min(1.0, reliability_score)),
                samples_used=len(samples),
                calibration_curve=calibration_curve,
                validation_metrics={
                    'brier_score': brier_score,
                    'mean_absolute_error': np.mean(np.abs(predictions - labels))
                }
            )

        except ImportError:
            logger.error("scikit-learn not available for isotonic regression")
            raise
        except Exception as e:
            logger.error(f"Isotonic regression training failed: {e}")
            raise

    def _train_logistic_regression(self, judge_name: str, samples: List[CalibrationSample]) -> CalibrationResult:
        """Train logistic regression calibration model."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import brier_score_loss
            import numpy as np

            # Prepare data
            confidences = np.array([s.judge_confidence for s in samples]).reshape(-1, 1)
            labels = np.array([1.0 if s.human_correct else 0.0 for s in samples])

            # Train logistic regression
            log_reg = LogisticRegression()
            log_reg.fit(confidences, labels)

            # Create calibration curve
            conf_range = np.linspace(0, 1, 100).reshape(-1, 1)
            calibrated_probs = log_reg.predict_proba(conf_range)[:, 1]
            calibration_curve = list(zip(conf_range.flatten(), calibrated_probs))

            # Calculate reliability score
            predictions = log_reg.predict_proba(confidences)[:, 1]
            brier_score = brier_score_loss(labels, predictions)
            reliability_score = 1.0 - brier_score

            return CalibrationResult(
                judge_name=judge_name,
                calibration_method="logistic",
                reliability_score=max(0.0, min(1.0, reliability_score)),
                samples_used=len(samples),
                calibration_curve=calibration_curve,
                validation_metrics={
                    'brier_score': brier_score,
                    'mean_absolute_error': np.mean(np.abs(predictions - labels))
                }
            )

        except ImportError:
            logger.error("scikit-learn not available for logistic regression")
            raise
        except Exception as e:
            logger.error(f"Logistic regression training failed: {e}")
            raise

    def calibrate_confidence(self, judge_name: str, confidence: float) -> float:
        """Calibrate a confidence score using the trained model."""
        if judge_name not in self.calibration_models:
            logger.warning(f"No calibration model for {judge_name}, returning original confidence")
            return confidence

        model = self.calibration_models[judge_name]

        # Find closest point on calibration curve
        closest_point = min(model.calibration_curve,
                          key=lambda x: abs(x[0] - confidence))

        return closest_point[1]

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get statistics about the calibration system."""
        if not self.samples:
            return {"status": "no_samples"}

        judge_stats = {}
        for judge_name in set(s.judge_name for s in self.samples):
            judge_samples = [s for s in self.samples if s.judge_name == judge_name]
            calibrated = judge_name in self.calibration_models

            judge_stats[judge_name] = {
                "samples": len(judge_samples),
                "calibrated": calibrated,
                "reliability_score": (
                    self.calibration_models[judge_name].reliability_score
                    if calibrated else None
                )
            }

        return {
            "total_samples": len(self.samples),
            "judges": judge_stats,
            "calibration_methods": list(set(
                model.calibration_method
                for model in self.calibration_models.values()
            ))
        }

    def save_calibration_models(self):
        """Save trained calibration models to disk."""
        models_file = os.path.join(self.calibration_dir, "calibration_models.json")

        try:
            models_data = {}
            for judge_name, result in self.calibration_models.items():
                models_data[judge_name] = asdict(result)

            with open(models_file, 'w') as f:
                json.dump(models_data, f, indent=2)

            logger.info(f"Saved calibration models for {len(self.calibration_models)} judges")

        except Exception as e:
            logger.error(f"Failed to save calibration models: {e}")

    def load_calibration_models(self):
        """Load trained calibration models from disk."""
        models_file = os.path.join(self.calibration_dir, "calibration_models.json")

        if not os.path.exists(models_file):
            return

        try:
            with open(models_file, 'r') as f:
                models_data = json.load(f)

            for judge_name, data in models_data.items():
                result = CalibrationResult(**data)
                self.calibration_models[judge_name] = result

            logger.info(f"Loaded calibration models for {len(self.calibration_models)} judges")

        except Exception as e:
            logger.error(f"Failed to load calibration models: {e}")