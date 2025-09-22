"""
Calibration module for mapping LLM judge confidences to human truth probabilities.
"""

import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Data point for calibration."""

    model_name: str
    confidence: float
    human_truth: bool
    problem_id: str
    judge_response: Dict[str, Any]


@dataclass
class CalibrationResult:
    """Result of calibration process."""

    model_name: str
    calibration_method: str
    reliability_weight: float
    brier_score: float
    log_loss_score: float
    calibration_curve: List[Tuple[float, float]]
    is_calibrated: bool


@dataclass
class ModelReliability:
    """Model reliability metrics."""

    model_name: str
    reliability_weight: float
    calibration_result: Optional[CalibrationResult]
    sample_count: int
    accuracy: float
    confidence_correlation: float


class Calibration:
    """Calibration manager for LLM judge confidences."""

    def __init__(
        self,
        calibration_data_path: str = "calibration_data.json",
        models_dir: str = "calibration_models",
    ):
        """Initialize calibration manager.

        Args:
            calibration_data_path: Path to store calibration data
            models_dir: Directory to store calibration models
        """
        self.calibration_data_path = calibration_data_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.calibration_data: List[CalibrationData] = []
        self.calibrated_models: Dict[str, CalibrationResult] = {}
        self.model_reliability: Dict[str, ModelReliability] = {}
        self._load_calibration_data()

    def _load_calibration_data(self):
        """Load existing calibration data."""
        if os.path.exists(self.calibration_data_path):
            try:
                with open(self.calibration_data_path, "r") as f:
                    data = json.load(f)
                    self.calibration_data = [CalibrationData(**item) for item in data]
                logger.info(
                    f"Loaded {len(self.calibration_data)} calibration data points"
                )
            except Exception as e:
                logger.error(f"Failed to load calibration data: {e}")
                self.calibration_data = []

    def _save_calibration_data(self):
        """Save calibration data to file."""
        try:
            data = [asdict(item) for item in self.calibration_data]
            with open(self.calibration_data_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.calibration_data)} calibration data points")
        except Exception as e:
            logger.error(f"Failed to save calibration data: {e}")

    def add_calibration_data(
        self,
        model_name: str,
        confidence: float,
        human_truth: bool,
        problem_id: str,
        judge_response: Dict[str, Any],
    ):
        """Add a new calibration data point.

        Args:
            model_name: Name of the model
            confidence: Confidence score from the model
            human_truth: Human annotation (True/False)
            problem_id: ID of the problem
            judge_response: Full judge response data
        """
        data_point = CalibrationData(
            model_name=model_name,
            confidence=confidence,
            human_truth=human_truth,
            problem_id=problem_id,
            judge_response=judge_response,
        )
        self.calibration_data.append(data_point)
        self._save_calibration_data()

    def sample_for_human_annotation(
        self, model_name: str, n_samples: int = 50
    ) -> List[Dict[str, Any]]:
        """Sample cases for human annotation.

        Args:
            model_name: Name of the model to sample from
            n_samples: Number of samples to return

        Returns:
            List of samples for human annotation
        """
        model_data = [d for d in self.calibration_data if d.model_name == model_name]

        if len(model_data) < n_samples:
            return [asdict(d) for d in model_data]

        # Stratified sampling by confidence levels
        low_conf = [d for d in model_data if d.confidence < 0.3]
        mid_conf = [d for d in model_data if 0.3 <= d.confidence <= 0.7]
        high_conf = [d for d in model_data if d.confidence > 0.7]

        samples = []
        samples_per_bucket = n_samples // 3

        for bucket in [low_conf, mid_conf, high_conf]:
            if len(bucket) > 0:
                bucket_samples = min(samples_per_bucket, len(bucket))
                samples.extend(bucket[:bucket_samples])

        return [asdict(sample) for sample in samples]

    def calibrate_model(
        self, model_name: str, method: str = "isotonic"
    ) -> CalibrationResult:
        """Calibrate a model's confidence scores.

        Args:
            model_name: Name of the model to calibrate
            method: Calibration method ('isotonic' or 'logistic')

        Returns:
            Calibration result
        """
        model_data = [d for d in self.calibration_data if d.model_name == model_name]

        if len(model_data) < 10:
            logger.warning(
                f"Insufficient data for calibration: {len(model_data)} samples"
            )
            return CalibrationResult(
                model_name=model_name,
                calibration_method=method,
                reliability_weight=1.0,
                brier_score=0.5,
                log_loss_score=float("inf"),
                calibration_curve=[],
                is_calibrated=False,
            )

        # Prepare data
        confidences = np.array([d.confidence for d in model_data])
        truths = np.array([d.human_truth for d in model_data])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            confidences.reshape(-1, 1), truths, test_size=0.2, random_state=42
        )

        # Fit calibration model
        if method == "isotonic":
            calibrator = CalibratedClassifierCV(
                base_estimator=LogisticRegression(), method="isotonic", cv=3
            )
        else:  # logistic
            calibrator = CalibratedClassifierCV(
                base_estimator=LogisticRegression(), method="sigmoid", cv=3
            )

        try:
            calibrator.fit(X_train, y_train)

            # Evaluate
            y_pred_proba = calibrator.predict_proba(X_test)[:, 1]
            brier_score = brier_score_loss(y_test, y_pred_proba)
            log_loss_score = log_loss(y_test, y_pred_proba)

            # Generate calibration curve
            calibration_curve = self._generate_calibration_curve(
                calibrator, confidences, truths
            )

            # Calculate reliability weight
            accuracy = np.mean(y_test == (y_pred_proba > 0.5))
            reliability_weight = min(2.0, max(0.1, accuracy * 2))

            result = CalibrationResult(
                model_name=model_name,
                calibration_method=method,
                reliability_weight=reliability_weight,
                brier_score=brier_score,
                log_loss_score=log_loss_score,
                calibration_curve=calibration_curve,
                is_calibrated=True,
            )

            # Save model
            model_path = self.models_dir / f"{model_name}_{method}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(calibrator, f)

            self.calibrated_models[f"{model_name}_{method}"] = result
            logger.info(
                f"Calibrated {model_name} with {method}: weight={reliability_weight:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Calibration failed for {model_name}: {e}")
            return CalibrationResult(
                model_name=model_name,
                calibration_method=method,
                reliability_weight=1.0,
                brier_score=0.5,
                log_loss_score=float("inf"),
                calibration_curve=[],
                is_calibrated=False,
            )

    def _generate_calibration_curve(
        self, calibrator, confidences: np.ndarray, truths: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Generate calibration curve points."""
        try:
            from sklearn.calibration import calibration_curve

            prob_true, prob_pred = calibration_curve(truths, confidences, n_bins=10)
            return list(zip(prob_pred, prob_true))
        except Exception as e:
            logger.warning(f"Failed to generate calibration curve: {e}")
            return []

    def calibrate_confidence(
        self, model_name: str, confidence: float, method: str = "isotonic"
    ) -> float:
        """Calibrate a single confidence score.

        Args:
            model_name: Name of the model
            confidence: Raw confidence score
            method: Calibration method

        Returns:
            Calibrated confidence score
        """
        model_key = f"{model_name}_{method}"

        if model_key not in self.calibrated_models:
            # Try to load existing model
            model_path = self.models_dir / f"{model_name}_{method}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        calibrator = pickle.load(f)
                    calibrated_prob = calibrator.predict_proba([[confidence]])[0, 1]
                    return float(calibrated_prob)
                except Exception as e:
                    logger.error(f"Failed to load calibration model: {e}")

            # If no model exists, return original confidence
            return confidence

        # Use in-memory model
        try:
            model_path = self.models_dir / f"{model_name}_{method}.pkl"
            with open(model_path, "rb") as f:
                calibrator = pickle.load(f)
            calibrated_prob = calibrator.predict_proba([[confidence]])[0, 1]
            return float(calibrated_prob)
        except Exception as e:
            logger.error(f"Failed to calibrate confidence: {e}")
            return confidence

    def calculate_model_reliability(self) -> Dict[str, ModelReliability]:
        """Calculate reliability weights for all models.

        Returns:
            Dictionary mapping model names to reliability metrics
        """
        model_names = list(set(d.model_name for d in self.calibration_data))

        for model_name in model_names:
            model_data = [
                d for d in self.calibration_data if d.model_name == model_name
            ]

            if len(model_data) < 5:
                # Insufficient data
                self.model_reliability[model_name] = ModelReliability(
                    model_name=model_name,
                    reliability_weight=1.0,
                    calibration_result=None,
                    sample_count=len(model_data),
                    accuracy=0.5,
                    confidence_correlation=0.0,
                )
                continue

            # Calculate metrics
            confidences = np.array([d.confidence for d in model_data])
            truths = np.array([d.human_truth for d in model_data])

            accuracy = np.mean(truths == (confidences > 0.5))
            correlation = np.corrcoef(confidences, truths.astype(float))[0, 1]

            # Get calibration result
            calibration_result = None
            for key, result in self.calibrated_models.items():
                if result.model_name == model_name:
                    calibration_result = result
                    break

            reliability_weight = 1.0
            if calibration_result and calibration_result.is_calibrated:
                reliability_weight = calibration_result.reliability_weight

            self.model_reliability[model_name] = ModelReliability(
                model_name=model_name,
                reliability_weight=reliability_weight,
                calibration_result=calibration_result,
                sample_count=len(model_data),
                accuracy=accuracy,
                confidence_correlation=correlation,
            )

        return self.model_reliability

    def get_model_weight(self, model_name: str) -> float:
        """Get reliability weight for a model.

        Args:
            model_name: Name of the model

        Returns:
            Reliability weight
        """
        if model_name in self.model_reliability:
            return self.model_reliability[model_name].reliability_weight
        return 1.0

    def export_calibration_report(self, output_path: str):
        """Export calibration report to JSON.

        Args:
            output_path: Path to save the report
        """
        report = {
            "calibration_data_count": len(self.calibration_data),
            "calibrated_models": {
                key: asdict(result) for key, result in self.calibrated_models.items()
            },
            "model_reliability": {
                key: asdict(reliability)
                for key, reliability in self.model_reliability.items()
            },
            "summary": {
                "total_models": len(self.model_reliability),
                "calibrated_models": len(
                    [r for r in self.calibrated_models.values() if r.is_calibrated]
                ),
                "average_reliability": np.mean(
                    [r.reliability_weight for r in self.model_reliability.values()]
                ),
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported calibration report to {output_path}")
