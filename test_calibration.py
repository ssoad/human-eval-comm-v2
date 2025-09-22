"""
Unit tests for Calibration module.
"""

import json
import os
import tempfile
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

from evaluators.calibration import (
    Calibration,
    CalibrationData,
    CalibrationResult,
    ModelReliability,
)


class TestCalibrationData:
    """Test cases for CalibrationData dataclass."""

    def test_calibration_data_creation(self):
        """Test CalibrationData creation."""
        data = CalibrationData(
            model_name="test-model",
            confidence=0.85,
            human_truth=True,
            problem_id="problem_1",
            judge_response={"score": 8.5, "rationale": "Good"},
        )

        assert data.model_name == "test-model"
        assert data.confidence == 0.85
        assert data.human_truth is True
        assert data.problem_id == "problem_1"
        assert data.judge_response["score"] == 8.5


class TestCalibrationResult:
    """Test cases for CalibrationResult dataclass."""

    def test_calibration_result_creation(self):
        """Test CalibrationResult creation."""
        result = CalibrationResult(
            model_name="test-model",
            calibration_method="isotonic",
            reliability_weight=0.8,
            brier_score=0.15,
            log_loss_score=0.45,
            calibration_curve=[(0.1, 0.1), (0.9, 0.9)],
            is_calibrated=True,
        )

        assert result.model_name == "test-model"
        assert result.calibration_method == "isotonic"
        assert result.reliability_weight == 0.8
        assert result.brier_score == 0.15
        assert result.is_calibrated is True


class TestModelReliability:
    """Test cases for ModelReliability dataclass."""

    def test_model_reliability_creation(self):
        """Test ModelReliability creation."""
        reliability = ModelReliability(
            model_name="test-model",
            reliability_weight=0.85,
            calibration_result=None,
            sample_count=100,
            accuracy=0.92,
            confidence_correlation=0.78,
        )

        assert reliability.model_name == "test-model"
        assert reliability.reliability_weight == 0.85
        assert reliability.sample_count == 100
        assert reliability.accuracy == 0.92
        assert reliability.confidence_correlation == 0.78


class TestCalibration:
    """Test cases for Calibration."""

    @pytest.fixture
    def calibration(self):
        """Create Calibration instance with temporary files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "calibration_data.json")
            models_dir = os.path.join(temp_dir, "models")

            cal = Calibration(data_path, models_dir)
            yield cal

    def test_init_new_calibration(self, calibration):
        """Test initialization of new calibration."""
        assert len(calibration.calibration_data) == 0
        assert len(calibration.calibrated_models) == 0
        assert len(calibration.model_reliability) == 0
        assert os.path.exists(calibration.models_dir)

    def test_init_with_existing_data(self):
        """Test initialization with existing calibration data."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = [
                {
                    "model_name": "test-model",
                    "confidence": 0.8,
                    "human_truth": True,
                    "problem_id": "problem_1",
                    "judge_response": {"score": 8.0},
                }
            ]
            json.dump(test_data, f)
            data_path = f.name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                models_dir = os.path.join(temp_dir, "models")
                cal = Calibration(data_path, models_dir)

                assert len(cal.calibration_data) == 1
                assert cal.calibration_data[0].model_name == "test-model"
                assert cal.calibration_data[0].confidence == 0.8
        finally:
            os.unlink(data_path)

    def test_add_calibration_data(self, calibration):
        """Test adding calibration data."""
        calibration.add_calibration_data(
            model_name="test-model",
            confidence=0.85,
            human_truth=True,
            problem_id="problem_1",
            judge_response={"score": 8.5},
        )

        assert len(calibration.calibration_data) == 1
        data = calibration.calibration_data[0]
        assert data.model_name == "test-model"
        assert data.confidence == 0.85
        assert data.human_truth is True

    def test_sample_for_human_annotation(self, calibration):
        """Test sampling for human annotation."""
        # Add some test data with varying confidence
        for i in range(5):
            calibration.add_calibration_data(
                model_name=f"model-{i % 2}",
                confidence=0.5 + i * 0.1,
                human_truth=True,
                problem_id=f"problem_{i}",
                judge_response={"score": 7.0 + i},
            )

        samples = calibration.sample_for_human_annotation(
            n_samples=3, min_confidence_diff=0.2
        )

        assert len(samples) <= 3
        for sample in samples:
            assert "problem_id" in sample
            assert "confidence_std" in sample
            assert "requires_annotation" in sample
            assert sample["requires_annotation"] is True

    def test_sample_for_human_annotation_insufficient_data(self, calibration):
        """Test sampling with insufficient data."""
        # Add minimal data
        calibration.add_calibration_data(
            model_name="test-model",
            confidence=0.8,
            human_truth=True,
            problem_id="problem_1",
            judge_response={"score": 8.0},
        )

        samples = calibration.sample_for_human_annotation(n_samples=10)

        assert len(samples) == 0

    @patch("sklearn.isotonic.IsotonicRegression")
    def test_calibrate_model_insufficient_data(self, mock_isotonic, calibration):
        """Test calibration with insufficient data."""
        # Add minimal data
        calibration.add_calibration_data(
            model_name="test-model",
            confidence=0.8,
            human_truth=True,
            problem_id="problem_1",
            judge_response={"score": 8.0},
        )

        result = calibration.calibrate_model("test-model")

        assert result.model_name == "test-model"
        assert result.is_calibrated is False
        assert result.reliability_weight == 0.5
        mock_isotonic.assert_not_called()

    @patch("sklearn.isotonic.IsotonicRegression")
    def test_calibrate_model_success(self, mock_isotonic, calibration):
        """Test successful model calibration."""
        # Add sufficient data
        for i in range(20):
            calibration.add_calibration_data(
                model_name="test-model",
                confidence=0.5 + (i % 10) * 0.05,
                human_truth=i % 3 != 0,  # Some false cases
                problem_id=f"problem_{i}",
                judge_response={"score": 7.0 + (i % 3)},
            )

        # Mock the calibrator
        mock_calibrator = Mock()
        mock_calibrator.fit.return_value = None
        mock_calibrator.predict.return_value = np.array([0.7, 0.8, 0.9])
        mock_isotonic.return_value = mock_calibrator

        # Mock sklearn metrics
        with patch("sklearn.metrics.brier_score_loss", return_value=0.2), patch(
            "sklearn.metrics.log_loss", return_value=0.5
        ), patch.object(calibration, "_save_calibration_model"):

            result = calibration.calibrate_model("test-model", method="isotonic")

        assert result.model_name == "test-model"
        assert result.calibration_method == "isotonic"
        assert result.is_calibrated is True
        assert 0 <= result.reliability_weight <= 1
        mock_calibrator.fit.assert_called_once()

    def test_calibrate_confidence_no_model(self, calibration):
        """Test calibrating confidence with no calibration model."""
        confidence = calibration.calibrate_confidence("nonexistent-model", 0.8)

        assert confidence == 0.8  # Should return original confidence

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.load")
    def test_calibrate_confidence_with_model(
        self, mock_pickle_load, mock_file, calibration
    ):
        """Test calibrating confidence with existing model."""
        # Mock the calibrator
        mock_calibrator = Mock()
        mock_calibrator.predict.return_value = np.array([0.85])
        mock_pickle_load.return_value = mock_calibrator

        # Mock file exists
        with patch("pathlib.Path.exists", return_value=True):
            confidence = calibration.calibrate_confidence(
                "test-model", 0.8, method="isotonic"
            )

        assert confidence == 0.85
        mock_calibrator.predict.assert_called_once_with([0.8])

    def test_calibrate_all_models(self, calibration):
        """Test calibrating all models."""
        # Add data for multiple models
        for model_name in ["model1", "model2"]:
            for i in range(15):
                calibration.add_calibration_data(
                    model_name=model_name,
                    confidence=0.5 + i * 0.03,
                    human_truth=i % 2 == 0,
                    problem_id=f"problem_{i}",
                    judge_response={"score": 7.0},
                )

        with patch.object(calibration, "calibrate_model") as mock_calibrate:
            mock_calibrate.return_value = CalibrationResult(
                model_name="test",
                calibration_method="isotonic",
                reliability_weight=0.8,
                brier_score=0.2,
                log_loss_score=0.5,
                calibration_curve=[],
                is_calibrated=True,
            )

            results = calibration.calibrate_all_models()

        assert len(results) == 2
        assert "model1" in results
        assert "model2" in results

    def test_calculate_model_reliability(self, calibration):
        """Test calculating model reliability metrics."""
        # Add test data
        for i in range(10):
            calibration.add_calibration_data(
                model_name="test-model",
                confidence=0.7 + (i % 3) * 0.1,
                human_truth=i % 2 == 0,
                problem_id=f"problem_{i}",
                judge_response={"score": 8.0},
            )

        reliability = calibration.calculate_model_reliability()

        assert "test-model" in reliability
        model_reliability = reliability["test-model"]
        assert model_reliability.model_name == "test-model"
        assert model_reliability.sample_count == 10
        assert 0 <= model_reliability.accuracy <= 1
        assert -1 <= model_reliability.confidence_correlation <= 1

    def test_get_weighted_consensus(self, calibration):
        """Test calculating weighted consensus."""
        # Add some model reliability data
        calibration.model_reliability = {
            "model1": ModelReliability(
                model_name="model1",
                reliability_weight=0.8,
                calibration_result=None,
                sample_count=10,
                accuracy=0.9,
                confidence_correlation=0.7,
            ),
            "model2": ModelReliability(
                model_name="model2",
                reliability_weight=0.6,
                calibration_result=None,
                sample_count=10,
                accuracy=0.8,
                confidence_correlation=0.6,
            ),
        }

        judge_responses = [
            {"model_name": "model1", "score": 8.0, "confidence": 0.9},
            {"model_name": "model2", "score": 7.0, "confidence": 0.8},
        ]

        consensus = calibration.get_weighted_consensus(judge_responses)

        assert "consensus_score" in consensus
        assert "consensus_confidence" in consensus
        assert "total_weight" in consensus
        assert consensus["total_weight"] == 1.4  # 0.8 + 0.6
        assert 0 <= consensus["consensus_score"] <= 10
        assert 0 <= consensus["consensus_confidence"] <= 1

    def test_get_weighted_consensus_empty(self, calibration):
        """Test weighted consensus with empty responses."""
        consensus = calibration.get_weighted_consensus([])

        assert consensus["consensus_score"] == 0.0
        assert consensus["consensus_confidence"] == 0.0

    def test_export_calibration_report(self, calibration):
        """Test exporting calibration report."""
        # Add some test data
        calibration.add_calibration_data(
            model_name="test-model",
            confidence=0.8,
            human_truth=True,
            problem_id="problem_1",
            judge_response={"score": 8.0},
        )

        # Add model reliability
        calibration.model_reliability = {
            "test-model": ModelReliability(
                model_name="test-model",
                reliability_weight=0.8,
                calibration_result=None,
                sample_count=1,
                accuracy=1.0,
                confidence_correlation=0.5,
            )
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report_path = f.name

        try:
            report = calibration.export_calibration_report(report_path)

            assert "summary" in report
            assert "model_reliability" in report
            assert "calibration_results" in report
            assert report["summary"]["total_data_points"] == 1
            assert "test-model" in report["model_reliability"]

            # Check file was created
            assert os.path.exists(report_path)

        finally:
            os.unlink(report_path)

    def test_empty_calibration_result(self, calibration):
        """Test creating empty calibration result."""
        result = calibration._empty_calibration_result("test-model")

        assert result.model_name == "test-model"
        assert result.calibration_method == "none"
        assert result.is_calibrated is False
        assert result.reliability_weight == 0.5
        assert result.brier_score == 1.0
        assert result.log_loss_score == 1.0
        assert len(result.calibration_curve) == 0

    def test_generate_calibration_curve(self, calibration):
        """Test generating calibration curve."""
        confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        truths = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # Perfect calibration
        calibrated_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        curve = calibration._generate_calibration_curve(
            confidences, truths, calibrated_probs
        )

        assert len(curve) > 0
        for prop, accuracy in curve:
            assert 0 <= prop <= 1
            assert 0 <= accuracy <= 1

    def test_calculate_reliability_weight(self, calibration):
        """Test calculating reliability weight."""
        weight = calibration._calculate_reliability_weight(
            brier_score=0.2, log_loss_score=0.5, sample_count=50
        )

        assert 0.1 <= weight <= 1.0

        # Test with perfect scores
        perfect_weight = calibration._calculate_reliability_weight(
            brier_score=0.0, log_loss_score=0.0, sample_count=100
        )

        assert perfect_weight == 1.0
