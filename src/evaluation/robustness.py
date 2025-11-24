"""
Robustness evaluation metrics for EmoBench.

Provides out-of-distribution detection, uncertainty quantification,
and fairness analysis for model robustness assessment.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class RobustnessEvaluator:
    """
    Evaluate model robustness with OOD detection and uncertainty metrics.

    Examples:
        >>> evaluator = RobustnessEvaluator()
        >>> ood_metrics = evaluator.evaluate_ood_detection(predictions, confidences, labels)
        >>> uncertainty = evaluator.uncertainty_quantification(predictions, probabilities)
    """

    def __init__(self, ood_threshold: float = 0.5):
        """
        Initialize robustness evaluator.

        Args:
            ood_threshold: Threshold for OOD detection (0-1)
        """
        self.ood_threshold = ood_threshold

    def evaluate_ood_detection(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        labels: np.ndarray,
        ood_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate out-of-distribution detection performance.

        Args:
            predictions: Model predictions
            confidences: Prediction confidences (0-1)
            labels: True labels (0 = ID, 1 = OOD if ood_labels provided)
            ood_labels: Optional binary labels for OOD vs ID

        Returns:
            Dict[str, float]: OOD detection metrics
        """
        metrics = {}

        # If no OOD labels provided, use confidence-based detection
        if ood_labels is None:
            # Simple threshold-based OOD detection
            ood_predictions = (confidences < self.ood_threshold).astype(int)
            # For evaluation, we need ground truth - this is just a placeholder
            logger.warning("No OOD labels provided, using confidence threshold only")
            metrics["confidence_threshold"] = self.ood_threshold
            metrics["low_confidence_ratio"] = np.mean(confidences < self.ood_threshold)
            return metrics

        # Binary classification evaluation for OOD detection
        try:
            # OOD detection as binary classification (0 = ID, 1 = OOD)
            ood_auc = roc_auc_score(ood_labels, 1 - confidences)  # Higher confidence = more ID-like
            metrics["ood_auc"] = ood_auc

            # FPR at 95% TPR (standard OOD metric)
            fpr_at_95_tpr = self._calculate_fpr_at_tpr(confidences, ood_labels, target_tpr=0.95)
            metrics["fpr_at_95_tpr"] = fpr_at_95_tpr

            # Detection accuracy at threshold
            ood_predictions = (1 - confidences < self.ood_threshold).astype(int)
            detection_accuracy = np.mean(ood_predictions == ood_labels)
            metrics["detection_accuracy"] = detection_accuracy

        except Exception as e:
            logger.warning(f"Could not compute OOD metrics: {e}")
            metrics["error"] = str(e)

        return metrics

    def uncertainty_quantification(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        method: str = "entropy",
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Quantify prediction uncertainty.

        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities (shape: n_samples x n_classes)
            method: Uncertainty quantification method ("entropy", "confidence", "variance")

        Returns:
            Dict: Uncertainty metrics
        """
        metrics = {}

        if method == "entropy":
            # Shannon entropy
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
            metrics["mean_entropy"] = float(np.mean(entropy))
            metrics["entropy_uncertainty"] = entropy

        elif method == "confidence":
            # 1 - max probability
            max_probs = np.max(probabilities, axis=1)
            uncertainty = 1 - max_probs
            metrics["mean_confidence_uncertainty"] = float(np.mean(uncertainty))
            metrics["confidence_uncertainty"] = uncertainty

        elif method == "variance":
            # Variance of probabilities
            variance = np.var(probabilities, axis=1)
            metrics["mean_variance"] = float(np.mean(variance))
            metrics["variance_uncertainty"] = variance

        # Calibration metrics
        try:
            ece = self._expected_calibration_error(predictions, probabilities)
            metrics["expected_calibration_error"] = ece
        except Exception as e:
            logger.debug(f"Could not compute ECE: {e}")

        return metrics

    def fairness_analysis(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attributes: np.ndarray,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Union[float, str, Dict]]:
        """
        Analyze fairness across demographic groups.

        Args:
            predictions: Model predictions
            labels: True labels
            protected_attributes: Protected attribute values (e.g., gender, race)
            metrics: Fairness metrics to compute

        Returns:
            Dict[str, float]: Fairness metrics
        """
        if metrics is None:
            metrics = ["demographic_parity", "equal_opportunity", "accuracy_parity"]

        fairness_results = {}

        unique_groups = np.unique(protected_attributes)

        if len(unique_groups) < 2:
            logger.warning("Need at least 2 groups for fairness analysis")
            return {"error": "Insufficient groups for fairness analysis", "group_metrics": {}}

        group_metrics = {}

        for group in unique_groups:
            mask = protected_attributes == group
            if np.sum(mask) == 0:
                continue

            group_preds = predictions[mask]
            group_labels = labels[mask]

            # Basic metrics per group
            accuracy = np.mean(group_preds == group_labels)
            tpr = np.mean(group_preds[group_labels == 1]) if np.any(group_labels == 1) else 0
            fpr = np.mean(group_preds[group_labels == 0] == 1) if np.any(group_labels == 0) else 0

            group_metrics[group] = {
                "accuracy": accuracy,
                "tpr": tpr,
                "fpr": fpr,
                "size": len(group_preds),
            }

        # Compute fairness disparities
        if "demographic_parity" in metrics:
            # Difference in positive prediction rates
            pos_rates = [group_metrics[g]["tpr"] for g in unique_groups if g in group_metrics]
            if len(pos_rates) > 1:
                fairness_results["demographic_parity_diff"] = max(pos_rates) - min(pos_rates)

        if "equal_opportunity" in metrics:
            # Difference in true positive rates
            tpr_values = [group_metrics[g]["tpr"] for g in unique_groups if g in group_metrics]
            if len(tpr_values) > 1:
                fairness_results["equal_opportunity_diff"] = max(tpr_values) - min(tpr_values)

        if "accuracy_parity" in metrics:
            # Difference in accuracy
            acc_values = [group_metrics[g]["accuracy"] for g in unique_groups if g in group_metrics]
            if len(acc_values) > 1:
                fairness_results["accuracy_parity_diff"] = max(acc_values) - min(acc_values)

        fairness_results["group_metrics"] = group_metrics
        return fairness_results

    def adversarial_robustness_test(
        self,
        model,
        test_data: np.ndarray,
        labels: np.ndarray,
        epsilon: float = 0.1,
        num_steps: int = 10,
    ) -> Dict[str, Union[float, str]]:
        """
        Test adversarial robustness using Fast Gradient Sign Method.

        Args:
            model: PyTorch model to test
            test_data: Test input data
            labels: True labels
            epsilon: Perturbation magnitude
            num_steps: Number of adversarial steps

        Returns:
            Dict[str, float]: Adversarial robustness metrics
        """
        try:
            import torch
            import torch.nn.functional as F

            model.eval()
            device = next(model.parameters()).device

            # Convert to tensors
            x = torch.tensor(test_data, dtype=torch.float32).to(device)
            y = torch.tensor(labels, dtype=torch.long).to(device)

            # Original predictions
            with torch.no_grad():
                orig_logits = model(x)
                orig_preds = torch.argmax(orig_logits, dim=1)

            # Generate adversarial examples (FGSM)
            x_adv = x.clone().detach().requires_grad_(True)
            for _ in range(num_steps):
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y)
                loss.backward()

                with torch.no_grad():
                    grad_sign = x_adv.grad.sign()
                    x_adv = x_adv + epsilon * grad_sign
                    x_adv = torch.clamp(x_adv, 0, 1)  # Assuming normalized inputs
                    x_adv.grad.zero_()

            # Adversarial predictions
            with torch.no_grad():
                adv_logits = model(x_adv)
                adv_preds = torch.argmax(adv_logits, dim=1)

            # Calculate metrics
            orig_accuracy = (orig_preds == y).float().mean().item()
            adv_accuracy = (adv_preds == y).float().mean().item()
            robust_accuracy = adv_accuracy  # Accuracy on adversarial examples
            attack_success_rate = 1 - (adv_preds == orig_preds).float().mean().item()

            return {
                "original_accuracy": orig_accuracy,
                "adversarial_accuracy": adv_accuracy,
                "robust_accuracy": robust_accuracy,
                "attack_success_rate": attack_success_rate,
                "accuracy_drop": orig_accuracy - adv_accuracy,
            }

        except ImportError:
            logger.warning("PyTorch not available for adversarial testing")
            return {"error": "PyTorch required for adversarial robustness testing"}
        except Exception as e:
            logger.error(f"Adversarial robustness test failed: {e}")
            return {"error": str(e)}

    def _calculate_fpr_at_tpr(
        self, confidences: np.ndarray, ood_labels: np.ndarray, target_tpr: float = 0.95
    ) -> float:
        """Calculate False Positive Rate at target True Positive Rate."""
        # Sort by confidence (ascending - lower confidence = more OOD-like)
        sorted_indices = np.argsort(confidences)
        sorted_ood_labels = ood_labels[sorted_indices]

        # Calculate TPR and FPR at different thresholds
        n_total = len(ood_labels)
        n_ood = np.sum(ood_labels)
        n_id = n_total - n_ood

        min_fpr = 1.0

        for i in range(1, n_total):
            # Threshold: reject top i samples as OOD
            threshold_idx = n_total - i
            predicted_ood = np.zeros(n_total)
            predicted_ood[sorted_indices[:threshold_idx]] = 1

            # Calculate TPR and FPR
            tpr = np.sum((predicted_ood == 1) & (ood_labels == 1)) / n_ood
            fpr = np.sum((predicted_ood == 1) & (ood_labels == 0)) / n_id

            if tpr >= target_tpr:
                min_fpr = min(min_fpr, fpr)

        return min_fpr

    def _expected_calibration_error(
        self, predictions: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        max_probs = np.max(probabilities, axis=1)

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(max_probs, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        total_samples = len(predictions)

        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            if not np.any(bin_mask):
                continue

            bin_size = np.sum(bin_mask)
            bin_confidence = np.mean(max_probs[bin_mask])
            bin_accuracy = np.mean(predictions[bin_mask] == np.argmax(probabilities[bin_mask], axis=1))

            ece += (bin_size / total_samples) * abs(bin_confidence - bin_accuracy)

        return ece


def evaluate_model_robustness(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    confidences: Optional[np.ndarray] = None,
    protected_attributes: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Comprehensive robustness evaluation.

    Args:
        predictions: Model predictions
        labels: True labels
        probabilities: Prediction probabilities
        confidences: Prediction confidences
        protected_attributes: Protected attributes for fairness analysis

    Returns:
        Dict[str, float]: Comprehensive robustness metrics
    """
    evaluator = RobustnessEvaluator()
    results = {}

    # Uncertainty quantification
    if probabilities is not None:
        uncertainty = evaluator.uncertainty_quantification(predictions, probabilities)
        results.update(uncertainty)

    # OOD detection
    if confidences is not None:
        ood_metrics = evaluator.evaluate_ood_detection(predictions, confidences, labels)
        results.update(ood_metrics)

    # Fairness analysis
    if protected_attributes is not None:
        fairness = evaluator.fairness_analysis(predictions, labels, protected_attributes)
        results.update(fairness)

    return results</content>
<parameter name="filePath">src/evaluation/robustness.py