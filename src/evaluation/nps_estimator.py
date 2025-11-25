"""
NPS (Net Promoter Score) estimation from sentiment analysis model predictions.

This module provides functionality to estimate NPS from binary sentiment predictions
by mapping prediction confidence scores to NPS categories.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def _evaluate_disneyland_dataset(models, dataset, test_data, checkpoints_dir, device, all_results):
    """Evaluate models on Disneyland dataset using all available checkpoints."""
    training_datasets = ["imdb", "sst2", "amazon", "yelp"]

    for model in models:
        for train_dataset in training_datasets:
            logger.info(f"Evaluating {model} (trained on {train_dataset}) on {dataset}")

            try:
                # Use specific checkpoint
                predictions, confidences = run_inference_with_checkpoint(
                    model, train_dataset, dataset, test_data, checkpoints_dir, device
                )

                # Extract actual ratings for accuracy
                actual_ratings = [
                    sample.get("original_rating", sample["label"]) for sample in test_data
                ]

                # Calculate NPS
                nps_result = calculate_nps_from_predictions(
                    predictions, confidences, actual_ratings
                )

                result = {
                    "model_name": model,
                    "training_dataset": train_dataset,
                    "test_dataset": dataset,
                    "timestamp": int(pd.Timestamp.now().timestamp()),
                    "total_samples": len(test_data),
                    "nps_score": nps_result["nps_score"],
                    "promoters_percent": nps_result["promoters_percent"],
                    "passives_percent": nps_result["passives_percent"],
                    "detractors_percent": nps_result["detractors_percent"],
                    "promoters_count": nps_result["promoters_count"],
                    "passives_count": nps_result["passives_count"],
                    "detractors_count": nps_result["detractors_count"],
                    "accuracy_percent": nps_result.get("accuracy_percent"),
                    "correct_predictions": nps_result.get("correct_predictions"),
                }

                all_results.append(result)
                logger.info(
                    f"NPS for {model} (trained on {train_dataset}) on {dataset}: {nps_result['nps_score']:.2f}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to evaluate {model} (trained on {train_dataset}) on {dataset}: {e}"
                )
                result = {
                    "model_name": model,
                    "training_dataset": train_dataset,
                    "test_dataset": dataset,
                    "timestamp": int(pd.Timestamp.now().timestamp()),
                    "total_samples": len(test_data),
                    "error": str(e),
                }
                all_results.append(result)


def _evaluate_standard_dataset(models, dataset, test_data, checkpoints_dir, device, all_results):
    """Evaluate models on standard datasets."""
    for model in models:
        logger.info(f"Evaluating model {model} on {dataset}")

        try:
            # Use fallback checkpoint logic
            predictions, confidences = run_inference(
                model, dataset, test_data, checkpoints_dir, device
            )

            # Extract actual ratings for accuracy
            actual_ratings = [
                sample.get("original_rating", sample["label"]) for sample in test_data
            ]

            # Calculate NPS
            nps_result = calculate_nps_from_predictions(predictions, confidences, actual_ratings)

            result = {
                "model_name": model,
                "training_dataset": dataset,  # For standard eval, training and test are same
                "test_dataset": dataset,
                "timestamp": int(pd.Timestamp.now().timestamp()),
                "total_samples": len(test_data),
                "nps_score": nps_result["nps_score"],
                "promoters_percent": nps_result["promoters_percent"],
                "passives_percent": nps_result["passives_percent"],
                "detractors_percent": nps_result["detractors_percent"],
                "promoters_count": nps_result["promoters_count"],
                "passives_count": nps_result["passives_count"],
                "detractors_count": nps_result["detractors_count"],
                "accuracy_percent": nps_result.get("accuracy_percent"),
                "correct_predictions": nps_result.get("correct_predictions"),
            }

            all_results.append(result)
            logger.info(f"NPS for {model} on {dataset}: {nps_result['nps_score']:.2f}")

        except Exception as e:
            logger.error(f"Failed to evaluate {model} on {dataset}: {e}")
            result = {
                "model_name": model,
                "training_dataset": dataset,
                "test_dataset": dataset,
                "timestamp": int(pd.Timestamp.now().timestamp()),
                "total_samples": len(test_data),
                "error": str(e),
            }
            all_results.append(result)


def estimate_nps_scores(
    models: List[str],
    datasets: List[str],
    checkpoints_dir: str,
    output_dir: str,
    device: str = "auto",
):
    """
    Estimate NPS scores for models on test datasets.

    For Disneyland dataset, evaluates all available checkpoints (trained on different datasets)
    to show cross-dataset performance.

    Args:
        models: List of model names to evaluate
        datasets: List of dataset names to evaluate on
        checkpoints_dir: Directory containing trained model checkpoints
        output_dir: Directory to save NPS results
        device: Device to use for inference
    """
    logger.info(f"Starting NPS estimation for {len(models)} models on {len(datasets)} datasets")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")

        # Load test data for this dataset
        test_data = load_test_data(dataset)
        if test_data is None:
            logger.warning(f"Could not load test data for {dataset}, skipping")
            continue

        if dataset == "disneyland":
            _evaluate_disneyland_dataset(
                models, dataset, test_data, checkpoints_dir, device, all_results
            )
        else:
            _evaluate_standard_dataset(
                models, dataset, test_data, checkpoints_dir, device, all_results
            )

    # Save results
    if all_results:
        # Save as JSON
        json_file = output_path / "nps_results.json"
        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"NPS results saved to {json_file}")

        # Save as CSV for easier analysis
        df = pd.DataFrame(all_results)
        csv_file = output_path / "nps_results.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"NPS results also saved as CSV to {csv_file}")

    return all_results


def load_test_data(dataset_name: str) -> List[Dict[str, Any]] | None:
    """
    Load test data for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        List of test samples with text and labels
    """
    try:
        from datasets import load_dataset

        # Load dataset configuration
        import yaml

        config_path = Path("config/datasets.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if dataset_name not in config["datasets"]:
            raise ValueError(f"Dataset {dataset_name} not found in config")

        dataset_config = config["datasets"][dataset_name]

        # Load raw dataset
        if dataset_config.get("source") == "kaggle":
            # Handle Kaggle datasets with local caching
            try:
                import kagglehub
                from datasets import Dataset
                import pandas as pd
                import os

                # Create local cache directory
                cache_dir = Path("data/raw")
                cache_dir.mkdir(parents=True, exist_ok=True)

                dataset_slug = dataset_config["dataset_id"].replace("/", "_")
                local_cache_path = cache_dir / f"{dataset_slug}.csv"

                if local_cache_path.exists():
                    logger.info(f"Loading cached Kaggle dataset from: {local_cache_path}")
                    # Load from local cache
                    try:
                        df = pd.read_csv(local_cache_path, encoding="utf-8")
                    except UnicodeDecodeError:
                        df = pd.read_csv(local_cache_path, encoding="latin-1")
                else:
                    # Download and cache
                    logger.info(f"Downloading Kaggle dataset: {dataset_config['dataset_id']}")
                    dataset_path = kagglehub.dataset_download(dataset_config["dataset_id"])

                    # Find the CSV file
                    csv_files = []
                    for root, dirs, files in os.walk(dataset_path):
                        for file in files:
                            if file.endswith(".csv"):
                                csv_files.append(os.path.join(root, file))

                    if not csv_files:
                        raise FileNotFoundError("No CSV files found in Kaggle dataset")

                    csv_path = csv_files[0]
                    logger.info(f"Loading CSV file: {csv_path}")

                    # Load CSV
                    try:
                        df = pd.read_csv(csv_path, encoding="utf-8")
                    except UnicodeDecodeError:
                        df = pd.read_csv(csv_path, encoding="latin-1")

                    # Cache locally
                    df.to_csv(local_cache_path, index=False)
                    logger.info(f"Cached dataset to: {local_cache_path}")

                # Convert to HuggingFace Dataset format
                dataset = Dataset.from_pandas(df)

            except ImportError as e:
                raise ImportError(f"kagglehub package required for Kaggle datasets: {e}")
        else:
            # Handle HuggingFace datasets
            dataset = load_dataset(
                dataset_config["dataset_id"], split="test", cache_dir=str(Path("data/raw"))
            )

        # Convert to list of dicts
        test_data = []
        text_col = dataset_config["text_column"]
        label_col = dataset_config["label_column"]

        for sample in dataset:
            # Handle rating mapping for datasets with star ratings
            raw_label = sample[label_col]  # type: ignore

            if dataset_config.get("source") == "kaggle" and "rating_mapping" in dataset_config:
                # Convert star ratings to binary sentiment
                threshold = dataset_config["rating_mapping"]["positive_threshold"]
                label = 1 if raw_label >= threshold else 0
                # Keep original rating for accuracy calculation
                original_rating = raw_label
            else:
                # Use raw label for standard datasets
                label = raw_label
                original_rating = raw_label

            # Include branch information for Disneyland dataset
            branch = getattr(sample, "Branch", None) if dataset_name == "disneyland" else None

            test_data.append(
                {
                    "text": sample[text_col],  # type: ignore
                    "label": label,
                    "original_rating": original_rating,
                    "branch": branch,
                }
            )

        # For large datasets, sample a subset for faster testing
        if len(test_data) > 1000:
            import random

            random.seed(42)  # For reproducible results
            original_size = len(test_data)
            test_data = random.sample(test_data, 1000)
            logger.info(f"Sampled 1000 test samples from {original_size} total for {dataset_name}")

        logger.info(f"Loaded {len(test_data)} test samples for {dataset_name}")
        return test_data

    except Exception as e:
        logger.error(f"Failed to load test data for {dataset_name}: {e}")
        return None


def run_inference_with_checkpoint(
    model_name: str,
    training_dataset: str,
    test_dataset: str,
    test_data: List[Dict[str, Any]],
    checkpoints_dir: str,
    device: str,
) -> Tuple[List[int], List[float]]:
    """
    Run inference using a specific checkpoint trained on training_dataset.

    Args:
        model_name: Name of the model
        training_dataset: Dataset the model was trained on
        test_dataset: Dataset to evaluate on
        test_data: Test data samples
        checkpoints_dir: Directory with model checkpoints
        device: Device to use

    Returns:
        Tuple of (predictions, confidence_scores)
    """
    try:
        max_length = _load_dataset_config(test_dataset)
        checkpoint_path = Path(checkpoints_dir) / f"{model_name}_{training_dataset}" / "final"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model, tokenizer = _load_model_and_tokenizer(checkpoint_path)
        _setup_model_device(model, device)
        texts = [sample["text"] for sample in test_data]
        _setup_tokenizer_padding(tokenizer, model_name)
        predictions, confidences = _run_inference_on_texts(texts, tokenizer, model, max_length)
        return predictions, confidences

    except Exception as e:
        logger.error(f"Inference failed for {model_name} on {test_dataset}: {e}")
        raise


def _load_dataset_config(dataset_name: str) -> int:
    """Load dataset configuration and return max_length."""
    import yaml

    config_path = Path("config/datasets.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["datasets"][dataset_name]["max_length"]


def _find_checkpoint_path(model_name: str, dataset_name: str, checkpoints_dir: str) -> Path:
    """Find the appropriate checkpoint path, with fallback to compatible datasets."""
    checkpoint_path = Path(checkpoints_dir) / f"{model_name}_{dataset_name}" / "final"

    if checkpoint_path.exists():
        return checkpoint_path

    # For new datasets, try to find a compatible checkpoint
    compatible_datasets = {
        "disneyland": ["yelp", "amazon", "imdb"],  # Customer reviews
        "yelp": ["amazon", "imdb", "disneyland"],
        "amazon": ["yelp", "imdb", "disneyland"],
        "imdb": ["yelp", "amazon", "disneyland"],
        "sst2": ["imdb", "yelp", "amazon"],
    }

    fallback_datasets = compatible_datasets.get(dataset_name, [])
    for fallback_dataset in fallback_datasets:
        fallback_path = Path(checkpoints_dir) / f"{model_name}_{fallback_dataset}" / "final"
        if fallback_path.exists():
            logger.info(
                f"Using checkpoint from {fallback_dataset} dataset for {dataset_name} evaluation"
            )
            return fallback_path

    raise FileNotFoundError(f"No compatible checkpoint found for {model_name} on {dataset_name}")


def _load_model_and_tokenizer(checkpoint_path: Path):
    """Load model and tokenizer from checkpoint."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer


def _setup_model_device(model, device: str):
    """Move model to the specified device."""
    import torch

    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        model.to("mps")
    else:
        model.to("cpu")
    model.eval()


def _setup_tokenizer_padding(tokenizer, model_name: str):
    """Setup padding token for tokenizer if needed."""
    has_padding_token = tokenizer.pad_token is not None or (
        hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None
    )

    if not has_padding_token:
        logger.warning(
            f"Tokenizer for {model_name} has no padding token. Using batch_size=1 for inference."
        )
        # Try to set a padding token if possible
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token for {model_name}")


def _run_inference_on_texts(texts, tokenizer, model, max_length: int):
    """Run inference on list of texts and return predictions and confidences."""
    import torch

    all_predictions = []
    all_confidences = []

    for text in texts:
        try:
            # Tokenize individual text
            inputs = tokenizer(
                text,
                max_length=max_length,
                padding=False,  # Don't pad single text
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                # Get predictions and confidence scores
                probs = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1).cpu().numpy().item()
                confidence = torch.max(probs, dim=-1)[0].cpu().numpy().item()

            all_predictions.append(prediction)
            all_confidences.append(confidence)
        except Exception as text_error:
            logger.warning(
                f"Failed to process individual text: {text_error}. Skipping this sample."
            )
            # Add neutral prediction for failed samples
            all_predictions.append(0)  # Neutral prediction
            all_confidences.append(0.5)  # Neutral confidence

    return all_predictions, all_confidences


def run_inference(
    model_name: str,
    dataset_name: str,
    test_data: List[Dict[str, Any]],
    checkpoints_dir: str,
    device: str,
) -> Tuple[List[int], List[float]]:
    """
    Run inference on test data and return predictions with confidence scores.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        test_data: Test data samples
        checkpoints_dir: Directory with model checkpoints
        device: Device to use

    Returns:
        Tuple of (predictions, confidence_scores)
    """
    try:
        max_length = _load_dataset_config(dataset_name)
        checkpoint_path = _find_checkpoint_path(model_name, dataset_name, checkpoints_dir)
        model, tokenizer = _load_model_and_tokenizer(checkpoint_path)

        _setup_model_device(model, device)
        texts = [sample["text"] for sample in test_data]
        _setup_tokenizer_padding(tokenizer, model_name)
        predictions, confidences = _run_inference_on_texts(texts, tokenizer, model, max_length)
        return predictions, confidences

    except Exception as e:
        logger.error(f"Inference failed for {model_name} on {dataset_name}: {e}")
        raise


def _categorize_predictions(
    predictions: List[int], confidences: List[float]
) -> Tuple[int, int, int]:
    """Categorize predictions into promoters, passives, detractors."""
    promoters = 0
    passives = 0
    detractors = 0

    for pred, conf in zip(predictions, confidences):
        if pred == 0:  # Negative prediction
            detractors += 1
        else:  # Positive prediction
            if conf >= 0.8:  # High confidence positive = Promoter
                promoters += 1
            elif conf >= 0.6:  # Medium confidence positive = Passive
                passives += 1
            else:  # Low confidence positive = Passive (could be either)
                passives += 1

    return promoters, passives, detractors


def _calculate_accuracy(predictions: List[int], actual_ratings: List[int]) -> Tuple[float, int]:
    """Calculate accuracy from predictions and actual ratings."""
    actual_sentiments = [1 if rating >= 4 else 0 for rating in actual_ratings]
    correct_predictions = sum(
        1 for pred, actual in zip(predictions, actual_sentiments) if pred == actual
    )
    accuracy = (correct_predictions / len(predictions)) * 100
    return accuracy, correct_predictions


def calculate_nps_from_predictions(
    predictions: List[int], confidences: List[float], actual_ratings: List[int] | None = None
) -> Dict[str, Any]:
    """
    Calculate NPS from binary predictions and confidence scores.

    NPS Categories based on sentiment confidence:
    - Promoters (9-10): High-confidence positive predictions
    - Passives (7-8): Medium-confidence positive predictions or low confidence predictions
    - Detractors (0-6): Negative predictions

    Args:
        predictions: Binary predictions (0=negative, 1=positive)
        confidences: Confidence scores for predictions
        actual_ratings: Optional actual ratings (1-5 scale) for accuracy calculation

    Returns:
        Dict with NPS calculation results
    """
    if len(predictions) != len(confidences):
        raise ValueError("Predictions and confidences must have same length")

    total_samples = len(predictions)

    # Categorize predictions
    promoters, passives, detractors = _categorize_predictions(predictions, confidences)

    # Calculate percentages
    promoters_percent = (promoters / total_samples) * 100
    passives_percent = (passives / total_samples) * 100
    detractors_percent = (detractors / total_samples) * 100

    # Calculate NPS
    nps_score = promoters_percent - detractors_percent

    result = {
        "nps_score": nps_score,
        "promoters_percent": promoters_percent,
        "passives_percent": passives_percent,
        "detractors_percent": detractors_percent,
        "promoters_count": promoters,
        "passives_count": passives,
        "detractors_count": detractors,
        "total_samples": total_samples,
    }

    # Calculate accuracy if actual ratings are provided
    if actual_ratings and len(actual_ratings) == len(predictions):
        accuracy, correct_predictions = _calculate_accuracy(predictions, actual_ratings)
        result["accuracy_percent"] = accuracy
        result["correct_predictions"] = correct_predictions
    else:
        result["accuracy_percent"] = None
        result["correct_predictions"] = None

    return result
