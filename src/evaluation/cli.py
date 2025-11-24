"""
Evaluation CLI commands for EmoBench.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def benchmark_models(args):
    """
    Run benchmark on selected models and datasets.

    Args:
        args: Parsed command line arguments
    """
    logger.info(f"Running benchmark on models: {args.models} on datasets: {args.datasets}")

    try:
        # Import evaluation components
        from src.evaluation.benchmark import benchmark_models as benchmark_func
        from src.utils.device import get_device

        # Get device
        device = get_device() if args.device == "auto" else args.device
        device_str = str(device) if hasattr(device, "type") else str(device)
        logger.info(f"Using device: {device_str}")

        # Run benchmark
        benchmark_func(
            models=args.models,
            datasets=args.datasets,
            checkpoints_dir=str(args.checkpoints_dir),
            output_dir=str(args.output_dir),
            device=device_str,
        )

        logger.info(f"Benchmark completed! Results saved to: {args.output_dir}")

    except ImportError as e:
        logger.error(f"Evaluation module not implemented yet: {e}")
        logger.info("Please implement the evaluation module first")
        raise
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
