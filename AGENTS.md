# AGENTS.md

## Commands

**Package Management (uv):**
```bash
uv sync                    # Install dependencies
uv add <package>           # Add dependency
uv run python main.py       # Run main script
uv run moodbench           # Run CLI (after uv sync)
```

**Development:**
```bash
uv run ruff check .         # Lint code
uv run ruff check src/evaluation/metrics.py  # Lint specific file
uv run black .             # Format code
uv run pytest              # Run all tests
uv run pytest tests/test_evaluation.py::TestMetricsCalculator::test_compute_basic_metrics  # Run single test
uv run pytest -k "test_metrics"  # Run tests by name pattern
uv run pytest --cov=src    # Run with coverage
uv run pytest --cov=src --cov-report=html  # Generate coverage report
```

**Training & Evaluation:**
```bash
uv run moodbench train --model DistilBERT-base --dataset imdb  # Train single model
./scripts/train_model.sh <model> <dataset>  # Train via script
./scripts/train_all.sh [dataset]           # Train all models
./scripts/evaluate_all.sh                  # Run full evaluation
uv run moodbench benchmark --models-dir experiments/checkpoints  # Run benchmark
uv run moodbench estimated-nps --all-models --all-datasets  # Estimate NPS from model predictions
python scripts/clean_results.py             # Clean invalid benchmark results with N/A values
```

## Code Style

**Formatting:** Black (100 char line length), Ruff for linting (Python 3.12+)
**Types:** Full type hints required, use `from typing import` for imports
**Imports:** Group stdlib → third-party → local; absolute imports preferred
**Naming:** snake_case functions/variables, PascalCase classes, UPPER_CASE constants
**Error Handling:** Log with `logger.error()` + raise descriptive exceptions
**Device Management:** Always use `src.utils.device.get_device()` for hardware compatibility
**Model Preparation:** Use `LoRAConfigManager.prepare_model()` for quantization
**Configuration:** Load YAML from `config/` with fallback defaults
**Logging:** `logging.getLogger(__name__)` for module-level loggers
**Docstrings:** Google-style with Args/Returns sections and examples
**Testing:** pytest with fixtures, parametrize for multiple test cases
**CLI:** Use argparse with subcommands, comprehensive help examples