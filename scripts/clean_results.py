#!/usr/bin/env python3
"""
Clean benchmark result files by removing entries with null/invalid metric values.
"""

import json
from pathlib import Path


def clean_benchmark_results(results_file: Path) -> bool:
    """Clean a benchmark results file by removing invalid entries."""
    print(f"Processing {results_file}...")

    # Load the results
    with open(results_file, "r") as f:
        data = json.load(f)

    cleaned_data = {}
    total_removed = 0
    total_kept = 0

    for model_name, results_list in data.items():
        cleaned_results = []

        for result in results_list:
            # Check if this result has valid metrics
            is_valid = True

            # Check for null values in key metrics
            key_metrics = [
                "metric_accuracy",
                "metric_f1",
                "latency_mean_ms",
                "throughput_samples_per_sec",
            ]

            for metric in key_metrics:
                if result.get(metric) is None:
                    is_valid = False
                    break

            # Also check for error field (indicates failed evaluation)
            if "error" in result:
                is_valid = False

            if is_valid:
                cleaned_results.append(result)
                total_kept += 1
            else:
                total_removed += 1
                print(
                    f"  Removing invalid result: {model_name} on {result.get('dataset', 'unknown')}"
                )

        if cleaned_results:
            cleaned_data[model_name] = cleaned_results

    if total_removed > 0:
        # Write back the cleaned data
        with open(results_file, "w") as f:
            json.dump(cleaned_data, f, indent=2)

        print(f"Cleaned {results_file}: removed {total_removed} invalid entries, kept {total_kept}")
        return True
    else:
        print(f"No invalid entries found in {results_file}")
        return False


def main():
    """Main entry point."""
    results_dir = Path("experiments/results")

    if not results_dir.exists():
        print("Results directory not found")
        return

    # Find all benchmark result files
    benchmark_files = list(results_dir.glob("benchmark_*.json"))

    if not benchmark_files:
        print("No benchmark result files found")
        return

    total_cleaned = 0
    for results_file in benchmark_files:
        if clean_benchmark_results(results_file):
            total_cleaned += 1

    print(f"\nProcessed {len(benchmark_files)} files, cleaned {total_cleaned} files")


if __name__ == "__main__":
    main()
