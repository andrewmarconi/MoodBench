"""
Benchmark UI components for MoodBench Gradio interface.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
import gradio as gr

# Available models and datasets (fallback if registry not available)
DEFAULT_MODELS = [
    "BERT-tiny",
    "BERT-mini",
    "BERT-small",
    "ELECTRA-small",
    "MiniLM-L12",
    "DistilBERT-base",
    "RoBERTa-base",
    # "GPT2-small",  # Temporarily disabled for Gradio - too slow for demo
]

DEFAULT_DATASETS = ["imdb", "sst2", "amazon", "yelp"]


def run_command_stream(command_list, timeout=300):
    """Run a command and stream output in real-time."""
    try:
        # Set environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)
        # Enable test mode for faster training in Gradio
        env["MOODBENCH_TEST_MODE"] = "1"

        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Stream output in real-time
        output_lines = []

        start_time = time.time()

        if process.stdout:
            while True:
                # Check for timeout
                if time.time() - start_time > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    yield f"❌ Command timed out after {timeout} seconds"
                    return

                if process.poll() is not None:
                    # Process finished, read remaining output
                    try:
                        remaining = process.stdout.read()
                        if remaining:
                            lines = remaining.strip().split("\n")
                            output_lines.extend(lines)
                            if lines:
                                yield "\n".join(output_lines[-20:])  # Keep last 20 lines
                    except Exception:
                        pass
                    break

                try:
                    # Read output with timeout
                    output = process.stdout.readline()
                    if not output:
                        # Check if process is still running
                        if process.poll() is None:
                            time.sleep(0.1)  # Small delay to prevent busy waiting
                            continue
                        else:
                            break

                    line = output.strip()
                    if line:
                        output_lines.append(line)
                        yield "\n".join(output_lines[-20:])  # Keep last 20 lines

                except Exception as e:
                    yield f"❌ Error reading output: {str(e)}"
                    break

        # Wait for process to finish if not already done
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = -1
            yield "❌ Command killed due to timeout"

        if return_code == 0:
            yield "✅ Command completed successfully!"
        else:
            yield f"❌ Command failed with return code {return_code}"

    except Exception as e:
        yield f"❌ Error: {str(e)}"


def run_benchmark(models, datasets):
    """Run benchmark on selected models and datasets with progress tracking."""
    if not models:
        yield (0, "❌ Please select at least one model to benchmark.")
        return
    if not datasets:
        yield (0, "❌ Please select at least one dataset to test on.")
        return

    total_combinations = len(models) * len(datasets)
    completed = 0

    yield (
        0,
        f"🏁 Starting benchmark of {len(models)} models on {len(datasets)} datasets ({total_combinations} total combinations)...",
    )

    # Build command with all models and datasets
    command = [
        sys.executable,
        "-m",
        "src.cli",
        "benchmark",
        "--checkpoints-dir",
        "experiments/checkpoints",
        "--device",
        "auto",
    ]

    # Add models
    command.extend(["--models"] + models)

    # Add datasets
    command.extend(["--datasets"] + datasets)

    for update in run_command_stream(command):
        # Calculate progress based on completion messages in the output
        # For now, we'll show 50% progress during execution and 100% at end
        if "completed" in update.lower() or "finished" in update.lower():
            completed += 1
            progress_percent = min(int((completed / total_combinations) * 100), 100)
            yield (progress_percent, update)
        else:
            yield (50, update)  # Show 50% during execution

    yield (100, f"🎉 Benchmark completed! Tested {len(models)} models on {len(datasets)} datasets.")


def create_benchmark_tab():
    """Create the benchmark tab for the Gradio interface."""
    with gr.TabItem("📊 Benchmark"):
        gr.Markdown("### Run benchmark on selected models and datasets")

        with gr.Row():
            with gr.Column():
                models_checkboxes_benchmark = gr.CheckboxGroup(
                    choices=DEFAULT_MODELS,
                    label="Models to Benchmark",
                    value=["BERT-tiny"],
                    info="Select models to include in benchmark",
                )
                datasets_checkboxes_benchmark = gr.CheckboxGroup(
                    choices=DEFAULT_DATASETS,
                    label="Datasets to Test On",
                    value=["imdb"],
                    info="Select datasets to evaluate models on",
                )

            with gr.Column():
                benchmark_progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Progress (%)",
                    interactive=False,
                )
                with gr.Accordion("Benchmark Details", open=False):
                    benchmark_output = gr.Textbox(
                        label="Status",
                        lines=15,
                        interactive=False,
                        show_copy_button=True,
                        autoscroll=True,
                    )

        benchmark_button = gr.Button("🏁 Start Benchmark", variant="primary")
        benchmark_button.click(
            fn=run_benchmark,
            inputs=[models_checkboxes_benchmark, datasets_checkboxes_benchmark],
            outputs=[benchmark_progress_bar, benchmark_output],
        )
