"""
Training UI components for MoodBench Gradio interface.
"""

import sys
from pathlib import Path
import gradio as gr

# Import shared constants and functions
from ..utils.device import get_device  # Assuming this exists, or we can define locally

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

import os
import sys
import subprocess
import time
from pathlib import Path


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


def train_models(models, datasets):
    """Train multiple models on multiple datasets with progress tracking."""
    if not models:
        yield (0, "❌ Please select at least one model to train.")
        return
    if not datasets:
        yield (0, "❌ Please select at least one dataset to train on.")
        return

    total_combinations = len(models) * len(datasets)
    completed = 0

    # Show progress bar and initial message
    yield (
        0,
        f"🚀 Starting training of {len(models)} models on {len(datasets)} datasets ({total_combinations} total combinations)...",
    )

    for dataset in datasets:
        for model in models:
            progress_percent = int((completed / total_combinations) * 100)
            yield (
                progress_percent,
                f"📋 Training {model} on {dataset}... ({completed + 1}/{total_combinations})",
            )

            command = [
                sys.executable,
                "-m",
                "src.cli",
                "train",
                "--model",
                model,
                "--dataset",
                dataset,
                "--device",
                "auto",
            ]

            for update in run_command_stream(command):
                # Keep progress at current level during training
                yield (progress_percent, f"[{model} on {dataset}] {update}")

            completed += 1
            progress_percent = int((completed / total_combinations) * 100)
            yield (
                progress_percent,
                f"✅ Completed {model} on {dataset} ({completed}/{total_combinations})",
            )

    yield (
        100,
        f"🎉 All training completed! Trained {len(models)} models on {len(datasets)} datasets.",
    )


def create_training_matrix():
    """Create an HTML table showing trained model-dataset combinations."""
    checkpoints_dir = Path("experiments/checkpoints")

    # Get all available models and datasets from the constants
    all_models = DEFAULT_MODELS
    all_datasets = DEFAULT_DATASETS

    # Check which combinations have been trained
    trained_combinations = {}

    if checkpoints_dir.exists():
        for item in checkpoints_dir.iterdir():
            if item.is_dir():
                # Parse model_dataset from directory name
                dir_name = item.name
                if "_" in dir_name:
                    parts = dir_name.split("_", 1)  # Split only on first underscore
                    if len(parts) == 2:
                        model, dataset = parts
                        # Check if final checkpoint exists
                        final_checkpoint = item / "final"
                        if final_checkpoint.exists():
                            trained_combinations[(model, dataset)] = True

    # Create HTML table
    html = """
    <style>
        .training-matrix {
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 14px;
            width: 100%;
        }
        .training-matrix th, .training-matrix td {
            border: 1px solid var(--border-color-primary, #e5e7eb);
            padding: 8px 12px;
            text-align: center;
        }
        .training-matrix th {
            background-color: var(--background-fill-secondary, #f9fafb);
            font-weight: bold;
            color: var(--body-text-color, #111827);
        }
        .training-matrix .model-header {
            background-color: var(--background-fill-primary, #ffffff);
            font-weight: 600;
        }
        .trained {
            background-color: rgba(34, 197, 94, 0.1);
            color: #16a34a;
        }
        .not-trained {
            background-color: rgba(239, 68, 68, 0.1);
            color: #dc2626;
        }
        .status-icon {
            font-size: 16px;
            font-weight: bold;
        }
        @media (prefers-color-scheme: dark) {
            .trained {
                background-color: rgba(34, 197, 94, 0.2);
                color: #4ade80;
            }
            .not-trained {
                background-color: rgba(239, 68, 68, 0.2);
                color: #f87171;
            }
        }
    </style>
    <table class="training-matrix">
        <thead>
            <tr>
                <th class="model-header">Model ↓ / Dataset →</th>
    """

    # Add dataset headers
    for dataset in all_datasets:
        html += f"<th>{dataset}</th>"
    html += "</tr></thead><tbody>"

    # Add rows for each model
    for model in all_models:
        html += f"<tr><td class='model-header'>{model}</td>"
        for dataset in all_datasets:
            is_trained = (model, dataset) in trained_combinations
            status_class = "trained" if is_trained else "not-trained"
            status_icon = "✅" if is_trained else "❌"
            status_text = "Trained" if is_trained else "Not Trained"
            html += f'<td class="{status_class}" title="{model} on {dataset}: {status_text}"><span class="status-icon">{status_icon}</span></td>'
        html += "</tr>"

    html += "</tbody></table>"

    return html


def create_training_tab():
    """Create the training tab for the Gradio interface."""
    with gr.TabItem("🚀 Train Models"):
        gr.Markdown("### Train multiple models on multiple datasets")

        gr.Markdown("""
        ⚠️ **Note about training timeouts:** Due to Gradio's request timeout limitations, training larger models or complex configurations may fail in the web interface.
        For long-running trainings, use the command line interface instead:

        ```bash
        uv run moodbench train --model <model_name> --dataset <dataset_name>
        ```

        The web interface works best for quick training runs on smaller models like BERT-tiny and BERT-mini.
        """)

        with gr.Row():
            # Column 1: Inputs and training matrix
            with gr.Column():
                models_checkboxes = gr.CheckboxGroup(
                    choices=DEFAULT_MODELS,
                    label="Models to Train",
                    value=["BERT-tiny", "BERT-mini"],
                    info="Select one or more models to train",
                )
                datasets_checkboxes = gr.CheckboxGroup(
                    choices=DEFAULT_DATASETS,
                    label="Datasets to Train On",
                    value=["imdb"],
                    info="Select one or more datasets to train on",
                )

                # Training status matrix
                training_matrix = create_training_matrix()
                matrix_display = gr.HTML(training_matrix)
                gr.Markdown("*Green cells (✅) indicate trained combinations*")

                # Refresh button for the matrix
                refresh_matrix_btn = gr.Button("🔄 Refresh Status", size="sm")
                refresh_matrix_btn.click(
                    fn=lambda: create_training_matrix(), outputs=matrix_display
                )

            # Column 2: Progress and output
            with gr.Column():
                train_progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Training Progress (%)",
                    interactive=False,
                )
                with gr.Accordion("Training Details", open=False):
                    train_output = gr.Textbox(
                        label="Status",
                        lines=15,
                        interactive=False,
                        show_copy_button=True,
                        autoscroll=True,
                    )

                train_button = gr.Button("🚀 Start Training", variant="primary")
                train_button.click(
                    fn=train_models,
                    inputs=[models_checkboxes, datasets_checkboxes],
                    outputs=[train_progress_bar, train_output],
                )
