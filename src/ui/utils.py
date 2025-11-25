"""
Shared utilities for UI components.
"""

import os
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
