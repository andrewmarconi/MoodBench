"""
Reports UI components for MoodBench Gradio interface.
"""

import sys
import os
from pathlib import Path
import gradio as gr

import subprocess
import time


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


def generate_reports():
    """Generate reports with real-time progress and return markdown content."""
    command = [
        sys.executable,
        "-m",
        "src.cli",
        "report",
        "--format",
        "all",
    ]

    yield (0, "📋 Generating reports...", "")

    progress_updates = []
    json_done = False
    csv_done = False
    markdown_done = False

    for update in run_command_stream(command):
        progress_updates.append(update)

        # Track completion of each format
        update_lower = update.lower()
        if "json" in update_lower and ("saved" in update_lower or "generated" in update_lower):
            json_done = True
        if "csv" in update_lower and ("saved" in update_lower or "generated" in update_lower):
            csv_done = True
        if "markdown" in update_lower and ("saved" in update_lower or "generated" in update_lower):
            markdown_done = True

        # Calculate progress
        completed = sum([json_done, csv_done, markdown_done])
        progress = int((completed / 3) * 100)

        yield (progress, "\n".join(progress_updates[-10:]), "")

    # Load and return the markdown report content
    try:
        markdown_path = "experiments/reports/moodbench_report.md"
        if os.path.exists(markdown_path):
            with open(markdown_path, "r") as f:
                markdown_content = f.read()
            yield (100, "✅ Reports generated successfully!", markdown_content)
        else:
            yield (100, "❌ Reports generated but markdown file not found", "")
    except Exception as e:
        yield (100, f"❌ Error loading markdown report: {str(e)}", "")


def create_reports_tab():
    """Create the reports tab for the Gradio interface."""
    with gr.TabItem("📋 Generate Reports"):
        gr.Markdown("### Generate comparison reports from benchmark results")

        with gr.Row():
            with gr.Column():
                report_markdown = gr.Markdown(
                    label="📊 Generated Report",
                    value="",
                    height=400,
                )

            with gr.Column():
                report_progress = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Generation Progress (%)",
                    interactive=False,
                )
                with gr.Accordion("Generation Details", open=False):
                    report_output = gr.Textbox(
                        label="Status",
                        lines=10,
                        interactive=False,
                        show_copy_button=True,
                        autoscroll=True,
                    )

        report_button = gr.Button("📋 Generate Reports", variant="primary")
        report_button.click(
            fn=generate_reports,
            inputs=[],
            outputs=[report_progress, report_output, report_markdown],
        )
