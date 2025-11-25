"""
Simple Gradio Web UI for MoodBench - Multi-LLM Sentiment Analysis Benchmark Framework

A basic web interface demonstrating MoodBench functionality.
Install required dependencies: pip install gradio
"""

import gradio as gr

# Import UI modules
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.ui.methodology import create_methodology_tab
    from src.ui.training import create_training_tab
    from src.ui.benchmark import create_benchmark_tab
    from src.ui.reports import create_reports_tab
    from src.ui.analysis import create_analysis_tab
    from src.ui.nps import create_nps_tab
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_interface():
    """Create the main Gradio interface."""
    with gr.Blocks(title="MoodBench - Multi-LLM Sentiment Analysis Benchmark") as interface:
        gr.Markdown("""
        # 🤖 MoodBench

        **Fast benchmarking of small language models (4M-410M parameters) for sentiment analysis.**

        This interface provides access to all MoodBench functionality through an intuitive web UI.
        """)

        with gr.Tabs():
            # Import and create all tabs from modules
            create_methodology_tab()
            create_training_tab()
            create_benchmark_tab()
            create_reports_tab()
            create_analysis_tab()
            create_nps_tab()

    return interface


def main():
    """Main entry point."""
    interface = create_interface()

    # Launch the interface
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
