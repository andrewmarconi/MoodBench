"""
UI modules for MoodBench Gradio interface.

This package contains modular components for the web interface,
organized by functionality to reduce complexity and improve maintainability.
"""

from . import training, benchmark, reports, analysis, nps, methodology

__all__ = ["training", "benchmark", "reports", "analysis", "nps", "methodology"]
