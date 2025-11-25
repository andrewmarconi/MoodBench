"""
Analysis UI components for MoodBench Gradio interface.
"""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr


def load_benchmark_results():
    """Load benchmark results from the results directory."""
    results_dir = Path("experiments/results")
    if not results_dir.exists():
        return pd.DataFrame()

    # Find all benchmark result files
    benchmark_files = list(results_dir.glob("benchmark_*.json"))

    if not benchmark_files:
        return pd.DataFrame()

    # Load and combine all results
    all_results = []
    for results_file in benchmark_files:
        try:
            with open(results_file, "r") as f:
                raw_results = json.load(f)

            # Flatten the results for DataFrame
            for model, results_list in raw_results.items():
                for result in results_list:
                    result["_source_file"] = results_file.name
                    all_results.append(result)

        except Exception:
            continue

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def create_dashboard_summary(results_df):
    """Create a summary of benchmark results."""
    if results_df.empty:
        return {
            "total_models": 0,
            "total_datasets": 0,
            "total_results": 0,
            "last_updated": "No data available",
            "top_model": "N/A",
        }

    summary = {
        "total_models": results_df["model_name"].nunique(),
        "total_datasets": results_df["dataset"].nunique(),
        "total_results": len(results_df),
        "last_updated": "Unknown",
    }

    # Find top performing model (highest accuracy)
    if "metric_accuracy" in results_df.columns:
        valid_data = results_df.dropna(subset=["metric_accuracy"])
        if not valid_data.empty:
            best_idx = valid_data["metric_accuracy"].idxmax()
            summary["top_model"] = valid_data.loc[best_idx, "model_name"]
        else:
            summary["top_model"] = "N/A"
    else:
        summary["top_model"] = "N/A"

    # Try to get timestamp
    if "timestamp" in results_df.columns:
        timestamps = pd.to_datetime(results_df["timestamp"], unit="s", errors="coerce")
        if not timestamps.empty:
            summary["last_updated"] = timestamps.max().strftime("%Y-%m-%d %H:%M:%S")

    return summary


def create_scatter_plot(results_df):
    """Create a scatter plot of latency vs accuracy with individual runs and model means."""
    if results_df.empty or "metric_accuracy" not in results_df.columns:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No Data Available", xaxis_title="Latency (ms)", yaxis_title="Accuracy"
        )
        return fig

    # Prepare data
    plot_data = results_df.dropna(subset=["metric_accuracy"])

    if plot_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Valid Data for Plotting", xaxis_title="Latency (ms)", yaxis_title="Accuracy"
        )
        return fig

    # Calculate per-model means
    model_means = (
        plot_data.groupby("model_name")
        .agg({"metric_accuracy": "mean", "latency_mean_ms": "mean"})
        .reset_index()
    )

    # Get unique models and assign colors
    unique_models = sorted(plot_data["model_name"].unique())
    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Create color mapping
    model_colors = {}
    for i, model in enumerate(unique_models):
        model_colors[model] = color_palette[i % len(color_palette)]

    # Create figure
    fig = go.Figure()

    # Add traces for each model (individual runs)
    for model in unique_models:
        model_data = plot_data[plot_data["model_name"] == model]
        model_color = model_colors[model]

        # Individual runs for this model
        fig.add_trace(
            go.Scatter(
                x=model_data["latency_mean_ms"],
                y=model_data["metric_accuracy"],
                mode="markers",
                name=f"{model} (runs)",
                marker=dict(
                    size=6, opacity=0.6, color=model_color, line=dict(width=1, color=model_color)
                ),
                hovertemplate=f"<b>{model}</b><br>"
                + "Latency: %{x:.2f}ms<br>"
                + "Accuracy: %{y:.4f}<br>"
                + "Dataset: %{customdata}<extra></extra>",
                customdata=model_data["dataset"],
                legendgroup=model,
                showlegend=True,
            )
        )

    # Add traces for model means
    for model in unique_models:
        model_mean = model_means[model_means["model_name"] == model]
        if not model_mean.empty:
            model_color = model_colors[model]

            fig.add_trace(
                go.Scatter(
                    x=model_mean["latency_mean_ms"],
                    y=model_mean["metric_accuracy"],
                    mode="markers",
                    name=f"{model} (mean)",
                    marker=dict(
                        size=14,
                        opacity=1.0,
                        color=model_color,
                        symbol="diamond",
                        line=dict(width=2, color="black"),
                    ),
                    hovertemplate=f"<b>{model} (Mean)</b><br>"
                    + "Avg Latency: %{x:.2f}ms<br>"
                    + "Avg Accuracy: %{y:.4f}<extra></extra>",
                    legendgroup=model,
                    showlegend=True,
                )
            )

    # Update layout
    fig.update_layout(
        title="Model Performance: Latency vs Accuracy",
        xaxis_title="Latency (ms)",
        yaxis_title="Accuracy",
        font=dict(size=12),
        title_font=dict(size=16),
        showlegend=False,  # Hide legend since hover provides details
        hovermode="closest",
    )

    return fig


def create_accuracy_by_dataset_chart(results_df):
    """Create bar chart showing model accuracy by dataset."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Aggregate data by model and dataset to prevent stacking of duplicates
    aggregated_df = (
        results_df.groupby(["model_name", "dataset"])
        .agg(
            {
                "metric_accuracy": "mean"  # Take mean if multiple runs exist
            }
        )
        .reset_index()
    )

    # Filter out rows with no accuracy data
    aggregated_df = aggregated_df.dropna(subset=["metric_accuracy"])

    if aggregated_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Accuracy Data Available")
        return fig

    # Create bar chart for accuracy
    fig = px.bar(
        aggregated_df,
        x="model_name",
        y="metric_accuracy",
        color="dataset",
        barmode="group",
        title="Model Accuracy by Dataset",
        labels={"model_name": "Model", "metric_accuracy": "Accuracy", "dataset": "Dataset"},
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)

    return fig


def create_f1_by_dataset_chart(results_df):
    """Create bar chart showing model F1 score by dataset."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Aggregate data by model and dataset to prevent stacking of duplicates
    aggregated_df = (
        results_df.groupby(["model_name", "dataset"])
        .agg(
            {
                "metric_f1": "mean"  # Take mean if multiple runs exist
            }
        )
        .reset_index()
    )

    # Filter out rows with no F1 data
    aggregated_df = aggregated_df.dropna(subset=["metric_f1"])

    if aggregated_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No F1 Data Available")
        return fig

    # Create bar chart for F1
    fig = px.bar(
        aggregated_df,
        x="model_name",
        y="metric_f1",
        color="dataset",
        barmode="group",
        title="Model F1 Score by Dataset",
        labels={"model_name": "Model", "metric_f1": "F1 Score", "dataset": "Dataset"},
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)

    return fig


def create_latency_breakdown_chart(results_df):
    """Create chart showing latency metrics breakdown per model (Mean, Median, P95, P99)."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Aggregate data by model and dataset to prevent stacking of duplicates
    aggregated_df = (
        results_df.groupby(["model_name", "dataset"])
        .agg(
            {
                "latency_mean_ms": "mean",
                "latency_median_ms": "mean",
                "latency_p95_ms": "mean",
                "latency_p99_ms": "mean",
            }
        )
        .reset_index()
    )

    # Prepare data for bar chart - we'll show mean, median, p95, p99 as separate traces
    latency_data = []

    for _, row in aggregated_df.iterrows():
        model = row["model_name"]
        dataset = row["dataset"]

        # Create entries for each latency metric (excluding TTFT)
        latency_data.extend(
            [
                {
                    "model": model,
                    "dataset": dataset,
                    "metric": "Mean",
                    "value": row.get("latency_mean_ms", 0),
                },
                {
                    "model": model,
                    "dataset": dataset,
                    "metric": "Median",
                    "value": row.get("latency_median_ms", 0),
                },
                {
                    "model": model,
                    "dataset": dataset,
                    "metric": "P95",
                    "value": row.get("latency_p95_ms", 0),
                },
                {
                    "model": model,
                    "dataset": dataset,
                    "metric": "P99",
                    "value": row.get("latency_p99_ms", 0),
                },
            ]
        )

    if not latency_data:
        fig = go.Figure()
        fig.update_layout(title="No Latency Data Available")
        return fig

    latency_df = pd.DataFrame(latency_data)

    # Create grouped bar chart - group by model and dataset combination
    latency_df["model_dataset"] = latency_df["model"] + " (" + latency_df["dataset"] + ")"

    fig = px.bar(
        latency_df,
        x="model_dataset",
        y="value",
        color="metric",
        barmode="group",
        title="Latency Breakdown per Model",
        labels={
            "model_dataset": "Model (Dataset)",
            "value": "Latency (ms)",
            "metric": "Latency Metric",
        },
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_tickangle=-45,
    )

    return fig


def create_ttft_chart(results_df):
    """Create bar chart showing Time to First Token (TTFT) per model."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Aggregate data by model and dataset to prevent stacking of duplicates
    aggregated_df = (
        results_df.groupby(["model_name", "dataset"])
        .agg(
            {
                "latency_ttft_ms": "mean"  # Take mean if multiple runs exist
            }
        )
        .reset_index()
    )

    # Filter out rows with no TTFT data
    aggregated_df = aggregated_df.dropna(subset=["latency_ttft_ms"])

    if aggregated_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No TTFT Data Available")
        return fig

    # Create bar chart for TTFT
    fig = px.bar(
        aggregated_df,
        x="model_name",
        y="latency_ttft_ms",
        color="dataset",
        barmode="group",
        title="Time to First Token (TTFT) per Model",
        labels={"model_name": "Model", "latency_ttft_ms": "TTFT (ms)", "dataset": "Dataset"},
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)

    return fig


def create_throughput_accuracy_scatter(results_df):
    """Create scatter plot of throughput vs accuracy."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    fig = px.scatter(
        results_df,
        x="throughput_samples_per_sec",
        y="metric_accuracy",
        color="model_name",
        symbol="dataset",
        title="Throughput vs. Accuracy Scatter",
        labels={
            "throughput_samples_per_sec": "Throughput (samples/sec)",
            "metric_accuracy": "Accuracy",
            "model_name": "Model",
            "dataset": "Dataset",
        },
        hover_data=["model_name", "dataset", "metric_f1"],
    )

    fig.update_layout(height=400, showlegend=True)

    return fig


def create_throughput_comparison_chart(results_df):
    """Create horizontal bar chart comparing throughput."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Aggregate data by model and dataset to prevent stacking of duplicates
    aggregated_df = (
        results_df.groupby(["model_name", "dataset"])
        .agg(
            {
                "throughput_samples_per_sec": "mean"  # Take mean if multiple runs exist
            }
        )
        .reset_index()
    )

    fig = px.bar(
        aggregated_df,  # Use aggregated data instead of raw results_df
        x="throughput_samples_per_sec",
        y="model_name",
        color="dataset",
        orientation="h",
        barmode="group",
        title="Comparative Throughput Analysis",
        labels={
            "throughput_samples_per_sec": "Throughput (samples/sec)",
            "model_name": "Model",
            "dataset": "Dataset",
        },
    )

    fig.update_layout(height=400, showlegend=True)

    return fig


def create_efficiency_bubble_chart(results_df):
    """Create bubble chart showing latency vs accuracy with throughput as size."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Filter out rows with NaN values in critical columns for bubble chart
    plot_data = results_df.dropna(
        subset=["latency_mean_ms", "metric_accuracy", "throughput_samples_per_sec"]
    )

    if plot_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Complete Data Available for Efficiency Chart",
            annotations=[
                dict(
                    text="Run benchmarks to generate throughput and latency data",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14),
                )
            ],
        )
        return fig

    fig = px.scatter(
        plot_data,
        x="latency_mean_ms",
        y="metric_accuracy",
        size="throughput_samples_per_sec",
        color="model_name",
        hover_data=["dataset", "metric_f1"],
        title="Model Efficiency Chart (Latency vs Accuracy, Size = Throughput)",
        labels={
            "latency_mean_ms": "Latency (ms)",
            "metric_accuracy": "Accuracy",
            "throughput_samples_per_sec": "Throughput",
            "model_name": "Model",
        },
    )

    fig.update_layout(height=400, showlegend=True)

    return fig


def create_latency_distribution_chart(results_df):
    """Create range plot showing latency percentiles per model, grouped by base model."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Create figure
    fig = go.Figure()

    # Dataset color mapping
    dataset_colors = {
        "imdb": "#3b82f6",  # blue
        "sst2": "#8b5cf6",  # purple
        "amazon": "#06b6d4",  # cyan
        "yelp": "#10b981",  # emerald
    }

    # Get unique model-dataset combinations
    unique_combinations = results_df.groupby(["model_name", "dataset"]).size().reset_index()
    unique_models = sorted(results_df["model_name"].unique())

    # Create y-axis positions for each unique model
    y_positions = []
    current_y = 0

    for model in unique_models:
        model_label = f"<b>{model}</b>"
        y_positions.append((model_label, current_y))

        # Get datasets for this specific model
        model_data = unique_combinations[unique_combinations["model_name"] == model]
        datasets_for_model = sorted(model_data["dataset"].unique())

        # Add points for each dataset of this model
        for dataset in datasets_for_model:
            dataset_data = results_df[
                (results_df["model_name"] == model) & (results_df["dataset"] == dataset)
            ]

            if not dataset_data.empty:
                # Use the first row (assuming aggregated data)
                row = dataset_data.iloc[0]

                # Add marker for this dataset
                fig.add_trace(
                    go.Scatter(
                        x=[row.get("latency_mean_ms", 0)],
                        y=[current_y],
                        mode="markers",
                        name=f"{model} ({dataset})",
                        marker=dict(
                            size=12,
                            color=dataset_colors.get(dataset, "#6b7280"),
                            symbol="diamond",
                        ),
                        hovertemplate=f"<b>{model} - {dataset}</b><br>"
                        + f"Mean: {row.get('latency_mean_ms', 0):.2f}ms<br>"
                        + f"Median: {row.get('latency_median_ms', 0):.2f}ms<br>"
                        + f"P95: {row.get('latency_p95_ms', 0):.2f}ms<br>"
                        + f"P99: {row.get('latency_p99_ms', 0):.2f}ms<extra></extra>",
                    )
                )

        current_y += 1  # Move to next row

    # Create custom y-axis labels
    y_labels = [label for label, _ in y_positions]
    y_values = [pos for _, pos in y_positions]

    # Update layout (axes pivoted, height tripled)
    fig.update_layout(
        title="Latency Distribution per Model",
        xaxis_title="Latency (ms)",
        yaxis_title="Model",
        showlegend=False,
        height=600,
        yaxis=dict(
            tickmode="array",
            tickvals=y_values,
            ticktext=y_labels,
        ),
    )

    return fig


def update_analysis_charts():
    """Update all analysis charts with latest data."""
    results_df = load_benchmark_results()
    summary = create_dashboard_summary(results_df)

    summary_md = f"""## 📊 Results Summary

- **Total Models:** {summary["total_models"]}
- **Total Datasets:** {summary["total_datasets"]}
- **Total Results:** {summary["total_results"]}
- **Last Updated:** {summary["last_updated"]}
- **Top Model:** {summary["top_model"]}

*💡 Use the **Reports** tab for data export and detailed analysis*"""

    scatter_fig = create_scatter_plot(results_df)
    accuracy_fig = create_accuracy_by_dataset_chart(results_df)
    f1_fig = create_f1_by_dataset_chart(results_df)
    latency_breakdown_fig = create_latency_breakdown_chart(results_df)
    ttft_fig = create_ttft_chart(results_df)
    throughput_accuracy_fig = create_throughput_accuracy_scatter(results_df)
    throughput_comparison_fig = create_throughput_comparison_chart(results_df)
    efficiency_bubble_fig = create_efficiency_bubble_chart(results_df)
    latency_distribution_fig = create_latency_distribution_chart(results_df)

    return (
        summary_md,
        scatter_fig,
        accuracy_fig,
        f1_fig,
        latency_breakdown_fig,
        ttft_fig,
        throughput_accuracy_fig,
        throughput_comparison_fig,
        efficiency_bubble_fig,
        latency_distribution_fig,
    )


def create_analysis_tab():
    """Create the analysis tab for the Gradio interface."""
    with gr.TabItem("📊 Analysis"):
        gr.Markdown("### Results Analysis")

        # Refresh button at the top
        refresh_btn = gr.Button("🔄 Refresh Analysis", variant="secondary")

        # Create updatable components with initial data
        initial_results = load_benchmark_results()
        initial_summary = create_dashboard_summary(initial_results)
        initial_scatter_fig = create_scatter_plot(initial_results)
        initial_accuracy_fig = create_accuracy_by_dataset_chart(initial_results)
        initial_f1_fig = create_f1_by_dataset_chart(initial_results)
        initial_latency_breakdown_fig = create_latency_breakdown_chart(initial_results)
        initial_ttft_fig = create_ttft_chart(initial_results)
        initial_throughput_accuracy_fig = create_throughput_accuracy_scatter(initial_results)
        initial_throughput_comparison_fig = create_throughput_comparison_chart(initial_results)
        initial_efficiency_bubble_fig = create_efficiency_bubble_chart(initial_results)
        initial_latency_distribution_fig = create_latency_distribution_chart(initial_results)

        initial_summary_md = f"""## 📊 Results Summary

- **Total Models:** {initial_summary["total_models"]}
- **Total Datasets:** {initial_summary["total_datasets"]}
- **Total Results:** {initial_summary["total_results"]}
- **Last Updated:** {initial_summary["last_updated"]}
- **Top Model:** {initial_summary["top_model"]}

*💡 Use the **Reports** tab for data export and detailed analysis*"""

        summary_text = gr.Markdown(value=initial_summary_md)

        # Row 1: Accuracy and Latency charts
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Model Accuracy by Dataset")
                accuracy_by_dataset_plot = gr.Plot(value=initial_accuracy_fig)
                with gr.Accordion("📖 What does this chart show?", open=False):
                    gr.Markdown("""
                    **Model Accuracy by Dataset** shows how well each model performs on different datasets.

                    - **Accuracy**: The percentage of predictions that are correct (higher is better)
                    - **Grouped bars**: Each model shows separate bars for different datasets
                    - **Color coding**: Different colors represent different datasets
                    - **Comparison**: Use this to see which models work best on specific types of data
                    """)

            with gr.Column():
                gr.Markdown("#### Latency vs Accuracy Scatter")
                scatter_plot = gr.Plot(value=initial_scatter_fig)
                with gr.Accordion("📖 What does this chart show?", open=False):
                    gr.Markdown("""
                    **Latency vs Accuracy Scatter** shows the trade-off between speed and accuracy.

                    - **X-axis**: Latency in milliseconds (lower is faster)
                    - **Y-axis**: Accuracy percentage (higher is better)
                    - **Points**: Individual benchmark runs (small dots)
                    - **Diamonds**: Average performance per model (large diamonds)
                    - **Color coding**: Different colors for different models
                    - **Insight**: Look for models in the "sweet spot" - high accuracy with low latency
                    """)

        # Row 2: F1 Score and Latency Breakdown
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Model F1 Score by Dataset")
                f1_by_dataset_plot = gr.Plot(value=initial_f1_fig)
                with gr.Accordion("📖 What does this chart show?", open=False):
                    gr.Markdown("""
                    **Model F1 Score by Dataset** shows balanced accuracy for positive/negative predictions.

                    - **F1 Score**: Harmonic mean of precision and recall (higher is better)
                    - **Balanced metric**: Better than accuracy when classes are imbalanced
                    - **Grouped bars**: Each model shows separate bars for different datasets
                    - **Color coding**: Different colors represent different datasets
                    - **Use case**: Important for sentiment analysis with uneven positive/negative samples
                    """)

            with gr.Column():
                gr.Markdown("#### Latency Breakdown per Model")
                latency_breakdown_plot = gr.Plot(value=initial_latency_breakdown_fig)
                with gr.Accordion("📖 What does this chart show?", open=False):
                    gr.Markdown("""
                    **Latency Breakdown** shows different latency percentiles for each model.

                    - **Mean**: Average latency across all predictions
                    - **Median**: Middle value when latencies are sorted
                    - **P95**: 95th percentile (95% of predictions are faster than this)
                    - **P99**: 99th percentile (99% of predictions are faster than this)
                    - **Grouped bars**: Each model shows multiple latency metrics
                    - **Insight**: P95/P99 show worst-case performance, important for real-time applications
                    """)

        # Row 3: TTFT and Throughput vs Accuracy
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Time to First Token (TTFT)")
                ttft_plot = gr.Plot(value=initial_ttft_fig)
                with gr.Accordion("📖 What does this chart show?", open=False):
                    gr.Markdown("""
                    **Time to First Token (TTFT)** measures how quickly models start generating responses.

                    - **TTFT**: Time from input to first output token (lower is better)
                    - **Important for UX**: Users perceive lag before any output appears
                    - **Grouped bars**: Each model shows TTFT for different datasets
                    - **Color coding**: Different colors represent different datasets
                    - **Real-time applications**: Critical for chatbots and interactive systems
                    """)

            with gr.Column():
                gr.Markdown("#### Throughput vs Accuracy")
                throughput_accuracy_plot = gr.Plot(value=initial_throughput_accuracy_fig)
                with gr.Accordion("📖 What does this chart show?", open=False):
                    gr.Markdown("""
                    **Throughput vs Accuracy** shows processing capacity vs quality trade-off.

                    - **X-axis**: Throughput in samples per second (higher is better)
                    - **Y-axis**: Accuracy percentage (higher is better)
                    - **Points**: Individual benchmark runs
                    - **Color coding**: Different colors for different models
                    - **Symbols**: Different shapes for different datasets
                    - **Batch processing**: Important for high-volume applications
                    """)

        # Row 4: Throughput Comparison and Efficiency Bubble
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Comparative Throughput Analysis")
                throughput_comparison_plot = gr.Plot(value=initial_throughput_comparison_fig)
                with gr.Accordion("📖 What does this chart show?", open=False):
                    gr.Markdown("""
                    **Comparative Throughput** shows processing speed across models.

                    - **Horizontal bars**: Throughput in samples per second
                    - **Grouped bars**: Each model shows throughput for different datasets
                    - **Color coding**: Different colors represent different datasets
                    - **Orientation**: Horizontal for easy comparison of model names
                    - **Batch efficiency**: Higher throughput = more efficient for batch processing
                    """)

            with gr.Column():
                gr.Markdown("#### Model Efficiency Chart")
                efficiency_bubble_plot = gr.Plot(value=initial_efficiency_bubble_fig)
                with gr.Accordion("📖 What does this chart show?", open=False):
                    gr.Markdown("""
                    **Efficiency Chart** combines multiple performance dimensions.

                    - **X-axis**: Latency in milliseconds (lower is faster)
                    - **Y-axis**: Accuracy percentage (higher is better)
                    - **Bubble size**: Throughput (larger = more efficient)
                    - **Color coding**: Different colors for different models
                    - **Hover data**: Additional metrics like F1 score and dataset
                    - **Multi-dimensional view**: Balance speed, accuracy, and capacity
                    """)

        # Row 5: Latency Distribution
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Latency Distribution per Model")
                latency_distribution_plot = gr.Plot(value=initial_latency_distribution_fig)
                with gr.Accordion("📖 What does this chart show?", open=False):
                    gr.Markdown("""
                    **Latency Distribution** shows how consistent model response times are.

                    - **Points**: Average latency for each model-dataset combination
                    - **Hover details**: Mean, median, P95, and P99 latency values
                    - **Consistency**: Lower spread between percentiles = more predictable performance
                    - **Real-time requirements**: P95/P99 important for SLA compliance
                    - **Outlier sensitivity**: Shows how models handle edge cases
                    """)

        # Connect refresh button to update all charts
        refresh_btn.click(
            fn=update_analysis_charts,
            inputs=[],
            outputs=[
                summary_text,
                scatter_plot,
                accuracy_by_dataset_plot,
                f1_by_dataset_plot,
                latency_breakdown_plot,
                ttft_plot,
                throughput_accuracy_plot,
                throughput_comparison_plot,
                efficiency_bubble_plot,
                latency_distribution_plot,
            ],
            queue=False,  # Don't queue the initial load
        )
