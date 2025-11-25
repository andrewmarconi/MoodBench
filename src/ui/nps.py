"""
NPS UI components for MoodBench Gradio interface.
"""

import json
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr


def load_nps_results():
    """Load NPS estimation results from the results directory."""
    results_dir = Path("experiments/results")
    if not results_dir.exists():
        return pd.DataFrame()

    # Find NPS result files
    nps_files = list(results_dir.glob("estimated_nps_results.json"))

    if not nps_files:
        return pd.DataFrame()

    # Load the most recent NPS results
    nps_file = max(nps_files, key=lambda x: x.stat().st_mtime)

    try:
        with open(nps_file, "r") as f:
            data = json.load(f)

        if "results" in data:
            return pd.DataFrame(data["results"])
        else:
            return pd.DataFrame(data)

    except Exception as e:
        print(f"Error loading NPS results: {e}")
        return pd.DataFrame()


def _prepare_stacked_chart_data(nps_df):
    """Prepare data for stacked bar chart."""
    if nps_df.empty:
        return None, None

    # Filter for Disneyland data
    dataset_col = "test_dataset" if "test_dataset" in nps_df.columns else "dataset"
    if "branch" in nps_df.columns:
        disneyland_data = nps_df[
            (nps_df[dataset_col] == "disneyland") & (nps_df["branch"].notna())
        ].copy()
    else:
        disneyland_data = nps_df[nps_df[dataset_col] == "disneyland"].copy()

    if disneyland_data.empty:
        return None, None

    has_branch_col = "branch" in disneyland_data.columns
    return disneyland_data, has_branch_col


def _create_plot_data_without_branches(disneyland_data):
    """Create plot data when no branch column is available."""
    aggregated_data = []
    for model_name, group in disneyland_data.groupby("model_name"):
        # Sum counts across all training datasets for this model
        total_promoters = group["promoters_count"].sum()
        total_passives = group["passives_count"].sum()
        total_detractors = group["detractors_count"].sum()
        total_samples = total_promoters + total_passives + total_detractors

        # Recalculate percentages so they add up to 100%
        if total_samples > 0:
            promoters_pct = (total_promoters / total_samples) * 100
            passives_pct = (total_passives / total_samples) * 100
            detractors_pct = (total_detractors / total_samples) * 100
            nps_score = promoters_pct - detractors_pct
        else:
            promoters_pct = passives_pct = detractors_pct = nps_score = 0

        aggregated_data.append(
            {
                "model_name": model_name,
                "promoters_count": total_promoters,
                "passives_count": total_passives,
                "detractors_count": total_detractors,
                "promoters_percent": promoters_pct,
                "passives_percent": passives_pct,
                "detractors_percent": detractors_pct,
                "nps_score": nps_score,
            }
        )

    # Prepare plot data from aggregated results
    plot_data = []
    for model_data in aggregated_data:
        model_name = model_data["model_name"]

        # Add detractors (bottom layer)
        plot_data.append(
            {
                "branch_model": model_name,
                "category": "Detractors",
                "percentage": model_data["detractors_percent"],
                "count": model_data["detractors_count"],
                "nps_score": model_data["nps_score"],
            }
        )

        # Add passives (middle layer)
        plot_data.append(
            {
                "branch_model": model_name,
                "category": "Passives",
                "percentage": model_data["passives_percent"],
                "count": model_data["passives_count"],
                "nps_score": model_data["nps_score"],
            }
        )

        # Add promoters (top layer)
        plot_data.append(
            {
                "branch_model": model_name,
                "category": "Promoters",
                "percentage": model_data["promoters_percent"],
                "count": model_data["promoters_count"],
                "nps_score": model_data["nps_score"],
            }
        )

    return plot_data


def _create_plot_data_with_branches(disneyland_data):
    """Create plot data when branch column is available."""
    plot_data = []
    for _, row in disneyland_data.iterrows():
        model_name = row["model_name"]
        branch = row["branch"].replace("Disneyland_", "")
        label = f"{branch}<br>{model_name}"

        # Add detractors (bottom layer)
        plot_data.append(
            {
                "branch_model": label,
                "category": "Detractors",
                "percentage": row["detractors_percent"],
                "count": row["detractors_count"],
                "nps_score": row["nps_score"],
            }
        )

        # Add passives (middle layer)
        plot_data.append(
            {
                "branch_model": label,
                "category": "Passives",
                "percentage": row["passives_percent"],
                "count": row["passives_count"],
                "nps_score": row["nps_score"],
            }
        )

        # Add promoters (top layer)
        plot_data.append(
            {
                "branch_model": label,
                "category": "Promoters",
                "percentage": row["promoters_percent"],
                "count": row["promoters_count"],
                "nps_score": row["nps_score"],
            }
        )

    return plot_data


def create_nps_stacked_chart(nps_df):
    """Create stacked bar chart showing NPS categories grouped by branch."""
    disneyland_data, has_branch_col = _prepare_stacked_chart_data(nps_df)
    if disneyland_data is None:
        fig = go.Figure()
        fig.update_layout(title="No NPS Data Available")
        return fig

    if has_branch_col:
        plot_data = _create_plot_data_with_branches(disneyland_data)
    else:
        plot_data = _create_plot_data_without_branches(disneyland_data)

    plot_df = pd.DataFrame(plot_data)

    # Create stacked bar chart with appropriate title and labels
    chart_title = (
        "NPS Categories Grouped by Branch" if has_branch_col else "NPS Categories by Model"
    )
    x_label = "Branch<br>Model" if has_branch_col else "Model"

    fig = px.bar(
        plot_df,
        x="branch_model",
        y="percentage",
        color="category",
        title=chart_title,
        labels={
            "branch_model": x_label,
            "percentage": "Percentage (%)",
            "category": "NPS Category",
        },
        color_discrete_map={
            "Promoters": "#22c55e",  # green
            "Passives": "#eab308",  # yellow
            "Detractors": "#ef4444",  # red
        },
    )

    fig.update_layout(xaxis_tickangle=-45, height=500, showlegend=True, barmode="stack")

    return fig


def create_nps_accuracy_heatmap(nps_df):
    """Create heatmap showing accuracy by training dataset and model."""
    if nps_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No NPS Data Available")
        return fig

    # Filter for valid accuracy data
    valid_data = nps_df.dropna(subset=["accuracy_percent"])

    if valid_data.empty:
        fig = go.Figure()
        fig.update_layout(title="No Accuracy Data Available")
        return fig

    # Group by model and training dataset, calculate mean accuracy
    heatmap_data = (
        valid_data.groupby(["model_name", "training_dataset"])["accuracy_percent"]
        .mean()
        .reset_index()
    )

    # Pivot to create matrix for heatmap
    heatmap_matrix = heatmap_data.pivot(
        index="model_name", columns="training_dataset", values="accuracy_percent"
    )

    # Fill NaN values with 0 for visualization
    heatmap_matrix = heatmap_matrix.fillna(0)

    # Create text labels for the heatmap
    text_labels = []
    for row in heatmap_matrix.values:
        text_row = []
        for val in row:
            if pd.isna(val) or val == 0:
                text_row.append("")
            else:
                text_row.append(f"{val:.1f}%")
        text_labels.append(text_row)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_matrix.values,
            x=heatmap_matrix.columns,
            y=heatmap_matrix.index,
            colorscale="RdYlGn",  # Red-Yellow-Green scale
            text=text_labels,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "black"},
            hoverongaps=False,
            hovertemplate="Model: %{y}<br>Training Dataset: %{x}<br>Accuracy: %{z:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Training Dataset",
        yaxis_title="Model",
        height=600,  # Increased from 500 to 600 for taller rows
        xaxis=dict(tickangle=-45),
    )

    return fig


def _create_actual_value_trace(category, actual_value):
    """Create a scatter trace for actual values."""
    return go.Scatter(
        x=[category],
        y=[actual_value],
        mode="markers+text",
        name=f"Actual {category.split(' ')[0]}",
        marker=dict(
            size=15,
            color="#8b5cf6",  # Purple/violet
            symbol="diamond",
            line=dict(width=3, color="white"),
        ),
        text=[f"Actual: {actual_value:.1f}" + ("%" if "%" in category else "")],
        textposition="top center",
        textfont=dict(size=10, color="#8b5cf6", weight="bold"),
        hovertemplate=f"<b>Actual {category}</b><br>"
        + f"Value: {actual_value:.1f}"
        + ("%" if "%" in category else "")
        + "<extra></extra>",
        showlegend=True,
    )


def _create_box_trace(category, values, color, nps_df):
    """Create a box trace for the given category."""
    # Create custom hover text for each data point
    hover_texts = []
    for j, val in enumerate(values):
        if j < len(nps_df):
            model_name = nps_df.iloc[j]["model_name"]
            training_ds = nps_df.iloc[j]["training_dataset"]
            hover_texts.append(
                f"<b>{category}</b><br>"
                f"Value: {val:.1f}" + ("%" if "%" in category else "") + "<br>"
                f"Model: {model_name}<br>"
                f"Training Dataset: {training_ds}<extra></extra>"
            )
        else:
            hover_texts.append(
                f"<b>{category}</b><br>"
                f"Value: {val:.1f}" + ("%" if "%" in category else "") + "<extra></extra>"
            )

    return go.Box(
        x=[category] * len(values),
        y=values,
        name=category,
        marker_color=color,
        boxmean=True,  # Show mean line
        hovertemplate=hover_texts,
        hoverlabel=dict(bgcolor="white", bordercolor=color),
    )


def _load_actual_nps_values():
    """Load and calculate actual NPS values from Disneyland dataset."""
    try:
        import yaml
        from pathlib import Path

        # Load dataset configuration
        config_path = Path("config/datasets.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        dataset_config = config["datasets"]["disneyland"]
        cache_dir = Path("data/raw")
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_cache_path = cache_dir / f"{dataset_config['dataset_id'].replace('/', '_')}.csv"

        if not local_cache_path.exists():
            return None

        try:
            df = pd.read_csv(local_cache_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(local_cache_path, encoding="latin-1")

        # Calculate actual values across all Disneyland parks
        disneyland_data = df[df["Branch"].str.contains("Disneyland", na=False)].copy()
        if disneyland_data.empty:
            return None

        promoters = len(disneyland_data[disneyland_data["Rating"] == 5])
        passives = len(disneyland_data[disneyland_data["Rating"] == 4])
        detractors = len(disneyland_data[disneyland_data["Rating"] <= 3])
        total = len(disneyland_data)

        if total == 0:
            return None

        actual_nps = ((promoters - detractors) / total) * 100
        actual_promoters_pct = (promoters / total) * 100
        actual_passives_pct = (passives / total) * 100
        actual_detractors_pct = (detractors / total) * 100

        return {
            "NPS Score": actual_nps,
            "Promoters (%)": actual_promoters_pct,
            "Passives (%)": actual_passives_pct,
            "Detractors (%)": actual_detractors_pct,
        }
    except Exception:
        return None


def create_nps_scatter_high_low(nps_df):
    """Create box plot showing range of values for NPS Score and each category with actual values."""
    if nps_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No NPS Data Available")
        return fig

    # Calculate actual values from Disneyland customer data
    actual_values = _load_actual_nps_values()
    actual_nps_value = actual_values.get("NPS Score") if actual_values else None

    # Prepare data for the 4 categories
    categories_data = {
        "NPS Score": nps_df["nps_score"].tolist(),
        "Promoters (%)": nps_df["promoters_percent"].tolist(),
        "Passives (%)": nps_df["passives_percent"].tolist(),
        "Detractors (%)": nps_df["detractors_percent"].tolist(),
    }

    # Create box plot
    fig = go.Figure()

    colors = ["#3b82f6", "#22c55e", "#eab308", "#ef4444"]  # Blue, Green, Yellow, Red

    for i, (category, values) in enumerate(categories_data.items()):
        if values:
            fig.add_trace(_create_box_trace(category, values, colors[i], nps_df))

    # Add actual values as prominent markers for each category
    if actual_values:
        for category, actual_value in actual_values.items():
            fig.add_trace(_create_actual_value_trace(category, actual_value))

    fig.update_layout(
        title="NPS Score & Category Ranges with Actual Values",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=800,  # Doubled from 400 to 800
        showlegend=False,
        boxmode="group",
    )

    return fig
    categories_data = {
        "NPS Score": nps_df["nps_score"].tolist(),
        "Promoters (%)": nps_df["promoters_percent"].tolist(),
        "Passives (%)": nps_df["passives_percent"].tolist(),
        "Detractors (%)": nps_df["detractors_percent"].tolist(),
    }

    # Create box plot
    fig = go.Figure()

    colors = ["#3b82f6", "#22c55e", "#eab308", "#ef4444"]  # Blue, Green, Yellow, Red

    for i, (category, values) in enumerate(categories_data.items()):
        if values:
            fig.add_trace(
                go.Box(
                    x=[category] * len(values),
                    y=values,
                    name=category,
                    marker_color=colors[i],
                    boxmean=True,  # Show mean line
                    hovertemplate=f"<b>{category}</b><br>"
                    + "Value: %{y:.1f}"
                    + ("%" if "%" in category else "")
                    + "<extra></extra>",
                )
            )

    # Add actual NPS reference line if available
    if actual_nps_value is not None:
        fig.add_trace(
            go.Scatter(
                x=["NPS Score"],  # Position on NPS Score category
                y=[actual_nps_value],
                mode="markers+text",
                name="Actual NPS (Customer Data)",
                marker=dict(
                    size=15,
                    color="#8b5cf6",  # Purple/violet
                    symbol="diamond",
                    line=dict(width=3, color="white"),
                ),
                text=[f"Actual NPS: {actual_nps_value:.1f}"],
                textposition="top center",
                textfont=dict(size=12, color="#8b5cf6", weight="bold"),
                hovertemplate="<b>Actual NPS (Customer Data)</b><br>"
                + f"Value: {actual_nps_value:.1f}<extra></extra>",
            )
        )

        # Add a reference line across the chart
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=3.5,  # Span all categories
            y0=actual_nps_value,
            y1=actual_nps_value,
            line=dict(color="#8b5cf6", width=2, dash="dot"),
            opacity=0.7,
        )

        # Add annotation
        fig.add_annotation(
            x=3.2,
            y=actual_nps_value,
            text=f"Actual NPS: {actual_nps_value:.1f}",
            showarrow=False,
            font=dict(size=10, color="#8b5cf6"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#8b5cf6",
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        title="NPS Score & Category Ranges with Actual NPS Reference",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
        boxmode="group",
    )

    return fig

    # Prepare data for the 4 categories
    categories_data = {
        "NPS Score": nps_df["nps_score"].tolist(),
        "Promoters (%)": nps_df["promoters_percent"].tolist(),
        "Passives (%)": nps_df["passives_percent"].tolist(),
        "Detractors (%)": nps_df["detractors_percent"].tolist(),
    }

    # Create box plot
    fig = go.Figure()

    colors = ["#3b82f6", "#22c55e", "#eab308", "#ef4444"]  # Blue, Green, Yellow, Red

    for i, (category, values) in enumerate(categories_data.items()):
        if values:
            fig.add_trace(
                go.Box(
                    x=[category] * len(values),
                    y=values,
                    name=category,
                    marker_color=colors[i],
                    boxmean=True,  # Show mean line
                    hovertemplate=f"<b>{category}</b><br>"
                    + "Value: %{y:.1f}"
                    + ("%" if "%" in category else "")
                    + "<extra></extra>",
                )
            )

    fig.update_layout(
        title="NPS Score & Category Ranges",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        boxmode="group",
    )

    return fig

    # Prepare data by branch
    branches = ["California", "Paris", "HongKong"]
    branch_data = {}

    # Calculate overall stats
    overall_promoters = []
    overall_passives = []
    overall_detractors = []

    for branch in branches:
        branch_key = f"Disneyland_{branch}"
        branch_rows = (
            nps_df[nps_df["branch"] == branch_key] if "branch" in nps_df.columns else nps_df
        )

        if not branch_rows.empty:
            promoters = branch_rows["promoters_percent"].tolist()
            passives = branch_rows["passives_percent"].tolist()
            detractors = branch_rows["detractors_percent"].tolist()

            branch_data[branch] = {
                "promoters": promoters,
                "passives": passives,
                "detractors": detractors,
            }

            overall_promoters.extend(promoters)
            overall_passives.extend(passives)
            overall_detractors.extend(detractors)

    # Add overall data
    if overall_promoters:
        branch_data["Overall"] = {
            "promoters": overall_promoters,
            "passives": overall_passives,
            "detractors": overall_detractors,
        }

    # Create box plot
    fig = go.Figure()

    categories = ["Promoters (%)", "Passives (%)", "Detractors (%)"]
    colors = ["#22c55e", "#eab308", "#ef4444"]  # Green, Yellow, Red

    for i, category in enumerate(categories):
        category_key = category.split(" ")[0].lower()  # "promoters", "passives", "detractors"

        x_values = []
        y_values = []

        for branch in ["California", "Paris", "HongKong", "Overall"]:
            if branch in branch_data:
                data = branch_data[branch][category_key]
                if data:
                    # Add box plot data
                    x_values.extend([branch] * len(data))
                    y_values.extend(data)

        if x_values and y_values:
            fig.add_trace(
                go.Box(
                    x=x_values,
                    y=y_values,
                    name=category,
                    marker_color=colors[i],
                    boxmean=True,  # Show mean line
                    hovertemplate=f"<b>{category}</b><br>"
                    + "Branch: %{x}<br>"
                    + "Value: %{y:.1f}%<br>"
                    + "<extra></extra>",
                )
            )

    fig.update_layout(
        title="NPS Category Ranges by Branch",
        xaxis_title="Branch",
        yaxis_title="Percentage (%)",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        boxmode="group",  # Group boxes by category
    )

    return fig

    # Prepare data for scatter plot
    scatter_data = []
    means = {"promoters": [], "passives": [], "detractors": []}

    for _, row in nps_df.iterrows():
        nps_score = row["nps_score"]
        promoters_pct = row["promoters_percent"]
        passives_pct = row["passives_percent"]
        detractors_pct = row["detractors_percent"]

        # Add individual data points
        scatter_data.extend(
            [
                {
                    "nps": nps_score,
                    "percentage": promoters_pct,
                    "category": "Promoters (%)",
                    "size": 4,
                },
                {
                    "nps": nps_score,
                    "percentage": passives_pct,
                    "category": "Passives (%)",
                    "size": 4,
                },
                {
                    "nps": nps_score,
                    "percentage": detractors_pct,
                    "category": "Detractors (%)",
                    "size": 4,
                },
            ]
        )

        # Collect data for means
        means["promoters"].append(promoters_pct)
        means["passives"].append(passives_pct)
        means["detractors"].append(detractors_pct)

    scatter_df = pd.DataFrame(scatter_data)

    # Calculate means
    mean_promoters = sum(means["promoters"]) / len(means["promoters"]) if means["promoters"] else 0
    mean_passives = sum(means["passives"]) / len(means["passives"]) if means["passives"] else 0
    mean_detractors = (
        sum(means["detractors"]) / len(means["detractors"]) if means["detractors"] else 0
    )

    # Create scatter plot
    fig = go.Figure()

    # Add individual data points
    for category in ["Promoters (%)", "Passives (%)", "Detractors (%)"]:
        category_data = scatter_df[scatter_df["category"] == category]
        fig.add_trace(
            go.Scatter(
                x=category_data["nps"],
                y=category_data["percentage"],
                mode="markers",
                name=f"{category} (individual)",
                marker=dict(
                    size=category_data["size"],
                    opacity=0.6,
                    color={
                        "Promoters (%)": "#22c55e",
                        "Passives (%)": "#eab308",
                        "Detractors (%)": "#ef4444",
                    }[category],
                ),
                hovertemplate=f"<b>{category}</b><br>"
                + "NPS Score: %{x:.1f}<br>"
                + "Percentage: %{y:.1f}%<extra></extra>",
            )
        )

    # Add mean lines
    mean_colors = {
        "Promoters (%)": "#16a34a",
        "Passives (%)": "#ca8a04",
        "Detractors (%)": "#dc2626",
    }
    mean_values = {
        "Promoters (%)": mean_promoters,
        "Passives (%)": mean_passives,
        "Detractors (%)": mean_detractors,
    }

    for category, mean_val in mean_values.items():
        fig.add_trace(
            go.Scatter(
                x=[scatter_df["nps"].min(), scatter_df["nps"].max()],
                y=[mean_val, mean_val],
                mode="lines",
                name=f"{category} (mean)",
                line=dict(color=mean_colors[category], width=3, dash="dash"),
                hovertemplate=f"<b>{category} Mean</b><br>"
                + "Average: {mean_val:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="NPS Score vs Category Percentages",
        xaxis_title="NPS Score",
        yaxis_title="Percentage (%)",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )

    return fig


def create_nps_summary_table(nps_df):
    """Create formatted summary table for NPS results."""
    if nps_df.empty:
        return pd.DataFrame()

    # Filter out entries with errors and select relevant columns
    valid_data = nps_df.dropna(subset=["nps_score"]).copy()

    if valid_data.empty:
        return pd.DataFrame()

    # Handle column name differences between old and new formats
    dataset_col = "test_dataset" if "test_dataset" in valid_data.columns else "dataset"

    # Format the table
    columns_to_select = [
        "model_name",
        dataset_col,
        "nps_score",
        "promoters_percent",
        "passives_percent",
        "detractors_percent",
    ]

    # Add training_dataset column if it exists (for new format)
    if "training_dataset" in valid_data.columns:
        columns_to_select.insert(2, "training_dataset")

    display_df = valid_data[columns_to_select].copy()

    # Rename columns for display
    column_names = [
        "Model",
        "Dataset",
        "NPS Score",
        "Promoters (%)",
        "Passives (%)",
        "Detractors (%)",
    ]
    if "training_dataset" in valid_data.columns:
        column_names.insert(2, "Training Dataset")

    display_df.columns = column_names

    # Format percentages
    for col in ["Promoters (%)", "Passives (%)", "Detractors (%)"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")

    # Format NPS score
    display_df["NPS Score"] = display_df["NPS Score"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
    )

    return display_df


def _prepare_initial_nps_data():
    """Prepare all initial data for NPS tab."""
    # Load initial data
    initial_nps_df = load_nps_results()

    # Calculate actual NPS for each branch
    california_nps = calculate_actual_nps_from_ratings(initial_nps_df, "California")
    paris_nps = calculate_actual_nps_from_ratings(initial_nps_df, "Paris")
    hongkong_nps = calculate_actual_nps_from_ratings(initial_nps_df, "HongKong")

    initial_california_gauge = create_nps_gauge_chart(california_nps, "California")
    initial_paris_gauge = create_nps_gauge_chart(paris_nps, "Paris")
    initial_hongkong_gauge = create_nps_gauge_chart(hongkong_nps, "Hong Kong")

    initial_california_chart = create_nps_stacked_chart_for_branch(initial_nps_df, "California")
    initial_paris_chart = create_nps_stacked_chart_for_branch(initial_nps_df, "Paris")
    initial_hongkong_chart = create_nps_stacked_chart_for_branch(initial_nps_df, "HongKong")

    # Get unique models for accuracy charts
    accuracy_chart_models = []
    if not initial_nps_df.empty:
        dataset_col = "test_dataset" if "test_dataset" in initial_nps_df.columns else "dataset"
        disneyland_data = initial_nps_df[
            (initial_nps_df[dataset_col] == "disneyland")
            & (initial_nps_df["accuracy_percent"].notna())
        ]
        if not disneyland_data.empty:
            # Get unique models, sorted alphabetically
            accuracy_chart_models = sorted(disneyland_data["model_name"].unique().tolist())

    return (
        initial_nps_df,
        california_nps,
        paris_nps,
        hongkong_nps,
        initial_california_gauge,
        initial_paris_gauge,
        initial_hongkong_gauge,
        initial_california_chart,
        initial_paris_chart,
        initial_hongkong_chart,
        accuracy_chart_models,
    )


def _load_disneyland_data():
    """Load Disneyland dataset from cache or download."""
    try:
        import kagglehub
        import yaml

        # Load dataset configuration
        config_path = Path("config/datasets.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        dataset_config = config["datasets"]["disneyland"]

        # Check for cached dataset
        cache_dir = Path("data/raw")
        cache_dir.mkdir(parents=True, exist_ok=True)

        dataset_slug = dataset_config["dataset_id"].replace("/", "_")
        local_cache_path = cache_dir / f"{dataset_slug}.csv"

        if local_cache_path.exists():
            # Load from cache
            try:
                df = pd.read_csv(local_cache_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(local_cache_path, encoding="latin-1")
        else:
            # Download and cache
            dataset_path = kagglehub.dataset_download(dataset_config["dataset_id"])

            # Find the CSV file
            csv_files = []
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(".csv"):
                        csv_files.append(os.path.join(root, file))

            if not csv_files:
                return None

            csv_path = csv_files[0]

            # Load CSV
            try:
                df = pd.read_csv(csv_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding="latin-1")

            # Cache locally
            df.to_csv(local_cache_path, index=False)

        return df

    except Exception as e:
        print(f"Error loading Disneyland data: {e}")
        return None


def _calculate_nps_from_ratings(ratings):
    """Calculate NPS metrics from star ratings."""
    promoters = len(ratings[ratings == 5])
    passives = len(ratings[ratings == 4])
    detractors = len(ratings[ratings <= 3])

    total_samples = len(ratings)

    if total_samples == 0:
        return None

    promoters_pct = (promoters / total_samples) * 100
    passives_pct = (passives / total_samples) * 100
    detractors_pct = (detractors / total_samples) * 100
    nps_score = promoters_pct - detractors_pct

    return {
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
        "promoters_percent": promoters_pct,
        "passives_percent": passives_pct,
        "detractors_percent": detractors_pct,
        "nps_score": nps_score,
        "total_samples": total_samples,
    }


def calculate_actual_nps_from_ratings(nps_df, branch):
    """Calculate actual NPS from customer ratings in the Disney dataset for a specific branch.

    Maps star ratings to NPS categories:
    - 5 stars → Promoter (9-10 on NPS scale)
    - 4 stars → Passive (7-8 on NPS scale)
    - 1-3 stars → Detractor (0-6 on NPS scale)

    Args:
        nps_df: DataFrame with NPS results (not used, but kept for compatibility)
        branch: Branch name (California, Paris, or HongKong)

    Returns:
        Dictionary with NPS metrics or None if data unavailable
    """
    df = _load_disneyland_data()
    if df is None:
        return None

    # Filter by branch
    branch_df = df[df["Branch"] == f"Disneyland_{branch}"].copy()

    if branch_df.empty:
        return None

    # Calculate NPS from star ratings
    nps_metrics = _calculate_nps_from_ratings(branch_df["Rating"])
    if nps_metrics is None:
        return None

    return {
        **nps_metrics,
        "branch": branch,
    }


def create_nps_gauge_chart(nps_data, branch_name):
    """Create a gauge chart showing NPS score for a branch."""
    if not nps_data:
        # Empty gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge",
                value=0,
                gauge={"axis": {"range": [-100, 100]}},
            )
        )
        fig.update_layout(height=300)
        return fig

    nps_score = nps_data["nps_score"]

    # Determine color based on NPS score
    if nps_score >= 30:
        color = "green"
    elif nps_score >= 0:
        color = "yellow"
    else:
        color = "red"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=nps_score,
            gauge={
                "axis": {"range": [-100, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [-100, 0], "color": "lightcoral"},
                    {"range": [0, 30], "color": "lightyellow"},
                    {"range": [30, 100], "color": "lightgreen"},
                ],
            },
        )
    )

    fig.update_layout(height=300, title=f"{branch_name} Actual NPS")
    return fig


def create_nps_stacked_chart_for_branch(nps_df, branch):
    """Create stacked bar chart showing NPS categories for a specific branch."""
    if nps_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No NPS Data for {branch}")
        return fig

    # Filter for specific branch
    dataset_col = "test_dataset" if "test_dataset" in nps_df.columns else "dataset"
    if "branch" in nps_df.columns:
        branch_data = nps_df[
            (nps_df[dataset_col] == "disneyland") & (nps_df["branch"] == f"Disneyland_{branch}")
        ].copy()
    else:
        # For aggregated results, use all Disneyland data (no branch filtering)
        branch_data = nps_df[nps_df[dataset_col] == "disneyland"].copy()

    if branch_data.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No Data for {branch}")
        return fig

    # Aggregate counts by model (across all training datasets)
    aggregated_data = []
    for model_name, group in branch_data.groupby("model_name"):
        # Sum counts across all training datasets for this model
        total_promoters = group["promoters_count"].sum()
        total_passives = group["passives_count"].sum()
        total_detractors = group["detractors_count"].sum()
        total_samples = total_promoters + total_passives + total_detractors

        # Recalculate percentages so they add up to 100%
        if total_samples > 0:
            promoters_pct = (total_promoters / total_samples) * 100
            passives_pct = (total_passives / total_samples) * 100
            detractors_pct = (total_detractors / total_samples) * 100
            nps_score = promoters_pct - detractors_pct
        else:
            promoters_pct = passives_pct = detractors_pct = nps_score = 0

        aggregated_data.append(
            {
                "model_name": model_name,
                "promoters_count": total_promoters,
                "passives_count": total_passives,
                "detractors_count": total_detractors,
                "promoters_percent": promoters_pct,
                "passives_percent": passives_pct,
                "detractors_percent": detractors_pct,
                "nps_score": nps_score,
            }
        )

    # Prepare data for stacked bar chart
    plot_data = []
    for model_data in aggregated_data:
        model_name = model_data["model_name"]

        # Add detractors (bottom layer)
        plot_data.append(
            {
                "model": model_name,
                "category": "Detractors",
                "percentage": model_data["detractors_percent"],
                "count": model_data["detractors_count"],
                "nps_score": model_data["nps_score"],
            }
        )

        # Add passives (middle layer)
        plot_data.append(
            {
                "model": model_name,
                "category": "Passives",
                "percentage": model_data["passives_percent"],
                "count": model_data["passives_count"],
                "nps_score": model_data["nps_score"],
            }
        )

        # Add promoters (top layer)
        plot_data.append(
            {
                "model": model_name,
                "category": "Promoters",
                "percentage": model_data["promoters_percent"],
                "count": model_data["promoters_count"],
                "nps_score": model_data["nps_score"],
            }
        )

    plot_df = pd.DataFrame(plot_data)

    # Create stacked bar chart
    fig = px.bar(
        plot_df,
        x="model",
        y="percentage",
        color="category",
        labels={"model": "Model", "percentage": "Percentage (%)", "category": "NPS Category"},
        color_discrete_map={
            "Promoters": "#22c55e",  # green
            "Passives": "#eab308",  # yellow
            "Detractors": "#ef4444",  # red
        },
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=280,
        showlegend=False,
        barmode="stack",
        margin=dict(t=10, b=40, l=40, r=10),
        title=f"{branch} Model NPS Categories",
    )

    return fig


def create_nps_accuracy_chart_by_model(nps_df, model_name):
    """Create bar chart showing accuracy by training dataset for a specific model.

    Args:
        nps_df: DataFrame with NPS results
        model_name: Model name (e.g., "BERT-tiny", "DistilBERT-base")

    Returns:
        Plotly figure with bars for each training dataset and an average line
    """
    if nps_df.empty:
        fig = go.Figure()
        fig.update_layout(margin=dict(t=10, b=40, l=40, r=10))
        return fig

    # Filter for Disneyland data with valid accuracy for this model
    disneyland_data = nps_df[
        (nps_df["test_dataset"] == "disneyland")
        & (nps_df["accuracy_percent"].notna())
        & (nps_df["model_name"] == model_name)
    ].copy()

    # Remove entries with errors
    if "error" in disneyland_data.columns:
        disneyland_data = disneyland_data[disneyland_data["error"].isna()]

    if disneyland_data.empty:
        fig = go.Figure()
        fig.update_layout(margin=dict(t=10, b=40, l=40, r=10))
        return fig

    # Group by training dataset and calculate average accuracy
    accuracy_data = []
    for dataset, group in disneyland_data.groupby("training_dataset"):
        avg_accuracy = group["accuracy_percent"].mean()
        accuracy_data.append({"training_dataset": dataset, "accuracy_percent": avg_accuracy})

    if not accuracy_data:
        fig = go.Figure()
        fig.update_layout(margin=dict(t=10, b=40, l=40, r=10))
        return fig

    accuracy_df = pd.DataFrame(accuracy_data)

    # Calculate overall average across all training datasets
    overall_avg = accuracy_df["accuracy_percent"].mean()

    # Calculate dynamic y-axis range
    min_accuracy = accuracy_df["accuracy_percent"].min()
    max_accuracy = 100

    # Set y-axis to start slightly below the minimum value (5% below or at least at 0)
    y_min = max(0, min(min_accuracy - 5, overall_avg - 10))

    # Color map for each training dataset
    color_map = {
        "imdb": "#3b82f6",  # blue
        "sst2": "#8b5cf6",  # purple
        "amazon": "#06b6d4",  # cyan
        "yelp": "#10b981",  # emerald
    }

    # Create bar chart
    fig = go.Figure()

    # Add bars for each training dataset
    for _, row in accuracy_df.iterrows():
        dataset = row["training_dataset"]
        accuracy = row["accuracy_percent"]
        fig.add_trace(
            go.Bar(
                x=[dataset],
                y=[accuracy],
                name=dataset,
                marker_color=color_map.get(dataset, "#3b82f6"),
                text=f"{accuracy:.1f}%",
                textposition="outside",
                textfont_size=9,
                showlegend=False,
            )
        )

    # Add average line
    fig.add_trace(
        go.Scatter(
            x=accuracy_df["training_dataset"].tolist(),
            y=[overall_avg] * len(accuracy_df),
            mode="lines",
            name=f"Average: {overall_avg:.1f}%",
            line=dict(color="red", width=2, dash="dash"),
            showlegend=True,
        )
    )

    fig.update_layout(
        xaxis_title="Training Dataset",
        yaxis_title="Accuracy (%)",
        xaxis_tickangle=0,
        height=280,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=10, b=40, l=40, r=10),
        yaxis=dict(range=[y_min, max_accuracy]),
        # title=f"{model_name} Accuracy by Training Dataset",
    )

    return fig


def create_nps_distribution_chart(nps_df):
    """Create stacked bar chart showing NPS category distribution."""
    if nps_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No NPS Data Available")
        return fig

    # Filter out entries with errors
    valid_data = nps_df.dropna(
        subset=["promoters_percent", "passives_percent", "detractors_percent"]
    )

    if valid_data.empty:
        fig = go.Figure()
        fig.update_layout(title="No Valid NPS Data Available")
        return fig

    # Prepare data for stacked bar chart
    plot_data = []
    for _, row in valid_data.iterrows():
        model_dataset = f"{row['model_name']}<br>({row['dataset']})"
        plot_data.extend(
            [
                {
                    "model_dataset": model_dataset,
                    "category": "Promoters",
                    "percentage": row["promoters_percent"],
                    "count": row["promoters_count"],
                },
                {
                    "model_dataset": model_dataset,
                    "category": "Passives",
                    "percentage": row["passives_percent"],
                    "count": row["passives_count"],
                },
                {
                    "model_dataset": model_dataset,
                    "category": "Detractors",
                    "percentage": row["detractors_percent"],
                    "count": row["detractors_count"],
                },
            ]
        )

    plot_df = pd.DataFrame(plot_data)

    # Create stacked bar chart
    fig = px.bar(
        plot_df,
        x="model_dataset",
        y="percentage",
        color="category",
        title="NPS Category Distribution by Model",
        labels={
            "model_dataset": "Model (Dataset)",
            "percentage": "Percentage (%)",
            "category": "NPS Category",
        },
        color_discrete_map={
            "Promoters": "#22c55e",  # green
            "Passives": "#eab308",  # yellow
            "Detractors": "#ef4444",  # red
        },
    )

    fig.update_layout(xaxis_tickangle=-45, height=300, showlegend=False, barmode="stack")

    return fig


def update_nps_dashboard():
    """Update NPS dashboard with latest results."""
    # Load latest NPS data
    nps_df = load_nps_results()

    # Get available model families from the data
    if not nps_df.empty:
        dataset_col = "test_dataset" if "test_dataset" in nps_df.columns else "dataset"
        # For cross-dataset evaluation, we don't have branch-specific data
        if "branch" in nps_df.columns:
            disneyland_data = nps_df[
                (nps_df[dataset_col] == "disneyland") & (nps_df["branch"].notna())
            ]
        else:
            disneyland_data = nps_df[nps_df[dataset_col] == "disneyland"]
        if not disneyland_data.empty:
            # Extract model families (e.g., "BERT" from "BERT-tiny")
            pass

    # Create visualizations showing all models

    nps_table = create_nps_summary_table(nps_df)
    accuracy_heatmap_fig = create_nps_accuracy_heatmap(nps_df)
    scatter_chart_fig = create_nps_scatter_high_low(nps_df)

    # Create summary markdown
    summary_md = "### No NPS Results Available\n\nRun NPS estimation first."

    # Calculate actual NPS for each branch
    california_nps = calculate_actual_nps_from_ratings(nps_df, "California")
    paris_nps = calculate_actual_nps_from_ratings(nps_df, "Paris")
    hongkong_nps = calculate_actual_nps_from_ratings(nps_df, "HongKong")

    california_gauge_fig = create_nps_gauge_chart(california_nps, "California")
    paris_gauge_fig = create_nps_gauge_chart(paris_nps, "Paris")
    hongkong_gauge_fig = create_nps_gauge_chart(hongkong_nps, "Hong Kong")

    california_chart_fig = create_nps_stacked_chart_for_branch(nps_df, "California")
    paris_chart_fig = create_nps_stacked_chart_for_branch(nps_df, "Paris")
    hongkong_chart_fig = create_nps_stacked_chart_for_branch(nps_df, "HongKong")

    if not nps_df.empty and not nps_table.empty:
        return (
            summary_md,
            california_gauge_fig,
            paris_gauge_fig,
            hongkong_gauge_fig,
            california_chart_fig,
            paris_chart_fig,
            hongkong_chart_fig,
            accuracy_heatmap_fig,
            scatter_chart_fig,
            nps_table,
            gr.Markdown(visible=False),
        )
    else:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No Data Available")
        return (
            summary_md,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            gr.Dataframe(visible=False),
            gr.Markdown(value="### No NPS data available. Run estimation first."),
        )


def create_nps_tab():
    """Create the NPS tab for the Gradio interface."""
    with gr.TabItem("🎯 e-NPS"):
        # Refresh button
        refresh_nps_btn = gr.Button("🔄 Refresh NPS Analysis", variant="secondary")

        # Information columns
        with gr.Row():
            with gr.Column():
                # Initialize with default value, will be updated by refresh
                current_results_md = gr.Markdown(value="#### Current Results\n\n*Loading data...*")
            with gr.Column():
                gr.Markdown("#### NPS Categories")
                gr.Markdown(
                    """
                **NPS Mapping:**
                - 5★ → Promoter (9-10 on NPS scale)
                - 4★ → Passive (7-8 on NPS scale)
                - 1-3★ → Detractor (0-6 on NPS scale)
                """
                )
            with gr.Column():
                gr.Markdown("#### NPS Formula")
                gr.Markdown(
                    """
                **NPS Score = % Promoters - % Detractors**

                *Range: -100 (all detractors) to +100 (all promoters)*
                """
                )

        # Load and prepare initial data
        initial_data = _prepare_initial_nps_data()
        (
            initial_nps_df,
            california_nps,
            paris_nps,
            hongkong_nps,
            initial_california_gauge,
            initial_paris_gauge,
            initial_hongkong_gauge,
            initial_california_chart,
            initial_paris_chart,
            initial_hongkong_chart,
            accuracy_chart_models,
        ) = initial_data

        # Create accuracy charts for all available models
        initial_nps_accuracy_charts = []
        for model_name in accuracy_chart_models:
            chart = create_nps_accuracy_chart_by_model(initial_nps_df, model_name)
            initial_nps_accuracy_charts.append((model_name, chart))

        initial_nps_table = create_nps_summary_table(initial_nps_df)
        initial_accuracy_heatmap = create_nps_accuracy_heatmap(initial_nps_df)
        initial_scatter_chart = create_nps_scatter_high_low(initial_nps_df)

        # Create summary markdown
        if not initial_nps_df.empty:
            dataset_col = "test_dataset" if "test_dataset" in initial_nps_df.columns else "dataset"
            if "branch" in initial_nps_df.columns:
                disneyland_results = initial_nps_df[
                    (initial_nps_df[dataset_col] == "disneyland")
                    & (initial_nps_df["branch"].notna())
                ].copy()
            else:
                disneyland_results = initial_nps_df[
                    initial_nps_df[dataset_col] == "disneyland"
                ].copy()

            if not disneyland_results.empty:
                # Calculate stats across all branches/models
                avg_nps = disneyland_results["nps_score"].mean()  # noqa: F841
                has_branch_col = "branch" in disneyland_results.columns

                # Find best performing entry (branch or model)
                best_nps = float("-inf")
                best_identifier = "N/A"  # noqa: F841
                for idx, row in disneyland_results.iterrows():
                    if row["nps_score"] > best_nps:
                        best_nps = row["nps_score"]
                        if has_branch_col and pd.notna(row.get("branch")):
                            best_identifier = str(row["branch"]).replace("Disneyland_", "")
                        else:
                            # Use model name and training dataset instead
                            model = row["model_name"]
                            training_ds = row.get("training_dataset", "unknown")
                            best_identifier = f"{model} (trained on {training_ds})"  # noqa: F841

                best_nps_score = best_nps  # noqa: F841
                avg_accuracy = disneyland_results["accuracy_percent"].mean()  # noqa: F841

        # NPS Analysis by Location
        gr.Markdown("#### NPS Analysis by Location")

        # California Row
        gr.Markdown("##### 🇺🇸 California")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Actual NPS Score**<br>(from customer ratings)")
                california_gauge = gr.Plot(value=initial_california_gauge)
            with gr.Column(scale=2):
                gr.Markdown("**NPS Categories by Model**<br>(estimated from predictions)")
                california_chart = gr.Plot(value=initial_california_chart)

        # Paris Row
        gr.Markdown("##### 🇫🇷 Paris")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Actual NPS Score**<br>(from customer ratings)")
                paris_gauge = gr.Plot(value=initial_paris_gauge)
            with gr.Column(scale=2):
                gr.Markdown("**NPS Categories by Model**<br>(estimated from predictions)")
                paris_chart = gr.Plot(value=initial_paris_chart)

        # Hong Kong Row
        gr.Markdown("##### 🇭🇰 Hong Kong")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Actual NPS Score**<br>(from customer ratings)")
                hongkong_gauge = gr.Plot(value=initial_hongkong_gauge)
            with gr.Column(scale=2):
                gr.Markdown("**NPS Categories by Model**<br>(estimated from predictions)")
                hongkong_chart = gr.Plot(value=initial_hongkong_chart)

        # Accuracy Charts by Model
        gr.Markdown("#### Model Accuracy by Training Dataset")
        gr.Markdown("*Comparison of how each model performs when trained on different datasets*")

        # Create charts in rows of 3
        accuracy_chart_components = []
        num_charts = len(initial_nps_accuracy_charts)

        for i in range(0, num_charts, 3):
            with gr.Row():
                for j in range(3):
                    idx = i + j
                    if idx < num_charts:
                        model_name, chart = initial_nps_accuracy_charts[idx]
                        with gr.Column(scale=1):
                            name_component = gr.Markdown(value=f"**{model_name}**")
                            chart_component = gr.Plot(value=chart)
                            accuracy_chart_components.append((name_component, chart_component))
                    else:
                        # Placeholder empty column
                        with gr.Column(scale=1):
                            pass

        with gr.Accordion("📖 What do these charts show?", open=False):
            gr.Markdown(
                """
            **NPS Categories by Location** shows the breakdown of predictions into NPS categories for each Disneyland park separately.

            - **Promoters (Green)**: High-confidence positive predictions (9-10 on NPS scale)
            - **Passives (Yellow)**: Medium-confidence positive or uncertain predictions (7-8)
            - **Detractors (Red)**: Negative predictions (0-6)
            - **Separate charts**: One chart per Disneyland location (California, Paris, Hong Kong)
            - **Model comparison**: Each location chart compares all evaluated models
            """
            )

        # Accuracy Charts Row
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Model Accuracy Heatmap")
                gr.Markdown("*Performance across training datasets*")
                accuracy_heatmap = gr.Plot(value=initial_accuracy_heatmap)
            with gr.Column():
                gr.Markdown("#### NPS Score vs Category Percentages")
                gr.Markdown("*Individual values with category means*")
                scatter_chart = gr.Plot(value=initial_scatter_chart)

        # NPS Results Table
        if not initial_nps_table.empty:
            nps_results_table = gr.Dataframe(value=initial_nps_table, label="NPS Results Summary")
            no_data_message = gr.Markdown(visible=False)
        else:
            nps_results_table = gr.Dataframe(visible=False)
            no_data_message = gr.Markdown(
                value="""
### No NPS Data Available

Run NPS estimation first:

    moodbench estimated-nps --all-models --all-datasets

This will evaluate all trained models and estimate NPS from their predictions.
"""
            )

        # Connect refresh button (updates summary and main charts)
        refresh_nps_btn.click(
            fn=update_nps_dashboard,
            inputs=[],
            outputs=[
                current_results_md,  # First column with current results
                california_gauge,
                paris_gauge,
                hongkong_gauge,
                california_chart,
                paris_chart,
                hongkong_chart,
                accuracy_heatmap,
                scatter_chart,
                nps_results_table,
                no_data_message,
            ],
        )
