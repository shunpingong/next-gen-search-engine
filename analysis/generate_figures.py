from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


WORKBOOK_NAME = "FYP Results.xlsx"
RAW_SHEET_NAME = "Raw Query-Level Records"
FIGURE_SIZE = (8, 5)
FIGURE_DPI = 300
CONFIG_ORDER = ["LLM-only", "RAG", "TAG"]
QUERY_TYPE_ORDER = ["Dynamic", "Static"]
MODEL_NARRATIVE_ORDER = [
    "GPT-4o",
    "DeepSeek-R1",
    "Llama 3.1 70B Instruct",
    "Claude 3.5 Sonnet",
]
MODEL_LABELS = {
    "GPT-4o": "GPT-4o",
    "DeepSeek-R1": "DeepSeek-R1",
    "Llama 3.1 70B Instruct": "Llama 3.1 70B",
    "Claude 3.5 Sonnet": "Claude 3.5 Sonnet",
}
TOOL_ROUTE_ORDER = ["Detailed", "Basic", "None"]
TOOL_DYNAMIC_ORDER = ["Basic", "Detailed", "None"]

CONFIG_COLORS = {
    "LLM-only": "#5B9BD5",
    "RAG": "#ED7D31",
    "TAG": "#70AD47",
}

TOOL_COLORS = {
    "Detailed": "#4F8C3A",
    "Basic": "#9FD37C",
    "None": "#C9C9C9",
}

LATENCY_COLORS = {
    "TAG\n(retrieval skipped)": "#00B050",
    "LLM-only\n(overall)": "#5B9BD5",
    "RAG\n(overall)": "#ED7D31",
    "TAG\n(overall)": "#70AD47",
    "TAG\n(retrieval active)": "#FF0000",
}

CONFIG_CODE_MAP = {
    "C1": "LLM-only",
    "C2": "RAG",
    "C3": "TAG",
    "LLM": "LLM-only",
    "LLM-ONLY": "LLM-only",
    "LLM ONLY": "LLM-only",
    "RAG": "RAG",
    "LLM + RAG": "RAG",
    "LLM+RAG": "RAG",
    "TAG": "TAG",
    "LLM + TAG": "TAG",
    "LLM+TAG": "TAG",
}


@dataclass
class MetricsBundle:
    overall_accuracy: pd.Series
    overall_coherence: pd.Series
    overall_latency: pd.Series
    query_accuracy: pd.DataFrame
    error_rates: pd.DataFrame
    dynamic_accuracy_by_model: pd.DataFrame
    latency_rows: list[tuple[str, float]]
    model_order: list[str]
    tag_tool_distribution: pd.DataFrame
    dynamic_accuracy_by_tool: pd.DataFrame
    retrieval_rate_by_config: pd.Series
    retrieved_accuracy_by_config: pd.Series
    refinement_accuracy: pd.Series
    gpt4o_tag_improvement_pp: float
    claude_tag_error: float


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.figsize": FIGURE_SIZE,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#B8B8B8",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
        }
    )


def find_header_row(raw_sheet: pd.DataFrame) -> int:
    required_tokens = {
        "Query_ID",
        "Query_Type",
        "Query_Text",
        "Model_Name",
        "Configuration",
        "Factual_Accuracy",
        "Latency_Total (s)",
    }

    for row_index in raw_sheet.index:
        row_tokens = set(raw_sheet.loc[row_index].dropna().astype(str).str.strip().tolist())
        if required_tokens.issubset(row_tokens):
            return int(row_index)

    raise ValueError(
        "Could not locate the query-level header row in the workbook."
    )


def normalize_configuration(value: object) -> str:
    if pd.isna(value):
        return "Unknown"

    raw_value = str(value).strip()
    return CONFIG_CODE_MAP.get(raw_value.upper(), raw_value)


def load_query_level_records(workbook_path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(workbook_path)
    sheet_name = RAW_SHEET_NAME if RAW_SHEET_NAME in workbook.sheet_names else workbook.sheet_names[0]
    raw_sheet = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)
    header_row_index = find_header_row(raw_sheet)

    headers = [str(cell).strip() for cell in raw_sheet.loc[header_row_index].tolist()]
    records = raw_sheet.loc[header_row_index + 1 :].copy()
    records.columns = headers
    records = records.loc[:, [column for column in records.columns if not str(column).startswith("Unnamed")]]
    records = records.dropna(how="all").reset_index(drop=True)

    text_columns = [
        "Query_ID",
        "Query_Type",
        "Model_Name",
        "Configuration",
        "Tool_Invoked",
        "Query_Refined",
    ]
    for column in text_columns:
        if column in records.columns:
            records[column] = records[column].astype("string").str.strip()

    numeric_columns = [
        "Factual_Accuracy",
        "Coherence_Score",
        "Latency_Total (s)",
        "Retrieval_Time (s)",
        "Ranking_Time (s)",
        "Num_Documents_Retrieved",
    ]
    for column in numeric_columns:
        if column in records.columns:
            records[column] = pd.to_numeric(records[column], errors="coerce")

    records["Configuration"] = records["Configuration"].apply(normalize_configuration)
    records["Query_Type"] = records["Query_Type"].replace(
        {"static": "Static", "dynamic": "Dynamic", "STATIC": "Static", "DYNAMIC": "Dynamic"}
    )

    records["Tool_Invoked"] = (
        records["Tool_Invoked"]
        .fillna("None")
        .replace("<NA>", "None")
        .astype("string")
        .str.strip()
        .replace("", "None")
    )
    records["Tool_Used"] = (
        records["Tool_Invoked"].str.lower().ne("none")
        | records["Num_Documents_Retrieved"].fillna(0).gt(0)
    )
    records["Retrieved"] = records["Num_Documents_Retrieved"].fillna(0).gt(0)
    records["Is_Error"] = records["Factual_Accuracy"].eq(0).astype(int)

    required_columns = [
        "Model_Name",
        "Configuration",
        "Query_Type",
    ]
    missing_columns = [column for column in required_columns if column not in records.columns]
    if missing_columns:
        raise ValueError(f"Workbook is missing required columns: {missing_columns}")

    return records.dropna(subset=required_columns).reset_index(drop=True)


def compute_metrics(records: pd.DataFrame) -> MetricsBundle:
    overall_accuracy = (
        records.groupby("Configuration")["Factual_Accuracy"]
        .mean()
        .reindex(CONFIG_ORDER)
    )
    overall_coherence = (
        records.groupby("Configuration")["Coherence_Score"]
        .mean()
        .reindex(CONFIG_ORDER)
    )
    overall_latency = (
        records.groupby("Configuration")["Latency_Total (s)"]
        .mean()
        .reindex(CONFIG_ORDER)
    )

    query_accuracy = (
        records.groupby(["Query_Type", "Configuration"])["Factual_Accuracy"]
        .mean()
        .unstack("Configuration")
        .reindex(index=QUERY_TYPE_ORDER, columns=CONFIG_ORDER)
    )

    error_rates = (
        records.groupby(["Model_Name", "Configuration"])["Is_Error"]
        .mean()
        .mul(100)
        .reset_index(name="Error_Rate")
    )
    error_matrix = error_rates.pivot(index="Model_Name", columns="Configuration", values="Error_Rate")
    error_matrix = error_matrix.reindex(columns=CONFIG_ORDER)
    model_order = (
        (error_matrix["LLM-only"] - error_matrix["TAG"])
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    dynamic_accuracy_by_model = (
        records[records["Query_Type"] == "Dynamic"]
        .groupby(["Model_Name", "Configuration"])["Factual_Accuracy"]
        .mean()
        .unstack("Configuration")
        .reindex(index=MODEL_NARRATIVE_ORDER, columns=CONFIG_ORDER)
    )

    tag_records = records[records["Configuration"] == "TAG"].copy()
    rag_records = records[records["Configuration"] == "RAG"].copy()
    tag_active = tag_records[tag_records["Tool_Used"]]
    tag_skipped = tag_records[~tag_records["Tool_Used"]]

    tag_tool_distribution = (
        tag_records["Tool_Invoked"]
        .value_counts()
        .reindex(TOOL_ROUTE_ORDER)
        .rename_axis("Tool_Invoked")
        .reset_index(name="Count")
    )
    tag_tool_distribution["Share"] = tag_tool_distribution["Count"] / tag_tool_distribution["Count"].sum()

    dynamic_accuracy_by_tool = (
        tag_records[tag_records["Query_Type"] == "Dynamic"]
        .groupby("Tool_Invoked")["Factual_Accuracy"]
        .agg(Count="count", Accuracy="mean")
        .reindex(TOOL_DYNAMIC_ORDER)
        .reset_index()
    )

    retrieval_rate_by_config = pd.Series(
        {
            "RAG": float(rag_records["Retrieved"].mean()),
            "TAG": float(tag_records["Retrieved"].mean()),
        }
    )
    retrieved_accuracy_by_config = pd.Series(
        {
            "RAG": float(rag_records[rag_records["Retrieved"]]["Factual_Accuracy"].mean()),
            "TAG": float(tag_active["Factual_Accuracy"].mean()),
        }
    )

    refinement_mask = (
        tag_records["Query_Refined"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("yes")
    )
    refinement_accuracy = pd.Series(
        {
            "Unrefined TAG": float(tag_records[~refinement_mask]["Factual_Accuracy"].mean()),
            "Refined TAG": float(tag_records[refinement_mask]["Factual_Accuracy"].mean()),
        }
    )

    latency_rows = sorted(
        [
            ("LLM-only\n(overall)", float(overall_latency["LLM-only"])),
            ("RAG\n(overall)", float(overall_latency["RAG"])),
            ("TAG\n(overall)", float(overall_latency["TAG"])),
            ("TAG\n(retrieval active)", float(tag_active["Latency_Total (s)"].mean())),
            ("TAG\n(retrieval skipped)", float(tag_skipped["Latency_Total (s)"].mean())),
        ],
        key=lambda row: row[1],
    )

    return MetricsBundle(
        overall_accuracy=overall_accuracy,
        overall_coherence=overall_coherence,
        overall_latency=overall_latency,
        query_accuracy=query_accuracy,
        error_rates=error_rates,
        dynamic_accuracy_by_model=dynamic_accuracy_by_model,
        latency_rows=latency_rows,
        model_order=model_order,
        tag_tool_distribution=tag_tool_distribution,
        dynamic_accuracy_by_tool=dynamic_accuracy_by_tool,
        retrieval_rate_by_config=retrieval_rate_by_config,
        retrieved_accuracy_by_config=retrieved_accuracy_by_config,
        refinement_accuracy=refinement_accuracy,
        gpt4o_tag_improvement_pp=float(
            error_matrix.loc["GPT-4o", "LLM-only"] - error_matrix.loc["GPT-4o", "TAG"]
        ),
        claude_tag_error=float(error_matrix.loc["Claude 3.5 Sonnet", "TAG"]),
    )


def legend_handles(labels: list[str] | None = None) -> list[Patch]:
    names = labels or CONFIG_ORDER
    return [Patch(facecolor=CONFIG_COLORS[name], edgecolor="none", label=name) for name in names]


def style_axes(ax: plt.Axes, grid_axis: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B0B0B0")
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.tick_params(colors="#333333")
    ax.set_axisbelow(True)

    if grid_axis == "y":
        ax.yaxis.grid(True, alpha=0.75)
        ax.xaxis.grid(False)
    elif grid_axis == "x":
        ax.xaxis.grid(True, alpha=0.75)
        ax.yaxis.grid(False)
    else:
        ax.grid(False)


def add_vertical_labels(ax: plt.Axes, bars, values: list[float], y_offset: float, fmt: str) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def add_horizontal_labels(ax: plt.Axes, bars, values: list[float], x_offset: float, fmt: str) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            value + x_offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            ha="left",
            va="center",
            fontsize=9,
        )


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    layout_rect: tuple[float, float, float, float] | None = None,
) -> None:
    if layout_rect is None:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=layout_rect)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def create_figure_1(metrics: MetricsBundle, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), dpi=FIGURE_DPI)
    positions = list(range(len(CONFIG_ORDER)))
    colors = [CONFIG_COLORS[config] for config in CONFIG_ORDER]
    panels = [
        (
            "Factual Accuracy",
            metrics.overall_accuracy,
            "Accuracy",
            (0, 0.6),
            0.012,
            "{:.2f}",
            MultipleLocator(0.1),
        ),
        (
            "Coherence",
            metrics.overall_coherence,
            "Mean Coherence (out of 5)",
            (0, 5.0),
            0.06,
            "{:.2f}",
            MultipleLocator(1.0),
        ),
        (
            "Latency",
            metrics.overall_latency,
            "Mean Latency (s)",
            (0, 50),
            1.0,
            "{:.2f}",
            MultipleLocator(10),
        ),
    ]

    for ax, (panel_title, series, y_label, y_limits, label_offset, label_fmt, locator) in zip(axes, panels):
        values = [float(series[config]) for config in CONFIG_ORDER]
        bars = ax.bar(
            positions,
            values,
            width=0.58,
            color=colors,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax.set_title(panel_title, pad=10, fontsize=12, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(CONFIG_ORDER, rotation=18)
        ax.set_ylabel(y_label)
        ax.set_ylim(*y_limits)
        ax.yaxis.set_major_locator(locator)
        style_axes(ax, grid_axis="y")
        add_vertical_labels(ax, bars, values, y_offset=label_offset, fmt=label_fmt)

    # fig.suptitle("Overall Performance Comparison by Configuration", fontsize=14, fontweight="bold", y=1.03)
    fig.legend(
        handles=legend_handles(),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    save_figure(fig, output_dir / "fig1_accuracy_overall.png", layout_rect=(0.0, 0.06, 1.0, 0.97))


def create_figure_2(metrics: MetricsBundle, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    group_centers = [0.0, 1.55]
    bar_width = 0.24
    offsets = [-bar_width, 0.0, bar_width]

    ax.axvspan(
        group_centers[0] - 0.56,
        group_centers[0] + 0.56,
        color="#EEF8F0",
        alpha=0.95,
        zorder=0,
    )

    bars_by_config: dict[str, object] = {}
    for config, offset in zip(CONFIG_ORDER, offsets):
        positions = [group_centers[0] + offset, group_centers[1] + offset]
        values = [
            float(metrics.query_accuracy.loc["Dynamic", config]),
            float(metrics.query_accuracy.loc["Static", config]),
        ]
        bars = ax.bar(
            positions,
            values,
            width=bar_width * 0.96,
            color=CONFIG_COLORS[config],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        bars_by_config[config] = bars
        add_vertical_labels(ax, bars, values, y_offset=0.018, fmt="{:.2f}")

    # ax.set_title("Factual Accuracy by Query Type and Configuration", pad=14)
    ax.set_ylabel("Factual Accuracy")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(["Dynamic Queries", "Static Queries"])
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    style_axes(ax, grid_axis="y")

    ax.legend(
        handles=legend_handles(),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )

    save_figure(fig, output_dir / "fig2_dynamic_static.png")


def create_figure_3(metrics: MetricsBundle, output_dir: Path) -> None:
    error_table = (
        metrics.error_rates.pivot(index="Model_Name", columns="Configuration", values="Error_Rate")
        .reindex(index=metrics.model_order, columns=CONFIG_ORDER)
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    centers = list(range(len(metrics.model_order)))
    bar_height = 0.22
    llm_y = [center - bar_height for center in centers]
    rag_y = centers
    tag_y = [center + bar_height for center in centers]

    llm_values = error_table["LLM-only"].tolist()
    rag_values = error_table["RAG"].tolist()
    tag_values = error_table["TAG"].tolist()

    llm_bars = ax.barh(
        llm_y,
        llm_values,
        height=bar_height,
        color=CONFIG_COLORS["LLM-only"],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    rag_bars = ax.barh(
        rag_y,
        rag_values,
        height=bar_height,
        color=CONFIG_COLORS["RAG"],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    tag_bars = ax.barh(
        tag_y,
        tag_values,
        height=bar_height,
        color=CONFIG_COLORS["TAG"],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )

    # ax.set_title("Error Rate (%) by Model and Configuration", pad=14)
    ax.set_xlabel("Error Rate (%)")
    ax.set_ylabel("Model")
    ax.set_yticks(centers)
    ax.set_yticklabels(metrics.model_order)
    ax.set_xlim(0, 75)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.invert_yaxis()
    style_axes(ax, grid_axis="x")

    add_horizontal_labels(ax, llm_bars, llm_values, x_offset=0.8, fmt="{:.0f}%")
    add_horizontal_labels(ax, rag_bars, rag_values, x_offset=0.8, fmt="{:.0f}%")
    add_horizontal_labels(ax, tag_bars, tag_values, x_offset=0.8, fmt="{:.0f}%")

    ax.axvline(50, color="#7A7A7A", linestyle="--", linewidth=1.1, zorder=2)
    ax.legend(
        handles=legend_handles(),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )

    save_figure(fig, output_dir / "fig3_error_rates.png")


def create_figure_4(metrics: MetricsBundle, output_dir: Path) -> None:
    labels = [row[0] for row in metrics.latency_rows]
    values = [row[1] for row in metrics.latency_rows]
    colors = [LATENCY_COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    positions = list(range(len(labels)))
    bars = ax.barh(
        positions,
        values,
        height=0.6,
        color=colors,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )

    # ax.set_title("Latency Comparison Across Configurations and Retrieval Conditions", pad=14)
    ax.set_xlabel("Mean Latency (seconds)")
    ax.set_ylabel("Condition")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 70)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.invert_yaxis()
    style_axes(ax, grid_axis="x")
    add_horizontal_labels(ax, bars, values, x_offset=0.9, fmt="{:.2f}")

    llm_baseline = next(value for label, value in metrics.latency_rows if label == "LLM-only\n(overall)")
    ax.axvline(llm_baseline, color="#7A7A7A", linestyle="--", linewidth=1.1, zorder=2)
    save_figure(fig, output_dir / "fig4_latency.png")


def create_figure_5(metrics: MetricsBundle, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    models = MODEL_NARRATIVE_ORDER
    x_positions = list(range(len(models)))
    bar_width = 0.22
    offsets = [-bar_width, 0.0, bar_width]

    for config, offset in zip(CONFIG_ORDER, offsets):
        values = [
            float(metrics.dynamic_accuracy_by_model.loc[model, config])
            for model in models
        ]
        bars = ax.bar(
            [position + offset for position in x_positions],
            values,
            width=bar_width * 0.95,
            color=CONFIG_COLORS[config],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        add_vertical_labels(ax, bars, values, y_offset=0.018, fmt="{:.2f}")

    # ax.set_title("Dynamic Query Accuracy by Model and Configuration", pad=14)
    ax.set_ylabel("Dynamic Accuracy")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([MODEL_LABELS[model] for model in models])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    style_axes(ax, grid_axis="y")

    ax.legend(
        handles=legend_handles(),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )

    save_figure(fig, output_dir / "fig5_dynamic_by_model.png")


def create_figure_6(metrics: MetricsBundle, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=FIGURE_DPI)

    routing_ax = axes[0]
    route_labels = metrics.tag_tool_distribution["Tool_Invoked"].tolist()
    route_values = metrics.tag_tool_distribution["Share"].astype(float).tolist()
    route_counts = metrics.tag_tool_distribution["Count"].astype(int).tolist()
    route_colors = [TOOL_COLORS[label] for label in route_labels]
    route_y = list(range(len(route_labels)))

    route_bars = routing_ax.barh(
        route_y,
        route_values,
        color=route_colors,
        height=0.6,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    routing_ax.set_title("TAG Query Routing", pad=10, fontsize=12, fontweight="bold")
    routing_ax.set_xlabel("Share of TAG Queries")
    routing_ax.set_yticks(route_y)
    routing_ax.set_yticklabels([f"{label} (n={count})" for label, count in zip(route_labels, route_counts)])
    routing_ax.set_xlim(0, 0.65)
    routing_ax.xaxis.set_major_locator(MultipleLocator(0.1))
    style_axes(routing_ax, grid_axis="x")
    add_horizontal_labels(routing_ax, route_bars, route_values, x_offset=0.012, fmt="{:.1%}")
    routing_ax.invert_yaxis()

    dynamic_ax = axes[1]
    dynamic_labels = metrics.dynamic_accuracy_by_tool["Tool_Invoked"].tolist()
    dynamic_values = metrics.dynamic_accuracy_by_tool["Accuracy"].astype(float).tolist()
    dynamic_counts = metrics.dynamic_accuracy_by_tool["Count"].astype(int).tolist()
    dynamic_colors = [TOOL_COLORS[label] for label in dynamic_labels]
    dynamic_x = list(range(len(dynamic_labels)))

    dynamic_bars = dynamic_ax.bar(
        dynamic_x,
        dynamic_values,
        color=dynamic_colors,
        width=0.58,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    dynamic_ax.set_title("Dynamic Accuracy by TAG Tool Route", pad=10, fontsize=12, fontweight="bold")
    dynamic_ax.set_ylabel("Dynamic Accuracy")
    dynamic_ax.set_xticks(dynamic_x)
    dynamic_ax.set_xticklabels([f"{label}\n(n={count})" for label, count in zip(dynamic_labels, dynamic_counts)])
    dynamic_ax.set_ylim(0, 1.08)
    dynamic_ax.yaxis.set_major_locator(MultipleLocator(0.2))
    style_axes(dynamic_ax, grid_axis="y")
    add_vertical_labels(dynamic_ax, dynamic_bars, dynamic_values, y_offset=0.02, fmt="{:.2f}")

    # fig.suptitle("TAG Tool Behaviour", fontsize=14, fontweight="bold", y=1.02)
    save_figure(fig, output_dir / "fig6_tag_tool_behaviour.png", layout_rect=(0.0, 0.0, 1.0, 0.97))


def create_figure_7(metrics: MetricsBundle, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=FIGURE_DPI)

    compare_ax = axes[0]
    compare_groups = ["Retrieved-Docs Rate", "Retrieved-Query Accuracy"]
    group_centers = [0.0, 1.2]
    bar_width = 0.22
    offsets = [-bar_width / 2, bar_width / 2]
    compare_series = [
        metrics.retrieval_rate_by_config,
        metrics.retrieved_accuracy_by_config,
    ]

    for config, offset in zip(["RAG", "TAG"], offsets):
        values = [float(series[config]) for series in compare_series]
        bars = compare_ax.bar(
            [center + offset for center in group_centers],
            values,
            width=bar_width * 0.95,
            color=CONFIG_COLORS[config],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        add_vertical_labels(compare_ax, bars, values, y_offset=0.02, fmt="{:.2f}")

    compare_ax.set_title("Retrieved Documents and Accuracy", pad=10, fontsize=12, fontweight="bold")
    compare_ax.set_ylabel("Share / Accuracy")
    compare_ax.set_xticks(group_centers)
    compare_ax.set_xticklabels(compare_groups)
    compare_max = max(float(series.max()) for series in compare_series)
    compare_ax.set_ylim(0, max(0.82, compare_max + 0.08))
    compare_ax.yaxis.set_major_locator(MultipleLocator(0.1))
    style_axes(compare_ax, grid_axis="y")
    compare_ax.legend(
        handles=legend_handles(["RAG", "TAG"]),
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
    )

    refine_ax = axes[1]
    refinement_labels = metrics.refinement_accuracy.index.tolist()
    refinement_values = metrics.refinement_accuracy.astype(float).tolist()
    refinement_colors = ["#C9C9C9", CONFIG_COLORS["TAG"]]
    refine_x = list(range(len(refinement_labels)))
    refine_bars = refine_ax.bar(
        refine_x,
        refinement_values,
        color=refinement_colors,
        width=0.58,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    refine_ax.set_title("Effect of Query Refinement", pad=10, fontsize=12, fontweight="bold")
    refine_ax.set_ylabel("Mean Accuracy")
    refine_ax.set_xticks(refine_x)
    refine_ax.set_xticklabels(["Unrefined TAG", "Refined TAG"])
    refine_ax.set_ylim(0, 0.62)
    refine_ax.yaxis.set_major_locator(MultipleLocator(0.1))
    style_axes(refine_ax, grid_axis="y")
    add_vertical_labels(refine_ax, refine_bars, refinement_values, y_offset=0.015, fmt="{:.2f}")

    # fig.suptitle("Retrieval Quality and Query Refinement", fontsize=14, fontweight="bold", y=1.02)
    save_figure(fig, output_dir / "fig7_retrieval_quality.png", layout_rect=(0.0, 0.03, 1.0, 0.97))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    workbook_path = script_dir / WORKBOOK_NAME
    if not workbook_path.exists():
        raise FileNotFoundError(f"Could not find workbook at {workbook_path}")

    configure_style()
    records = load_query_level_records(workbook_path)
    metrics = compute_metrics(records)

    create_figure_1(metrics, script_dir)
    create_figure_2(metrics, script_dir)
    create_figure_3(metrics, script_dir)
    create_figure_4(metrics, script_dir)
    create_figure_5(metrics, script_dir)
    create_figure_6(metrics, script_dir)
    create_figure_7(metrics, script_dir)


if __name__ == "__main__":
    main()
