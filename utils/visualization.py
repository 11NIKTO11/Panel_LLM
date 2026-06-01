import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from utils.constants import VOTED, NOT_VOTED

def visualize_comprehensive_results(
    simulated_series: pd.Series,
    actual_series: pd.Series,
    parties: list[str],
    model_name: str = None,
    actual_is_claimed: bool = False
):
    """
    Generates a two-part plot using pure Matplotlib:
    1. A bar chart for election attendance (VOTED vs NOT_VOTED) comparing predicted vs reference.
    2. A grouped bar chart comparing predicted party probabilities with reference results.

    Args:
        simulated_series: Aggregated predicted series containing [VOTED, NOT_VOTED] and party indices.
        actual_series: Reference aggregated series with the same schema.
        model_name: Optional model name to include in the figure title.
        actual_is_claimed: If True, label the reference series as "Claimed"; otherwise label as "Actual".

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Labels based on reference type
    ref_label = 'Claimed' if actual_is_claimed else 'Actual'
    party_ref_legend = f"{ref_label} Result"
    attendance_ref_legend = ref_label

    # Create a figure with two subplots, stacked vertically
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(9, 12),
        sharex=False,
        gridspec_kw={'height_ratios': [1, 5]} # ax2 is 5x taller than ax1
    )
    title = 'Analysis of Simulated Voting Behavior vs. '
    title += f'{ref_label} Results'
    if model_name:
        title += f"\nModel: {model_name}"
    fig.suptitle(title, fontsize=20, y=0.96)

    # --- Plot 1: Election Attendance Probability ---
    # Use canonical order when present to avoid swapped bar order
    canonical_attendance = [VOTED, NOT_VOTED]
    # Determine available labels from Simulated/actual
    attendance_labels = [lbl for lbl in canonical_attendance if lbl in simulated_series.index or lbl in actual_series.index]

    simulated_attendance_values = [float(simulated_series.get(label, 0)) for label in attendance_labels]
    actual_attendance_values = [float(actual_series.get(label, 0)) for label in attendance_labels]

    y = np.arange(len(attendance_labels))
    height = 0.35

    # Plot simulated and reference attendance bars (Simulated on top)
    rects1 = ax1.barh(y + height / 2, simulated_attendance_values, height,
                      label='Simulated', color=['lightgreen', 'lightcoral'][:len(attendance_labels)])
    rects2 = ax1.barh(y - height/2, actual_attendance_values, height,
                     label=attendance_ref_legend, color=['darkgreen', 'darkred'][:len(attendance_labels)])

    ax1.set_title(f'Simulated vs. {ref_label} Election Attendance', fontsize=14)
    ax1.set_yticks(y, attendance_labels)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=12)

    # Add value labels
    ax1.bar_label(rects1, padding=3, fmt='{:.1%}', fontsize=10)
    ax1.bar_label(rects2, padding=3, fmt='{:.1%}', fontsize=10)

    # --- Plot 2: Simulated vs. Reference Party Results ---
    # Party labels: everything except attendance keys; prefer labels sorted by actual descending
    attendance_keys = {VOTED, NOT_VOTED}
    # Extract party parts of the series
    actual_parties = actual_series.drop(labels=[k for k in actual_series.index if k in attendance_keys])
    simulated_parties = simulated_series.drop(labels=[k for k in simulated_series.index if k in attendance_keys])

    # Determine party labels in descending order by actual values for stable presentation

    # Align series to these labels
    actual_probs = [float(actual_parties.get(lbl, 0)) for lbl in parties]
    simulated_probs = [float(simulated_parties.get(lbl, 0)) for lbl in parties]

    y = np.arange(len(parties))  # the label locations
    height = 0.4  # the height of the bars

    # Plot the bars (Simulated on top)
    rects1 = ax2.barh(y - height / 2, simulated_probs, height, label='Simulated', color='skyblue')
    rects2 = ax2.barh(y + height / 2, actual_probs, height, label=party_ref_legend, color='steelblue')

    # Add some text for labels, title and axes ticks
    ax2.set_title(f'Comparison of Simulated vs. {ref_label} Election Results', fontsize=16)
    ax2.set_xlabel('Probability / Vote Share', fontsize=12)
    wrapped_parties = [textwrap.fill(label, width=21) for label in parties]
    ax2.set_yticks(y, wrapped_parties)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.legend(loc='lower right', fontsize=12)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Add value labels for both sets of bars
    ax2.bar_label(rects1, padding=3, fmt='{:.1%}', fontsize=10)
    ax2.bar_label(rects2, padding=3, fmt='{:.1%}', fontsize=10)

    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
    # Do not force-show here; return the figure so caller can save/show as needed
    return fig


def visualize_party_errors(party_error: list[float]):
    """
    Plot histogram of absolute party probability sum errors and return the figure.

    Args:
        party_error: List of absolute errors |1 - sum(probabilities)| per respondent with an issue.

    Returns:
        matplotlib.figure.Figure | None: The created figure, or None if there is nothing to plot or an error occurs.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        if len(party_error) > 0:
            fig = plt.figure(figsize=(8, 4))
            plt.hist(party_error, bins=min(50, max(10, int(np.sqrt(len(party_error))))), color='#1f77b4',
                     edgecolor='white')
            plt.title('Distribution of party probability sum errors |1 - sum(probabilities)|')
            plt.xlabel('Absolute error from 1')
            plt.ylabel('Count of respondents')
            plt.grid(axis='y', alpha=0.2)
            plt.show()
            return fig
        else:
            print('No party probability sum errors to plot (all sums within tolerance).')
            return None
    except Exception as e:
        print(f'Could not generate party error plot: {e}')
        return None


def visualize_region_errors(
    mae_df: pd.DataFrame,
    metric: str = "MAE",
    region_column: str = "region",
    ncols: int = 4,
    vmin: float = 0,
    vmax: float = 5,
    figsize_per_plot: tuple = (4, 3.5),
    title: str = None
)-> plt.figure:
    """
    Create 3x3 heatmap subplots for each region showing pairwise error comparisons.

    Parameters:
    -----------
    mae_df : pd.DataFrame
        DataFrame with columns:
        - region_column: Region identifier
        - {metric}_predicted_vs_claimed: Error between predicted and claimed
        - {metric}_predicted_vs_actual: Error between predicted and actual
        - {metric}_claimed_vs_actual: Error between claimed and actual
    metric : str
        Metric name (e.g., "MAE", "RMSE") - used to find column names
    region_column : str
        Name of the column containing region identifiers
    ncols : int
        Number of plots per row in the figure
    vmin : float
        Minimum value for colormap scale
    vmax : float
        Maximum value for colormap scale
    figsize_per_plot : tuple
        Size (width, height) for each individual subplot
    title : str
        Overall figure title (optional)

    Returns:
    --------
    matplotlib.figure.Figure: The created figure
    """
    # Prepare column names based on metric
    col_pred_clmd = "Predicted vs Actual"
    col_pred_actu = "Predicted vs Claimed"
    col_clmd_actu = "Actual vs Claimed"

    # Check required columns exist
    required_cols = [region_column, col_pred_clmd, col_pred_actu, col_clmd_actu]
    missing = [col for col in required_cols if col not in mae_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Calculate subplot layout
    n_regions = len(mae_df)
    nrows = int(np.ceil(n_regions / ncols))

    # Create figure
    fig_width = figsize_per_plot[0] * ncols
    fig_height = figsize_per_plot[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    # Flatten axes array for easier iteration
    if n_regions == 1:
        axes = np.array([axes])
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    # Style settings
    plt.style.use("default")
    sns.set_theme(style="white")
    cmap = plt.cm.plasma

    # Labels for the 3x3 matrix
    labels = ["Predicted", "Claimed", "Actual"]

    # Plot each region
    for idx, (_, row) in enumerate(mae_df.iterrows()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        region_name = row[region_column]

        # Build 3x3 error matrix
        # Rows: Predicted, Claimed, Actual
        # Cols: Predicted, Claimed, Actual
        errors = np.array([
            [0.0,                    row[col_pred_clmd],  row[col_pred_actu]],  # Predicted vs ...
            [row[col_pred_clmd],     0.0,                 row[col_clmd_actu]],  # Claimed vs ...
            [row[col_pred_actu],     row[col_clmd_actu],  0.0]                  # Actual vs ...
        ])

        # Convert to percentage if values are in 0-1 range
        if errors.max() <= 1.0:
            errors = errors * 100

        # Create heatmap
        sns.heatmap(
            errors,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            cbar=True,
            vmin=vmin,
            vmax=vmax,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": f"{metric} (%)"},
            #annot_kws = {"size": 20}
        )

        ax.set_title(region_name, fontsize=10, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_regions, len(axes)):
        axes[idx].axis('off')

    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f"{metric} Error Comparison by Region", fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def visualize_party_probability_distribution(
    party_results_df: pd.DataFrame,
    parties: list[str],
    title: str | None = None,
    kind: str = "box",           # "box" or "violin"
    add_points: bool = True,
    sample: int | None = None,
    point_size: float = 2.0,
    alpha: float = 0.3,
) -> plt.Figure:
    """
    Visualize distribution of party probabilities with one column per party.

    Parameters
    ----------
    party_results_df : pd.DataFrame
        DataFrame containing probability columns for each party (wide format).
    parties : list[str]
        Ordered list of party column names to include.
    title : str | None
        Optional plot title.
    kind : str
        "box" for boxplot or "violin" for violin plot.
    add_points : bool
        Whether to overlay individual observations as a jittered stripplot.
    sample : int | None
        If set, randomly sample this many rows to reduce overplotting.
    point_size : float
        Size of points in the stripplot overlay.
    alpha : float
        Alpha of points in the stripplot overlay.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    # Select and clean data
    df = party_results_df[parties].copy()
    # Clip to [0, 1] just in case and optionally sample to avoid heavy overplotting
    df = df.clip(lower=0, upper=1)
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=42)

    # Melt to long format: columns -> 'party', values -> 'value'
    long_df = df.reset_index(drop=True).melt(var_name="party", value_name="value")

    # Figure sizing proportional to number of parties
    fig_width = max(10, int(len(parties) * 0.9))
    fig, ax = plt.subplots(figsize=(fig_width, 10))

    sns.set_theme(style="whitegrid")

    if kind.lower() == "violin":
        sns.violinplot(data=long_df, x="party", y="value", ax=ax, inner=None, cut=0, linewidth=1, color="#89CFF0")
    else:
        # default to box
        sns.boxplot(data=long_df, x="party", y="value", ax=ax, showfliers=False, color="#89CFF0")

    if add_points:
        sns.stripplot(
            data=long_df, x="party", y="value", ax=ax,
            color="black", size=point_size, alpha=alpha, jitter=0.2
        )

    ax.set_ylabel("Probability")
    ax.set_xlabel("Party")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=90)

    if title is None:
        title = "Distribution of Party Probabilities (per Respondent)"
    ax.set_title(title)

    plt.tight_layout()
    return fig
