import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import VotingResult
from typing import List
import seaborn as sns


from Data import (
    VOTED, NOT_VOTED,
    ANO, SPOLU, PIRSTAN, KSCM, SPD, CSSD,
    TSS, PRISAHA, JINA_STRANA
)
import pandas as pd

def aggregate_results(df: pd.DataFrame, weighted: bool = True, normalized: bool = True) -> pd.Series:
    """
    Aggregates probabilities for parties and election attendance from a pandas DataFrame and
    returns a single pandas Series combining attendance and normalized party distribution.

    Args:
        df: pandas DataFrame with columns VOTED, NOT_VOTED, and party columns named via Data.py constants.
        weighted: Uses the VOTED column as row weight when True; otherwise weight = 1.
        normalized: Normalize per-row party probabilities to sum to 1 before aggregation.

    Returns:
        pd.Series: index contains [VOTED, NOT_VOTED] plus party columns; values are aggregated.
    """
    PARTY_COLUMNS = [
        ANO, SPOLU, PIRSTAN, KSCM, SPD, CSSD,
        TSS, PRISAHA, JINA_STRANA
    ]

    # Ensure required columns exist (soft check)
    missing = [c for c in [VOTED, NOT_VOTED] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in DataFrame: {missing}")

    # Select party columns that are present in the DataFrame
    parties_present = [c for c in PARTY_COLUMNS if c in df.columns]
    if not parties_present:
        # If none of predefined party columns are present, try all columns excluding known fields
        parties_present = [c for c in df.columns if c not in ['region', VOTED, NOT_VOTED]]

    # Row-wise normalization factor
    if normalized:
        denom = df[parties_present].sum(axis=1).replace(0, np.nan)
    else:
        denom = 1.0

    # Weight per row
    if weighted:
        w = df[VOTED]
    else:
        w = 1.0

    weighted_parties = (df[parties_present].div(denom, axis=0).fillna(0)).multiply(w, axis=0)
    party_sums = weighted_parties.mean(axis=0)

    # Normalize party distribution to sum to 1 across parties
    total = party_sums.sum()
    normalized_parties = party_sums / total if total > 0 else party_sums * 0

    attendance = pd.Series({
        VOTED: float(df[VOTED].mean()),
        NOT_VOTED: float(df[NOT_VOTED].mean()),
    })

    result_series = pd.concat([attendance, normalized_parties])
    return result_series


def visualize_comprehensive_results(
    predicted_series: pd.Series,
    actual_series: pd.Series,
    model_name: str = None,
    actual_is_claimed: bool = False
):
    """
    Generates a two-part plot using pure Matplotlib:
    1. A bar chart for election attendance (VOTED vs NOT_VOTED) comparing predicted vs reference.
    2. A grouped bar chart comparing predicted party probabilities with reference results.

    Args:
        predicted_series: Aggregated predicted series containing [VOTED, NOT_VOTED] and party indices.
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
        figsize=(12, 14),
        sharex=False,
        gridspec_kw={'height_ratios': [1, 5]} # ax2 is 5x taller than ax1
    )
    title = 'Analysis of Predicted Voting Behavior vs. '
    title += f'{ref_label} Results'
    if model_name:
        title += f"\nModel: {model_name}"
    fig.suptitle(title, fontsize=20, y=0.96)

    # --- Plot 1: Election Attendance Probability ---
    # Use canonical order when present to avoid swapped bar order
    canonical_attendance = [VOTED, NOT_VOTED]
    # Determine available labels from predicted/actual
    attendance_labels = [lbl for lbl in canonical_attendance if lbl in predicted_series.index or lbl in actual_series.index]

    predicted_attendance_values = [float(predicted_series.get(label, 0)) for label in attendance_labels]
    actual_attendance_values = [float(actual_series.get(label, 0)) for label in attendance_labels]

    y = np.arange(len(attendance_labels))
    height = 0.35

    # Plot predicted and reference attendance bars (Predicted on top)
    rects1 = ax1.barh(y + height/2, predicted_attendance_values, height,
                     label='Predicted', color=['lightgreen', 'lightcoral'][:len(attendance_labels)])
    rects2 = ax1.barh(y - height/2, actual_attendance_values, height,
                     label=attendance_ref_legend, color=['darkgreen', 'darkred'][:len(attendance_labels)])

    ax1.set_title(f'Predicted vs. {ref_label} Election Attendance', fontsize=14)
    ax1.set_yticks(y, attendance_labels)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=12)

    # Add value labels
    ax1.bar_label(rects1, padding=3, fmt='{:.1%}', fontsize=10)
    ax1.bar_label(rects2, padding=3, fmt='{:.1%}', fontsize=10)

    # --- Plot 2: Predicted vs. Reference Party Results ---
    # Party labels: everything except attendance keys; prefer labels sorted by actual descending
    attendance_keys = {VOTED, NOT_VOTED}
    # Extract party parts of the series
    actual_parties = actual_series.drop(labels=[k for k in actual_series.index if k in attendance_keys])
    predicted_parties = predicted_series.drop(labels=[k for k in predicted_series.index if k in attendance_keys])

    # Determine party labels in descending order by actual values for stable presentation
    labels = list(actual_parties.sort_values(ascending=False).index)
    # Align series to these labels
    actual_probs = [float(actual_parties.get(lbl, 0)) for lbl in labels]
    predicted_probs = [float(predicted_parties.get(lbl, 0)) for lbl in labels]

    y = np.arange(len(labels))  # the label locations
    height = 0.4  # the height of the bars

    # Plot the bars (Predicted on top)
    rects1 = ax2.barh(y - height / 2, predicted_probs, height, label='Predicted', color='skyblue')
    rects2 = ax2.barh(y + height / 2, actual_probs, height, label=party_ref_legend, color='steelblue')

    # Add some text for labels, title and axes ticks
    ax2.set_title(f'Comparison of Predicted vs. {ref_label} Election Results', fontsize=16)
    ax2.set_xlabel('Probability / Vote Share', fontsize=12)
    ax2.set_yticks(y, labels)
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


def visualize_party_errors(party_error: List[float]):
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


def evaluate_polls(predicted: dict, claimed: dict, actual: dict, metric: str = "MAE", poll_name: str = "", model_name: str = "", vmax: float = 5):
    """
    Compare predicted, claimed, and actual results with pairwise errors.

    Parameters:
    -----------
    predicted : dict
        Predicted probabilities {party: probability (0-1)}
    claimed : dict
        Claimed probabilities (e.g., exit poll) {party: probability (0-1)}
    actual : dict
        Actual election results {party: probability (0-1)}
    metric : str, "MAE" or "RMSE"
        Error metric to compute

    Returns:
    --------
    None (displays 3x3 heatmap)
    """
    parties = list(actual.keys())

    datasets = {
        "Predicted": predicted,
        "Claimed": claimed,
        "Actual": actual,
    }

    labels = list(datasets.keys())
    n = len(labels)
    errors = np.zeros((n, n))

    # Compute pairwise errors
    for i, (name_i, dist_i) in enumerate(datasets.items()):
        for j, (name_j, dist_j) in enumerate(datasets.items()):
            diffs = []
            for party in parties:
                diff = dist_i.get(party, 0) - dist_j.get(party, 0)
                if metric.upper() == "MAE":
                    diffs.append(abs(diff) * 100)
                elif metric.upper() == "RMSE":
                    diffs.append((diff ** 2) ** 0.5 * 100)
                else:
                    raise ValueError("Metric must be 'MAE' or 'RMSE'")
            if metric.upper() == "MAE":
                errors[i, j] = np.mean(diffs)
            else:  # RMSE
                errors[i, j] = np.sqrt(np.mean([d ** 2 for d in diffs]))

    # Force white background and consistent plasma colormap
    plt.style.use("default")
    sns.set_theme(style="white")
    cmap = plt.cm.plasma

    # Heatmap
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(
        errors,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        cbar_kws={"label": "Error (%)"},
        vmin=0,
        vmax=vmax,
        xticklabels=labels,
        yticklabels=labels,
    )

    title = f"{poll_name} Errors ({metric.upper()})"
    if model_name:
        title += f" — Model: {model_name}"

    plt.title(title)
    plt.tight_layout()
    plt.show()

    return fig
