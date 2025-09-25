import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import VotingResult
from typing import List
import seaborn as sns


def aggregate_results(results: List['VotingResult'], weighted: bool = True, normalized: bool = True) -> [dict, dict]:
    """
    Aggregates probabilities for parties and election attendance.

    Args:
        results: Iterable of VotingResult instances to aggregate.
        weighted: If True, each respondent's party probabilities are weighted by
            their probability of having voted (result.voted_or_not.voted). If False,
            all respondents contribute equally (weight = 1) regardless of their
            voting probability.
        normalized: If True, each respondent's party probabilities are first
            normalized to sum to 1 before aggregation (dividing by the sum of that
            respondent's party probabilities). This mitigates bias if a respondent's
            party probabilities do not sum to 1. If False, raw probabilities are
            used without per-respondent normalization.

    Returns:
        Tuple of two dictionaries:
          - party_dict: {party_name -> aggregated probability} normalized to sum to 1.
          - attendance_dict: {"Voted" -> avg voted prob, "Not Voted" -> avg not_voted prob}.
    """
    # Aggregate party probabilities
    party_probs = {}
    for result in results:
        weight = result.voted_or_not.voted if weighted else 1
        prob_sum = sum([p.probability for p in result.parties]) if normalized else 1
        for party in result.parties:
            if party.name not in party_probs:
                party_probs[party.name] = []
            party_probs[party.name].append(
                party.probability * weight / (prob_sum or 1)
            )

    # Calculate average probabilities and normalize
    party_averages = {k: sum(v) / len(v) for k, v in party_probs.items()}
    total_prob = sum(party_averages.values())
    if total_prob > 0:
        party_dict = {k: v / total_prob for k, v in party_averages.items()}
    else:
        party_dict = party_averages

    # Sort by probability (highest first)
    party_dict = dict(sorted(party_dict.items(), key=lambda x: x[1], reverse=True))

    # Aggregate voted/not_voted probabilities
    voted_sum = sum(r.voted_or_not.voted for r in results)
    not_voted_sum = sum(r.voted_or_not.not_voted for r in results)
    total_respondents = len(results)

    attendance_dict = {
        "Voted": voted_sum / total_respondents,
        "Not Voted": not_voted_sum / total_respondents
    }

    return party_dict, attendance_dict


def visualize_comprehensive_results(
    predicted_party_dict: dict,
    predicted_attendance_dict: dict,
    actual_party_dict: dict,
    actual_attendance_dict: dict,
    model_name: str = None,
    actual_is_claimed: bool = False
):
    """
    Generates a two-part plot using pure Matplotlib:
    1. A bar chart for predicted election attendance.
    2. A grouped bar chart comparing predicted party probabilities with actual/claimed results.

    Args:
        predicted_party_dict: Aggregated predicted party probabilities (sum to 1).
        predicted_attendance_dict: Predicted attendance distribution, e.g., {"Voted": p, "Not Voted": q}.
        actual_party_dict: Reference party distribution (official actual or respondents' claimed).
        actual_attendance_dict: Reference attendance distribution (official actual or respondents' claimed).
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
        title += f" — Model: {model_name}"
    fig.suptitle(title, fontsize=20, y=0.96)

    # --- Plot 1: Election Attendance Probability ---
    # Use canonical order when present to avoid swapped bar order
    canonical_attendance = ["Voted", "Not Voted"]
    # Keep only those present, then append any extras (stable)
    attendance_labels = [lbl for lbl in canonical_attendance if lbl in predicted_attendance_dict]
    extras = [k for k in predicted_attendance_dict.keys() if k not in attendance_labels]
    attendance_labels += extras

    predicted_attendance_values = [predicted_attendance_dict.get(label, 0) for label in attendance_labels]
    actual_attendance_values = [actual_attendance_dict.get(label, 0) for label in attendance_labels]

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
    # Prepare data for grouped bar chart using predicted order to align bars
    labels = list(actual_party_dict.keys())
    actual_probs = list(actual_party_dict.values())

    # Map reference results to the same order as predicted
    predicted_probs = [predicted_party_dict.get(label, 0) for label in labels]

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
    plt.figure(figsize=(6, 5))
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
