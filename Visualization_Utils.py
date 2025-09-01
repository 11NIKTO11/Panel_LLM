import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import VotingResult
from typing import List


def aggregate_results(results: List['VotingResult'], weighted: bool = True) -> [dict, dict]:
    """
    Aggregates probabilities for parties and election attendance.
    Returns AggregatedResults containing dictionaries instead of DataFrame and Series.
    """
    # Aggregate party probabilities
    party_probs = {}
    for result in results:
        weight = result.voted_or_not.voted if weighted else 1
        for party in result.parties:
            if party.name not in party_probs:
                party_probs[party.name] = []
            party_probs[party.name].append(party.probability * weight)

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
    actual_attendance_dict: dict
):
    """
    Generates a two-part plot using pure Matplotlib:
    1. A bar chart for predicted election attendance.
    2. A grouped bar chart comparing predicted party probabilities with actual results.
    """
    # Create a figure with two subplots, stacked vertically
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 14),
        sharex=False,
        gridspec_kw={'height_ratios': [1, 5]} # ax2 is 5x taller than ax1
    )
    fig.suptitle('Analysis of Predicted Voting Behavior vs. 2021 Actual Results', fontsize=20, y=0.96)

    # --- Plot 1: Election Attendance Probability ---
    attendance_labels = list(predicted_attendance_dict.keys())
    predicted_attendance_values = list(predicted_attendance_dict.values())
    actual_attendance_values = [actual_attendance_dict.get(label, 0) for label in attendance_labels]
    
    y = np.arange(len(attendance_labels))
    height = 0.35
    
    # Plot predicted and actual attendance bars
    rects1 = ax1.barh(y + height/2, predicted_attendance_values, height, 
                     label='Predicted', color=['lightgreen', 'lightcoral'])
    rects2 = ax1.barh(y - height/2, actual_attendance_values, height, 
                     label='Actual', color=['darkgreen', 'darkred'])
    
    ax1.set_title('Predicted vs. Actual Election Attendance', fontsize=14)
    ax1.set_yticks(y, attendance_labels)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=12)
    
    # Add value labels
    ax1.bar_label(rects1, padding=3, fmt='{:.1%}', fontsize=10)
    ax1.bar_label(rects2, padding=3, fmt='{:.1%}', fontsize=10)

    # --- Plot 2: Predicted vs. Actual Party Results ---
    # Prepare data for grouped bar chart
    labels = list(predicted_party_dict.keys())
    predicted_probs = list(predicted_party_dict.values())

    # Map actual results to the same order as predicted
    actual_probs = [actual_party_dict.get(label, 0) for label in labels]

    y = np.arange(len(labels))  # the label locations
    height = 0.4  # the height of the bars

    # Plot the bars
    rects1 = ax2.barh(y + height / 2, predicted_probs, height, label='Predicted', color='skyblue')
    rects2 = ax2.barh(y - height / 2, actual_probs, height, label='Actual 2021 Result', color='steelblue')

    # Add some text for labels, title and axes ticks
    ax2.set_title('Comparison of Predicted vs. Actual Election Results', fontsize=16)
    ax2.set_xlabel('Probability / Vote Share', fontsize=12)
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.legend(fontsize=12)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Add value labels for both sets of bars
    ax2.bar_label(rects1, padding=3, fmt='{:.1%}', fontsize=10)
    ax2.bar_label(rects2, padding=3, fmt='{:.1%}', fontsize=10)

    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()