import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np

def visualize_comprehensive_results(
    predicted_df: pd.DataFrame,
    attendance_s: pd.Series,
    actual_results: dict
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
    ax1.barh(attendance_s.index, attendance_s.values, color=['#4CAF50', '#F44336'], height=0.6)
    ax1.set_title('Predicted Election Attendance', fontsize=14)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_xlim(0, 1)
    # Add value labels
    for index, value in enumerate(attendance_s.values):
        ax1.text(value + 0.02, index, f'{value:.1%}', va='center', fontsize=12)

    # --- Plot 2: Predicted vs. Actual Party Results ---
    # Prepare data for grouped bar chart
    labels = predicted_df.index
    predicted_probs = predicted_df['Predicted Probability']

    # Map actual results to the same order as predicted, converting to 0-1 scale
    actual_probs_percent = [actual_results.get(label, 0) for label in labels]
    actual_probs = [p / 100.0 for p in actual_probs_percent]

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