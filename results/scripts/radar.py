import matplotlib.pyplot as plt
import numpy as np

# Function to create a radar chart
def create_radar_chart(title, labels, data, strategies, colors):
    # Number of variables we're plotting.
    num_vars = len(labels)

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is made in a circular (not polygon) space
    angles += angles[:1]

    # Draw the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for strategy, color in zip(strategies, colors):
        values = data[strategy]
        values += values[:1]  # Complete the loop
        ax.fill(angles, values, color, alpha=0.25)
        ax.plot(angles, values, color=color, linewidth=2, label=strategy)

    # Fix the labels to be at the correct angles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    # Set the range of the radial axis
    ax.set_ylim(0, 1)  # Assuming values are between 0 and 1

    # Add a title and display the chart
    # ax.set_title(title, size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=10)
    plt.savefig(f'results/plots/{title}.png', bbox_inches='tight')

# Example data
datasets = ['ISEAR', 'Openbook', 'RT-Polarity', 'FEVER']
metrics = ['Online Accuracy', 'Final Accuracy', 'FLOPS', 'Training Time']
strategies = ['Front-loading', 'Margin Sampling', 'Entropy', 'Coreset', 'Query by Committee']
colors = ['b', 'g', 'r', 'c', 'm']


# Example performance values for all strategies across different datasets and metrics
performance_data = {
    'Online Accuracy': {
        'Front-loading': [0.646, 0.725, 0.880, 0.736],
        'Margin Sampling': [0.665, 0.727, 0.890, 0.747],
        'Entropy': [0.657, 0.718, 0.887, 0.734],
        'Coreset': [0.649, 0.734, 0.888, 0.724],
        'Query by Committee': [0.657, 0.763, 0.879, 0.748]
    },
    'Final Accuracy': {
        'Front-loading': [0.612, 0.640, 0.884, 0.683],
        'Margin Sampling': [0.619, 0.637, 0.887, 0.683],
        'Entropy': [0.620, 0.656, 0.888, 0.676],
        'Coreset': [0.615, 0.657, 0.886, 0.683],
        'Query by Committee': [0.622, 0.672, 0.882, 0.678]
    },
    'FLOPS': {
        'Front-loading': [6.69, 5.95, 4.29, 4.89],
        'Margin Sampling': [9.14, 8.91, 10.35, 8.75],
        'Entropy': [11.52, 14.63, 7.32, 8.59],
        'Coreset': [6.89, 11.81, 8.64, 7.77],
        'Query by Committee': [10.65, 15.38, 3.63, 8.42]
    },
    'Training Time': {
        'Front-loading': [2427, 1808, 1702, 2509],
        'Margin Sampling': [3447, 2682, 4096, 4419],
        'Entropy': [4231, 4541, 2812, 4754],
        'Coreset': [2550, 3903, 3524, 4321],
        'Query by Committee': [3841, 5080, 1579, 4725]
    }
}

# Normalize the performance data for FLOPS and Training Time to fit on the radar chart
for metric in ['FLOPS', 'Training Time']:
    max_value = max(max(performance_data[metric][strategy]) for strategy in strategies)
    for strategy in strategies:
        performance_data[metric][strategy] = [value / max_value for value in performance_data[metric][strategy]]

# Generate radar charts for each metric
for metric in metrics:
    create_radar_chart(f'{metric} radar', datasets, performance_data[metric], strategies, colors)
