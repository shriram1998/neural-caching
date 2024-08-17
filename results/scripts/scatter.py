import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
final_accuracy = np.array([
    [0.612, 0.640, 0.884, 0.683],
    [0.619, 0.637, 0.887, 0.683],
    [0.620, 0.656, 0.888, 0.676],
    [0.615, 0.657, 0.886, 0.683],
    [0.622, 0.672, 0.882, 0.678]
])

flops = np.array([
    [6.69, 5.95, 4.29, 4.89],
    [9.14, 8.91, 10.35, 8.75],
    [11.52, 14.63, 7.32, 8.59],
    [6.89, 11.81, 8.64, 7.77],
    [10.65, 15.38, 3.63, 8.42]
])

training_time = np.array([
    [2427, 1808, 1702, 2509],
    [3447, 2682, 4096, 4419],
    [4231, 4541, 2812, 4754],
    [2550, 3903, 3524, 4321],
    [3841, 5080, 1579, 4725]
])

# Flatten the arrays and create a DataFrame
data = pd.DataFrame({
    'Final Accuracy': final_accuracy.flatten(),
    'FLOPS (E+15)': flops.flatten(),
    'Training Time (s)': training_time.flatten()
})

# Create a color palette for different datasets
datasets = ['ISEAR', 'Openbook', 'RT-Polarity', 'FEVER']
color_palette = sns.color_palette("husl", n_colors=len(datasets))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot: Final Accuracy vs FLOPS
sns.scatterplot(data=data, x='FLOPS (E+15)', y='Final Accuracy', ax=ax1, hue=np.tile(datasets, 5), palette=color_palette)
ax1.set_title('Final Accuracy vs FLOPS')
ax1.set_xlabel('FLOPS (E+15)')
ax1.set_ylabel('Final Accuracy')

# Scatter plot: Final Accuracy vs Training Time
sns.scatterplot(data=data, x='Training Time (s)', y='Final Accuracy', ax=ax2, hue=np.tile(datasets, 5), palette=color_palette)
ax2.set_title('Final Accuracy vs Training Time')
ax2.set_xlabel('Training Time (s)')
ax2.set_ylabel('Final Accuracy')

# Adjust layout and show plot
plt.tight_layout()
plt.savefig('scatter_plots.png', bbox_inches='tight')

# Optionally, save the plot
# plt.savefig('accuracy_vs_cost_scatter_plots.png', dpi=300, bbox_inches='tight')

# Calculate correlations
correlation_flops = data['Final Accuracy'].corr(data['FLOPS (E+15)'])
correlation_time = data['Final Accuracy'].corr(data['Training Time (s)'])

print(f"Correlation between Final Accuracy and FLOPS: {correlation_flops:.3f}")
print(f"Correlation between Final Accuracy and Training Time: {correlation_time:.3f}")