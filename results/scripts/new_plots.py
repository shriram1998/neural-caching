# Creating a heatmap to compare training methods with strategies

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for the heatmap
training_methods = ['Complete re-training', 'Incremental', 'Incremental with EWC', 'Replay', 'Replay (50%)']
strategies = ['FR', 'MS', 'EN', 'CS', 'QBC']

# Example data matrix for the heatmap
data_matrix = np.array([
    [0.747, 0.757, 0.749, 0.749, 0.762],  # Complete re-training
    [0.739, 0.757, 0.746, 0.745, 0.759],  # Incremental
    [0.741, 0.759, 0.746, 0.746, 0.758],  # Incremental with EWC
    [0.743, 0.758, 0.748, 0.748, 0.759],  # Replay
    [0.746, 0.765, 0.754, 0.746, 0.762]   # Replay (50%)
])

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data_matrix, annot=True, cmap="YlGnBu", xticklabels=strategies, yticklabels=training_methods)
plt.title('Comparison of Training Methods with Strategies (Accuracy)')
plt.xlabel('Strategies')
plt.ylabel('Training Methods')
plt.tight_layout()
plt.savefig('results/plots/heatmap_training_methods_strategies.png', bbox_inches='tight')
