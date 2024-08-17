import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data for the heatmap
data = {
    'FR': [0.705, 0.693, 0.698, 0.700, 0.704],
    'MS': [0.707, 0.698, 0.697, 0.704, 0.708],
    'EN': [0.710, 0.699, 0.699, 0.703, 0.708],
    'CS': [0.710, 0.699, 0.698, 0.707, 0.704],
    'QBC': [0.714, 0.702, 0.702, 0.708, 0.710]
}

# Index labels for rows
index = ['Complete re-training', 'Incremental', 'EWC', 'Replay', 'Replay (50%)']

# Create a DataFrame
df = pd.DataFrame(data, index=index)

# Generate the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="YlGnBu", linewidths=.5)

# Add titles and labels
plt.title('Comparison of Training Methods vs Strategies (Final Accuracy)')
plt.xlabel('Strategy')
plt.ylabel('Training Method')

# Display the heatmap
plt.tight_layout()
plt.savefig('heatmap_training_methods_strategies.png', bbox_inches='tight')
