import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Example performance data for all strategies across multiple metrics
# data = {
#     'Training method': ['FR', 'MS', 'EN', 'CS', 'QBC'],
#     'Online Accuracy': [0.747, 0.757, 0.749, 0.749, 0.762],
#     'Final Accuracy': [0.705, 0.707, 0.710, 0.710, 0.714],
#     'FLOPs': [5.46, 9.29, 10.52, 8.78, 9.52],
#     'Training Time': [2112, 3661, 4085, 3575, 3806]
# }
data = {
    'Training method': ['Complete retraining', 'Incremental', 'EWC', 'Replay (100%)', 'Replay (50%)'],
    # 'Online Accuracy': [0.758, 0.758, 0.759, 0.758, 0.765],
    'Final Accuracy': [0.706, 0.698, 0.697, 0.704, 0.708],
    'Training Time': [3661, 895, 968, 2602, 2222],
    'FLOPs': [9.28, 2.17, 2.29, 6.64, 5.86]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Normalize the data for better visualization
columns_to_normalize = ['Final Accuracy', 'Training Time', 'FLOPs']
df_normalized = df.copy()
for column in columns_to_normalize:
    df_normalized[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# Create the parallel coordinates plot
plt.figure(figsize=(10, 6))
parallel_coordinates(df_normalized, class_column='Training method', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
# plt.title('Comparison of Training methods')
plt.xlabel('Metrics')
plt.ylabel('Normalized Values')
plt.grid(True)
plt.savefig('results/plots/parallel_methods_MS.png', bbox_inches='tight')
