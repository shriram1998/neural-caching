import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Example performance data for all strategies across multiple metrics
data = {
    'Strategy': ['FR', 'MS', 'EN', 'CS', 'QBC'],
    'Online Accuracy': [0.747, 0.757, 0.749, 0.749, 0.762],
    'Final Accuracy': [0.705, 0.707, 0.710, 0.710, 0.714],
    'FLOPS': [5.46, 9.29, 10.52, 8.78, 9.52],
    'Training Time': [2112, 3661, 4085, 3575, 3806]
}

# Normalize FLOPS and Training Time for better comparison
max_flops = max(data['FLOPS'])
max_time = max(data['Training Time'])

data['FLOPS'] = [value / max_flops for value in data['FLOPS']]
data['Training Time'] = [value / max_time for value in data['Training Time']]

# Create a DataFrame
df = pd.DataFrame(data)

# Create the parallel coordinates plot
plt.figure(figsize=(10, 6))
parallel_coordinates(df, class_column='Strategy', colormap=plt.get_cmap("Set1"))
plt.title('Parallel Coordinates Plot for Strategy Performance')
plt.xlabel('Metrics')
plt.ylabel('Normalized Values')
plt.grid(True)
plt.savefig('results/plots/parallel_coordinates.png', bbox_inches='tight')
