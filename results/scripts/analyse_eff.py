import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the data
path='results/openbook/b1'
data = pd.read_csv(path+'.csv')  # Replace with your file path
budgets = [1000, 1500, 2000, 2500, 3000, 3500]

# Define conditions
conditions = {
    'Complete re-training': {'args/buffer_percent': 1.0, 'args/ewc': 'no', 'args/incremental': 'no'},
    'Incremental': {'args/buffer_percent': 0.0, 'args/ewc': 'no', 'args/incremental': 'yes'},
    'EWC': {'args/buffer_percent': 0.0, 'args/ewc': 'yes', 'args/incremental': 'yes'},
    'Replay': {'args/buffer_percent': 1.0, 'args/ewc': 'no', 'args/incremental': 'yes'},
    'Replay (50%)': {'args/buffer_percent': 0.5, 'args/ewc': 'no', 'args/incremental': 'yes'}
}

# conditions_strategies = {
#     'FR': {'args/strategy': 'b1', 'args/buffer_percent': 0.5},
#     'MS': {'args/strategy': 'BT', 'args/buffer_percent': 0.5},
#     'EN': {'args/strategy': 'EN', 'args/buffer_percent': 0.5},
#     'CS': {'args/strategy': 'CS', 'args/buffer_percent': 0.5},
#     'QBC': {'args/strategy': 'MV', 'args/buffer_percent': 0.5}
# }

# conditions=conditions_strategies

# Function to extract budget values from column names
def extract_budgets(data):
    return [f"online/{budget}-gold_0 (average)" for budget in budgets]

# Function to aggregate accuracy and std across seeds for a given condition and budget
def aggregate_accuracy(data, condition, budget_columns):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1)]
    means = [filtered_data[col].mean() for col in budget_columns]
    stds = [filtered_data[col].std() for col in budget_columns]
    return means, stds

# Function to aggregate test accuracy and std
def aggregate_accuracy_test(data, condition):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1)]
    means = [filtered_data[filtered_data['args/budget'] == budget]['test/test_gold_acc (last)'].mean() for budget in budgets]
    stds = [filtered_data[filtered_data['args/budget'] == budget]['test/test_gold_acc (last)'].std() for budget in budgets]
    return means, stds

# Function to aggregate efficiency metrics and std
def aggregate_eff(data, condition, flops_col='train/total_flops (last)', time_col='train/total_time_elapsed (last)'):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1)]
    flops_means = [filtered_data[filtered_data['args/budget'] == budget][flops_col].mean() for budget in budgets]
    flops_stds = [filtered_data[filtered_data['args/budget'] == budget][flops_col].std() for budget in budgets]
    time_means = [filtered_data[filtered_data['args/budget'] == budget][time_col].mean() for budget in budgets]
    time_stds = [filtered_data[filtered_data['args/budget'] == budget][time_col].std() for budget in budgets]
    return (flops_means, flops_stds), (time_means, time_stds)

# Function to calculate AUC using the trapezoidal rule
def calculate_auc(num, den=budgets):
    auc = np.trapz(num, den)
    normalized_auc = auc / (den[-1] - den[0])
    return round(normalized_auc, 3)

# Extract budget columns
budget_columns = extract_budgets(data)

# Aggregate data, generate tables, and calculate AUC
auc_values_online = {}
auc_values_test = {}
auc_values_flops = {}
auc_values_time = {}
aggregated_data = {}

for condition_name, condition in conditions.items():
    aggregated_accuracies, accuracies_std = aggregate_accuracy(data, condition, budget_columns)
    aggregated_accuracies_test, accuracies_test_std = aggregate_accuracy_test(data, condition)
    (total_flops, flops_std), (total_time, time_std) = aggregate_eff(data, condition)
    
    aggregated_data[condition_name] = pd.DataFrame({
        'budget': budgets,
        'online': aggregated_accuracies,
        'online_std': accuracies_std,
        'test': aggregated_accuracies_test,
        'test_std': accuracies_test_std,
        'total_flops': total_flops,
        'flops_std': flops_std,
        'total_time': total_time,
        'time_std': time_std
    })
    
    auc_values_online[condition_name] = calculate_auc(aggregated_accuracies)
    auc_values_test[condition_name] = calculate_auc(aggregated_accuracies_test)
    auc_values_flops[condition_name] = round(calculate_auc(total_flops) / 1e15, 2)
    auc_values_time[condition_name] = round(calculate_auc(total_time), 0)

print('Online AUC')
print(auc_values_online)
print('Test AUC')
print(auc_values_test)
print('Flops AUC')
print(auc_values_flops)
print('Time AUC')
print(auc_values_time)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

colors = plt.cm.rainbow(np.linspace(0, 1, len(conditions)))

# Online Accuracy
for i, (condition_name, color) in enumerate(zip(conditions, colors)):
    df = aggregated_data[condition_name]
    axs[0, 0].errorbar(budgets, df['online'], yerr=df['online_std'], fmt='-o', capsize=0, label=f"{condition_name}", color=color)

axs[0, 0].set_title("Online Accuracy Comparison", fontsize=16)
axs[0, 0].set_xlabel('Budget', fontsize=14)
axs[0, 0].set_ylabel('Online Accuracy', fontsize=14)
axs[0, 0].legend(fontsize=12)
axs[0, 0].tick_params(axis='both', which='major', labelsize=12)

# Test Accuracy
for i, (condition_name, color) in enumerate(zip(conditions, colors)):
    df = aggregated_data[condition_name]
    axs[0, 1].errorbar(budgets, df['test'], yerr=df['test_std'], fmt='-o', capsize=0, label=f"{condition_name}", color=color)

axs[0, 1].set_title("Test Accuracy Comparison", fontsize=16)
axs[0, 1].set_xlabel('Budget', fontsize=14)
axs[0, 1].set_ylabel('Test Accuracy', fontsize=14)
axs[0, 1].legend(fontsize=12)
axs[0, 1].tick_params(axis='both', which='major', labelsize=12)

# FLOPS
for i, (condition_name, color) in enumerate(zip(conditions, colors)):
    df = aggregated_data[condition_name]
    axs[1, 0].errorbar(budgets, df['total_flops'], yerr=df['flops_std'], fmt='-o', capsize=0, label=f"{condition_name}", color=color)

axs[1, 0].set_title("FLOPS Comparison", fontsize=16)
axs[1, 0].set_xlabel('Budget', fontsize=14)
axs[1, 0].set_ylabel('Flops', fontsize=14)
axs[1, 0].legend(fontsize=12)
axs[1, 0].tick_params(axis='both', which='major', labelsize=12)

# Training Time
for i, (condition_name, color) in enumerate(zip(conditions, colors)):
    df = aggregated_data[condition_name]
    axs[1, 1].errorbar(budgets, df['total_time'], yerr=df['time_std'], fmt='-o', capsize=0, label=f"{condition_name}", color=color)

axs[1, 1].set_title("Training Time Comparison", fontsize=16)
axs[1, 1].set_xlabel('Budget', fontsize=14)
axs[1, 1].set_ylabel('Training Time', fontsize=14)
axs[1, 1].legend(fontsize=12)
axs[1, 1].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
# Save the figure to a file
plt.savefig(path+'.png')
plt.savefig(path+'.pdf')
plt.close()