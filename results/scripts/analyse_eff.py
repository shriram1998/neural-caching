import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the data
path = 'results/isear/random_eviction'
data = pd.read_csv(path + '.csv')  # Replace with your file path
budgets = [1000, 1500, 2000, 2500, 3000, 3500]

# Define conditions
# conditions = {
#     'Complete re-training': {'args/buffer_percent': 1.0, 'args/ewc': 'no', 'args/incremental': 'no'},
#     'Incremental': {'args/buffer_percent': 0.0, 'args/incremental': 'yes'},
#     'EWC': {'args/buffer_percent': 0.0, 'args/ewc': 'yes', 'args/incremental': 'yes'},
#     'Replay': {'args/buffer_percent': 1.0, 'args/incremental': 'yes'},
#     'Replay (50%)': {'args/buffer_percent': 0.5, 'args/ewc': 'no', 'args/incremental': 'yes'}
# }
conditions = {
    '0.25': {'args/buffer_percent': 0.25},
    '0.5': {'args/buffer_percent': 0.5},
    '0.75': {'args/buffer_percent': 0.75},
}

# Define strategies
strategies = ['b1']

# Function to extract budget values from column names
def extract_budgets(data):
    return [f"online/{budget}-gold_0 (average)" for budget in budgets]

# Function to aggregate accuracy and std across seeds for a given condition, strategy, and budget
def aggregate_accuracy(data, condition, strategy, budget_columns):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & (data['args/strategy'] == strategy)]
    means = [filtered_data[col].mean() for col in budget_columns]
    stds = [filtered_data[col].std() for col in budget_columns]
    return means, stds

# Function to aggregate test accuracy and std
def aggregate_accuracy_test(data, condition, strategy):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & (data['args/strategy'] == strategy)]
    means = [filtered_data[filtered_data['args/budget'] == budget]['test/test_gold_acc (last)'].mean() for budget in budgets]
    stds = [filtered_data[filtered_data['args/budget'] == budget]['test/test_gold_acc (last)'].std() for budget in budgets]
    return means, stds

# Function to aggregate efficiency metrics and std
def aggregate_eff(data, condition, strategy, flops_col='train/total_flops (last)', time_col='train/total_time_elapsed (last)'):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & (data['args/strategy'] == strategy)]
    
    flops_means = []
    flops_stds = []
    time_means = []
    time_stds = []
    
    for budget in budgets:
        flops_data = filtered_data[filtered_data['args/budget'] == budget][flops_col]
        time_data = filtered_data[filtered_data['args/budget'] == budget][time_col]
        
        # Filter out rows where flops or time are null or zero
        valid_flops_data = flops_data[(flops_data.notnull()) & (flops_data != 0)]
        valid_time_data = time_data[(time_data.notnull()) & (time_data != 0)]
        
        flops_means.append(valid_flops_data.mean() if not valid_flops_data.empty else np.nan)
        flops_stds.append(valid_flops_data.std() if not valid_flops_data.empty else np.nan)
        time_means.append(valid_time_data.mean() if not valid_time_data.empty else np.nan)
        time_stds.append(valid_time_data.std() if not valid_time_data.empty else np.nan)
    
    return (flops_means, flops_stds), (time_means, time_stds)


# Function to calculate AUC using the trapezoidal rule
def calculate_auc(num, den=budgets):
    auc = np.trapz(num, den)
    normalized_auc = auc / (den[-1] - den[0])
    return round(normalized_auc, 3)

# Extract budget columns
budget_columns = extract_budgets(data)

# Aggregate data, generate tables, and calculate AUC
auc_values_online = defaultdict(dict)
auc_values_test = defaultdict(dict)
auc_values_flops = defaultdict(dict)
auc_values_time = defaultdict(dict)
aggregated_data = defaultdict(dict)

for condition_name, condition in conditions.items():
    for strategy in strategies:
        aggregated_accuracies, accuracies_std = aggregate_accuracy(data, condition, strategy, budget_columns)
        aggregated_accuracies_test, accuracies_test_std = aggregate_accuracy_test(data, condition, strategy)
        (total_flops, flops_std), (total_time, time_std) = aggregate_eff(data, condition, strategy)
        
        aggregated_data[condition_name][strategy] = pd.DataFrame({
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
        
        auc_values_online[condition_name][strategy] = calculate_auc(aggregated_accuracies)
        auc_values_test[condition_name][strategy] = calculate_auc(aggregated_accuracies_test)
        auc_values_flops[condition_name][strategy] = round(calculate_auc(total_flops) / 1e15, 2)
        auc_values_time[condition_name][strategy] = round(calculate_auc(total_time), 0)

# Print AUC values
for metric, auc_values in [('Online', auc_values_online), ('Test', auc_values_test), ('Flops', auc_values_flops), ('Time', auc_values_time)]:
    print(f'{metric} AUC')
    for condition in conditions:
        print(f'{condition}:')
        for strategy in strategies:
            print(f'  {strategy}: {auc_values[condition][strategy]}')
    print()

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

colors = plt.cm.rainbow(np.linspace(0, 1, len(conditions) * len(strategies)))

# Online Accuracy
color_index = 0
for condition_name in conditions:
    for strategy in strategies:
        df = aggregated_data[condition_name][strategy]
        axs[0, 0].errorbar(budgets, df['online'], yerr=df['online_std'], fmt='-o', capsize=0, label=f"{condition_name} - {strategy}", color=colors[color_index])
        color_index += 1

axs[0, 0].set_title("Online Accuracy Comparison", fontsize=16)
axs[0, 0].set_xlabel('Budget', fontsize=14)
axs[0, 0].set_ylabel('Online Accuracy', fontsize=14)
axs[0, 0].legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
axs[0, 0].tick_params(axis='both', which='major', labelsize=12)

# Test Accuracy
color_index = 0
for condition_name in conditions:
    for strategy in strategies:
        df = aggregated_data[condition_name][strategy]
        axs[0, 1].errorbar(budgets, df['test'], yerr=df['test_std'], fmt='-o', capsize=0, label=f"{condition_name} - {strategy}", color=colors[color_index])
        color_index += 1

axs[0, 1].set_title("Test Accuracy Comparison", fontsize=16)
axs[0, 1].set_xlabel('Budget', fontsize=14)
axs[0, 1].set_ylabel('Test Accuracy', fontsize=14)
axs[0, 1].legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
axs[0, 1].tick_params(axis='both', which='major', labelsize=12)

# FLOPS
color_index = 0
for condition_name in conditions:
    for strategy in strategies:
        df = aggregated_data[condition_name][strategy]
        axs[1, 0].errorbar(budgets, df['total_flops'], yerr=df['flops_std'], fmt='-o', capsize=0, label=f"{condition_name} - {strategy}", color=colors[color_index])
        color_index += 1

axs[1, 0].set_title("FLOPS Comparison", fontsize=16)
axs[1, 0].set_xlabel('Budget', fontsize=14)
axs[1, 0].set_ylabel('Flops', fontsize=14)
axs[1, 0].legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
axs[1, 0].tick_params(axis='both', which='major', labelsize=12)

# Training Time
color_index = 0
for condition_name in conditions:
    for strategy in strategies:
        df = aggregated_data[condition_name][strategy]
        axs[1, 1].errorbar(budgets, df['total_time'], yerr=df['time_std'], fmt='-o', capsize=0, label=f"{condition_name} - {strategy}", color=colors[color_index])
        color_index += 1

axs[1, 1].set_title("Training Time Comparison", fontsize=16)
axs[1, 1].set_xlabel('Budget', fontsize=14)
axs[1, 1].set_ylabel('Training Time', fontsize=14)
axs[1, 1].legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
axs[1, 1].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
# Save the figure to a file
plt.savefig(path + '.png', bbox_inches='tight')
# plt.savefig(path + '.pdf', bbox_inches='tight')
plt.close()