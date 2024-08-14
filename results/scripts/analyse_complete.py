import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the data for multiple datasets
datasets = ['isear', 'openbook', 'polarity', 'fever']

path_base = 'results/{}/all_data'
all_data = {dataset: pd.read_csv(path_base.format(dataset) + '.csv') for dataset in datasets}

budgets = [1000, 1500, 2000, 2500, 3000, 3500]

# Define conditions and strategies (unchanged)
conditions = {
    'Complete re-training': {'args/buffer_percent': 1.0, 'args/ewc': 'no', 'args/incremental': 'no'},
    'Incremental': {'args/buffer_percent': 0.0, 'args/ewc': 'no', 'args/incremental': 'yes'},
    'EWC': {'args/buffer_percent': 0.0, 'args/ewc': 'yes', 'args/incremental': 'yes'},
    'Replay': {'args/buffer_percent': 1.0, 'args/ewc': 'no', 'args/incremental': 'yes'},
    'Replay (50%)': {'args/buffer_percent': 0.5, 'args/ewc': 'no', 'args/incremental': 'yes'}
}

strategies = ['b1', 'BT', 'EN', 'CS', 'MV']

# Function definitions (unchanged)
def extract_budgets(data):
    return [f"online/{budget}-gold_0 (average)" for budget in budgets]

def aggregate_accuracy_multi(all_data, condition, strategy, budget_columns):
    all_means = []
    all_stds = []
    for data in all_data.values():
        filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & (data['args/strategy'] == strategy)]
        means = [filtered_data[col].mean() for col in budget_columns]
        stds = [filtered_data[col].std() for col in budget_columns]
        all_means.append(means)
        all_stds.append(stds)
    return np.mean(all_means, axis=0), np.mean(all_stds, axis=0)

def aggregate_accuracy_test_multi(all_data, condition, strategy):
    all_means = []
    all_stds = []
    for data in all_data.values():
        filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & (data['args/strategy'] == strategy)]
        means = [filtered_data[filtered_data['args/budget'] == budget]['test/test_gold_acc (last)'].mean() for budget in budgets]
        stds = [filtered_data[filtered_data['args/budget'] == budget]['test/test_gold_acc (last)'].std() for budget in budgets]
        all_means.append(means)
        all_stds.append(stds)
    return np.mean(all_means, axis=0), np.mean(all_stds, axis=0)

def aggregate_eff_multi(all_data, condition, strategy, flops_col='train/total_flops (last)', time_col='train/total_time_elapsed (last)'):
    all_flops_means, all_flops_stds, all_time_means, all_time_stds = [], [], [], []
    for data in all_data.values():
        filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & (data['args/strategy'] == strategy)]
        flops_means = [filtered_data[filtered_data['args/budget'] == budget][flops_col].mean() for budget in budgets]
        flops_stds = [filtered_data[filtered_data['args/budget'] == budget][flops_col].std() for budget in budgets]
        time_means = [filtered_data[filtered_data['args/budget'] == budget][time_col].mean() for budget in budgets]
        time_stds = [filtered_data[filtered_data['args/budget'] == budget][time_col].std() for budget in budgets]
        all_flops_means.append(flops_means)
        all_flops_stds.append(flops_stds)
        all_time_means.append(time_means)
        all_time_stds.append(time_stds)
    return (np.mean(all_flops_means, axis=0), np.mean(all_flops_stds, axis=0)), (np.mean(all_time_means, axis=0), np.mean(all_time_stds, axis=0))

# Extract budget columns (use the first dataset as reference)
budget_columns = extract_budgets(list(all_data.values())[0])

# Aggregate data for individual strategies
aggregated_data = defaultdict(lambda: defaultdict(dict))

for condition_name, condition in conditions.items():
    for strategy in strategies:
        aggregated_accuracies, accuracies_std = aggregate_accuracy_multi(all_data, condition, strategy, budget_columns)
        aggregated_accuracies_test, accuracies_test_std = aggregate_accuracy_test_multi(all_data, condition, strategy)
        (total_flops, flops_std), (total_time, time_std) = aggregate_eff_multi(all_data, condition, strategy)
        
        aggregated_data[condition_name][strategy] = {
            'online': aggregated_accuracies,
            'online_std': accuracies_std,
            'test': aggregated_accuracies_test,
            'test_std': accuracies_test_std,
            'total_flops': total_flops,
            'total_flops_std': flops_std,
            'total_time': total_time,
            'total_time_std': time_std
        }

# Function to create individual strategy plots
def create_strategy_plot(strategy):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(conditions)))
    
    for i, (plot_metric, title) in enumerate([
        ('online', f'Online Accuracy'),
        ('test', f'Test Accuracy'),
        ('total_flops', f'FLOPS'),
        ('total_time', f'Training Time')
    ]):
        ax = axs[i // 2, i % 2]
        for j, condition_name in enumerate(conditions):
            data = aggregated_data[condition_name][strategy]
            ax.errorbar(budgets, data[plot_metric], 
                        yerr=data[f'{plot_metric}_std'], 
                        fmt='-o', capsize=0, label=condition_name, color=colors[j])
        
        ax.set_title(f"{title}", fontsize=16)
        ax.set_xlabel('Budget', fontsize=14)
        ax.set_ylabel(plot_metric.replace('_', ' ').title(), fontsize=14)
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'results/{strategy}_comparison.png', bbox_inches='tight')
    plt.close()

# Create and save individual strategy plots
for strategy in strategies:
    create_strategy_plot(strategy)

# Aggregate data across strategies
averaged_data = defaultdict(lambda: defaultdict(list))

for condition_name in conditions:
    for strategy in strategies:
        for metric in ['online', 'online_std', 'test', 'test_std', 'total_flops', 'total_flops_std', 'total_time', 'total_time_std']:
            averaged_data[condition_name][metric].append(aggregated_data[condition_name][strategy][metric])

# Average across strategies
for condition_name in conditions:
    for metric in averaged_data[condition_name]:
        averaged_data[condition_name][metric] = np.mean(averaged_data[condition_name][metric], axis=0)

# Create the overall comparison plot (averaged across datasets and strategies)
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

colors = plt.cm.rainbow(np.linspace(0, 1, len(conditions)))

metrics = ['online', 'test', 'total_flops', 'total_time']
titles = ['Online Accuracy', 'Test Accuracy', 'FLOPS', 'Training Time']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axs[i // 2, i % 2]
    for j, condition_name in enumerate(conditions):
        ax.errorbar(budgets, averaged_data[condition_name][metric], 
                    yerr=averaged_data[condition_name][f'{metric}_std'], 
                    fmt='-o', capsize=0, label=condition_name, color=colors[j])

    ax.set_title(f"{title}", fontsize=16)
    ax.set_xlabel('Budget', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
# Save the overall comparison figure
plt.savefig('results/average_across_datasets_and_strategies.png', bbox_inches='tight')
plt.close()