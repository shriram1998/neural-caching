import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the data for multiple datasets
datasets = ['isear', 'openbook', 'polarity', 'fever']

path_base = 'results/{}/all_data'
all_data = {dataset: pd.read_csv(path_base.format(dataset) + '.csv') for dataset in datasets}

budgets = [1000, 1500, 2000, 2500, 3000, 3500]

# Define conditions and strategies
conditions = {
    'Complete re-training': {'args/buffer_percent': 1.0, 'args/ewc': 'no', 'args/incremental': 'no'},
    'Incremental': {'args/buffer_percent': 0.0, 'args/ewc': 'no', 'args/incremental': 'yes'},
    'EWC': {'args/buffer_percent': 0.0, 'args/ewc': 'yes', 'args/incremental': 'yes'},
    'Replay': {'args/buffer_percent': 1.0, 'args/ewc': 'no', 'args/incremental': 'yes'},
    'Replay (50%)': {'args/buffer_percent': 0.5, 'args/ewc': 'no', 'args/incremental': 'yes'}
}

strategies = ['b1']

# Function definitions (unchanged)
def extract_budgets(data):
    return [f"online/{budget}-gold_0 (average)" for budget in budgets]

def aggregate_accuracy(data, condition, strategy, budget_columns):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & (data['args/strategy'] == strategy)]
    means = [filtered_data[col].mean() for col in budget_columns]
    stds = [filtered_data[col].std() for col in budget_columns]
    return means, stds

def aggregate_accuracy_test(data, condition, strategy):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & (data['args/strategy'] == strategy)]
    means = [filtered_data[filtered_data['args/budget'] == budget]['test/test_gold_acc (last)'].mean() for budget in budgets]
    stds = [filtered_data[filtered_data['args/budget'] == budget]['test/test_gold_acc (last)'].std() for budget in budgets]
    return means, stds

def aggregate_eff(data, condition, strategy, flops_col='train/total_flops (last)', time_col='train/total_time_elapsed (last)'):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & (data['args/strategy'] == strategy)]
    flops_means = [filtered_data[filtered_data['args/budget'] == budget][flops_col].mean() for budget in budgets]
    flops_stds = [filtered_data[filtered_data['args/budget'] == budget][flops_col].std() for budget in budgets]
    time_means = [filtered_data[filtered_data['args/budget'] == budget][time_col].mean() for budget in budgets]
    time_stds = [filtered_data[filtered_data['args/budget'] == budget][time_col].std() for budget in budgets]
    return (flops_means, flops_stds), (time_means, time_stds)

# Extract budget columns (use the first dataset as reference)
budget_columns = extract_budgets(list(all_data.values())[0])

# Aggregate data for individual datasets and conditions
aggregated_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

for dataset, data in all_data.items():
    for condition_name, condition in conditions.items():
        for strategy in strategies:
            aggregated_accuracies, accuracies_std = aggregate_accuracy(data, condition, strategy, budget_columns)
            aggregated_accuracies_test, accuracies_test_std = aggregate_accuracy_test(data, condition, strategy)
            (total_flops, flops_std), (total_time, time_std) = aggregate_eff(data, condition, strategy)
            
            if strategy not in aggregated_data[dataset][condition_name]:
                aggregated_data[dataset][condition_name][strategy] = {
                    'online': [], 'online_std': [],
                    'test': [], 'test_std': [],
                    'total_flops': [], 'total_flops_std': [],
                    'total_time': [], 'total_time_std': []
                }
            
            aggregated_data[dataset][condition_name][strategy]['online'].append(aggregated_accuracies)
            aggregated_data[dataset][condition_name][strategy]['online_std'].append(accuracies_std)
            aggregated_data[dataset][condition_name][strategy]['test'].append(aggregated_accuracies_test)
            aggregated_data[dataset][condition_name][strategy]['test_std'].append(accuracies_test_std)
            aggregated_data[dataset][condition_name][strategy]['total_flops'].append(total_flops)
            aggregated_data[dataset][condition_name][strategy]['total_flops_std'].append(flops_std)
            aggregated_data[dataset][condition_name][strategy]['total_time'].append(total_time)
            aggregated_data[dataset][condition_name][strategy]['total_time_std'].append(time_std)

# Average across strategies
for dataset in datasets:
    for condition_name in conditions:
        for metric in ['online', 'online_std', 'test', 'test_std', 'total_flops', 'total_flops_std', 'total_time', 'total_time_std']:
            aggregated_data[dataset][condition_name][metric] = np.mean(
                [aggregated_data[dataset][condition_name][strategy][metric] for strategy in strategies],
                axis=0
            )

metrics = ['online', 'test', 'total_flops', 'total_time']
y_labels = ['Accuracy (online)', 'Accuracy (test)', 'FLOPs (E+16)', 'Training Time (s)']
dataset_names = ['ISEAR', 'OPENBOOK', 'RT-POLARITY', 'FEVER']
# Function to create plots for each metric across all datasets
def create_metric_plots(metric, ylabel):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(conditions)))
    
    for i, dataset in enumerate(datasets):
        ax = axs[i // 2, i % 2]
        for j, condition_name in enumerate(conditions):
            data = aggregated_data[dataset][condition_name][metric]
            std = aggregated_data[dataset][condition_name][f'{metric}_std']
            
            # Debugging information
            # print(f"Dataset: {dataset}, Condition: {condition_name}, Metric: {metric}")
            # print(f"Budgets length: {len(budgets)}")
            # print(f"Data length: {len(data)}")
            # print(f"Std length: {len(std)}")
            
            # Ensure data and std are lists or 1D numpy arrays
            data = np.array(data).flatten()
            std = np.array(std).flatten()
            
            # Check if lengths match, if not, trim or pad
            if len(data) != len(budgets):
                min_len = min(len(data), len(budgets))
                data = data[:min_len]
                std = std[:min_len]
                print(f"Warning: Data length mismatch. Trimmed to {min_len}")
            
            ax.errorbar(budgets[:len(data)], data, yerr=std, fmt='-o', capsize=0, label=condition_name, color=colors[j])
        
        ax.set_title(dataset_names[i], fontsize=20)
        if i>=2:
            ax.set_xlabel('Budget (number of LLM calls)', fontsize=18)
        if i%2==0:
            ax.set_ylabel(ylabel, fontsize=18)
        if i==0:
            ax.legend(fontsize=24, loc='best')
        ax.tick_params(axis='both', which='major', labelsize=18)
        # ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/metrics/b1_{metric}_comparison_all_datasets.png', bbox_inches='tight')
    plt.close()

for metric, ylabel in zip(metrics, y_labels):
    create_metric_plots(metric, ylabel)
