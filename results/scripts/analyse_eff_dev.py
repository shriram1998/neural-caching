import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define paths for the datasets
dataset_paths = {
    'ISEAR': 'results/isear/label',
    'Openbook': 'results/openbook/label',
    'RT-Polarity': 'results/polarity/label',
    'FEVER': 'results/fever/label'
}

budgets = [1000, 1500, 2000, 2500, 3000, 3500]

# Define conditions
conditions = {
    'Complete re-training': {'args/buffer_percent': 1.0, 'args/incremental': 'no'},
    'Incremental': {'args/buffer_percent': 0.0, 'args/incremental': 'yes'},
    'EWC': {'args/buffer_percent': 0.0, 'args/ewc': 'yes', 'args/incremental': 'yes'},
    'Replay': {'args/buffer_percent': 1.0, 'args/incremental': 'yes'},
    'Replay (50%)': {'args/buffer_percent': 0.5, 'args/ewc': 'no', 'args/incremental': 'yes'}
}

# Define strategies
strategies = ['b1','BT']

# Function to extract budget values from column names
def extract_budgets(data):
    return [f"online/{budget}-gold_0 (average)" for budget in budgets]

# Function to aggregate accuracy across seeds for a given condition, strategy, and budget
def aggregate_accuracy(data, condition, strategy, budget_columns, seed):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & 
                             (data['args/strategy'] == strategy) &
                             (data['args/seed'] == seed)]
    means = [filtered_data[col].mean() for col in budget_columns]
    return means

# Function to aggregate test accuracy
def aggregate_accuracy_test(data, condition, strategy, seed):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & 
                             (data['args/strategy'] == strategy) &
                             (data['args/seed'] == seed)]
    means = [filtered_data[filtered_data['args/budget'] == budget]['test/test_gold_acc (last)'].mean() for budget in budgets]
    return means

# Function to aggregate efficiency metrics
def aggregate_eff(data, condition, strategy, seed, flops_col='train/total_flops (last)', time_col='train/total_time_elapsed (last)'):
    filtered_data = data.loc[(data[list(condition)] == pd.Series(condition)).all(axis=1) & 
                             (data['args/strategy'] == strategy) &
                             (data['args/seed'] == seed)]
    
    flops_means = []
    time_means = []
    
    for budget in budgets:
        flops_data = filtered_data[filtered_data['args/budget'] == budget][flops_col]
        time_data = filtered_data[filtered_data['args/budget'] == budget][time_col]
        
        # Filter out rows where flops or time are null or zero
        valid_flops_data = flops_data[(flops_data.notnull()) & (flops_data != 0)]
        valid_time_data = time_data[(time_data.notnull()) & (time_data != 0)]
        
        flops_means.append(valid_flops_data.mean() if not valid_flops_data.empty else np.nan)
        time_means.append(valid_time_data.mean() if not valid_time_data.empty else np.nan)
    
    return flops_means, time_means

# Function to calculate AUC using the trapezoidal rule
def calculate_auc(num, den=budgets):
    auc = np.trapz(num, den)
    normalized_auc = auc / (den[-1] - den[0])
    return round(normalized_auc, 3)

# Process each dataset
overall_auc_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for dataset_name, path in dataset_paths.items():
    # Load the data
    data = pd.read_csv(path + '.csv')
    
    # Extract budget columns
    budget_columns = extract_budgets(data)
    
    # Get unique seeds
    seeds = data['args/seed'].unique()
    
    # Aggregate data, generate tables, and calculate AUC for each seed
    auc_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for seed in seeds:
        for condition_name, condition in conditions.items():
            for strategy in strategies:
                aggregated_accuracies = aggregate_accuracy(data, condition, strategy, budget_columns, seed)
                aggregated_accuracies_test = aggregate_accuracy_test(data, condition, strategy, seed)
                total_flops, total_time = aggregate_eff(data, condition, strategy, seed)
                
                auc_values['online'][condition_name][strategy].append(calculate_auc(aggregated_accuracies))
                auc_values['test'][condition_name][strategy].append(calculate_auc(aggregated_accuracies_test))
                auc_values['flops'][condition_name][strategy].append(calculate_auc(total_flops))
                auc_values['time'][condition_name][strategy].append(calculate_auc(total_time))
    
    # Store AUC values for each dataset
    print(f"\nResults for {dataset_name} Dataset:")
    for metric in auc_values.keys():
        for condition_name in conditions:
            for strategy in strategies:
                mean_values = auc_values[metric][condition_name][strategy]
                mean = np.mean(mean_values)
                std = np.std(mean_values)
                if metric == 'flops':
                    mean /= 1e15
                    std /= 1e15
                    mean = round(mean, 2)
                    std = round(std, 2)
                elif metric == 'time':
                    mean = round(mean, 0)
                    std = round(std, 0)
                else:
                    mean = round(mean, 3)
                    std = round(std, 3)
                
                print(f'{metric.capitalize()} AUC for {condition_name} - {strategy}: {mean} ± {std}')
                
                # Store in overall dictionary for later calculation
                overall_auc_values[metric][condition_name][strategy].append({
                    'mean': mean,
                    'std': std
                })

# Calculate overall mean and standard deviation across datasets
overall_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

print("\nOverall Results for All Datasets Combined:")
for metric in overall_auc_values.keys():
    for condition_name in conditions:
        for strategy in strategies:
            # Calculate mean of the means
            means = [dataset['mean'] for dataset in overall_auc_values[metric][condition_name][strategy]]
            overall_mean = np.mean(means)
            
            # Calculate RMS of the standard deviations
            stds = [dataset['std'] for dataset in overall_auc_values[metric][condition_name][strategy]]
            overall_std = np.sqrt(np.mean(np.square(stds)))
            
            if metric == 'flops':
                overall_mean = round(overall_mean, 2)
                overall_std = round(overall_std, 2)
            elif metric == 'time':
                overall_mean = round(overall_mean, 0)
                overall_std = round(overall_std, 0)
            else:
                overall_mean = round(overall_mean, 3)
                overall_std = round(overall_std, 3)
            
            overall_stats[metric][condition_name][strategy]['mean'] = overall_mean
            overall_stats[metric][condition_name][strategy]['std'] = overall_std
            
            print(f'{metric.capitalize()} AUC for {condition_name} - {strategy}: {overall_mean} ± {overall_std}')
