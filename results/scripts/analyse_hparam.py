import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the data
data = pd.read_csv('results/isear/online (5).csv')  # Replace with your file path
budgets = [1000, 1500, 2000, 2500, 3000, 3500]

# Function to extract budget values from column names
def extract_budgets(data, budgets):
    budget_columns = [f"online/{budget}-gold_0 (average)" for budget in budgets]
    return budget_columns

# Function to aggregate accuracies
def aggregate_accuracy(data, hyperparameter, hyperparameter_value, strategy, budget_columns):
    if hyperparameter:
        filtered_data = data[(data[hyperparameter] == hyperparameter_value) & (data['args/strategy'] == strategy)]
    else:
        filtered_data = data[data['args/strategy'] == strategy]
    
    aggregated_accuracies = []
    for col in budget_columns:
        aggregated_accuracies.append(filtered_data[col].mean())
    return aggregated_accuracies

def aggregate_accuracy_test(data, hyperparameter, hyperparameter_value, strategy, budgets):
    if hyperparameter:
        filtered_data = data[(data[hyperparameter] == hyperparameter_value) & (data['args/strategy'] == strategy)]
    else:
        filtered_data = data[data['args/strategy'] == strategy]
    
    aggregated_accuracies_test = []
    for col in budgets:
        temp = filtered_data[filtered_data['args/budget'] == col]
        aggregated_accuracies_test.append(temp['test/test_gold_acc (last)'].mean())
    return aggregated_accuracies_test

# Function to calculate AUC using the trapezoidal rule
def calculate_auc(accuracies, budgets):
    auc = np.trapz(accuracies, budgets)
    budget_range = 2500
    normalized_auc = auc / budget_range
    return round(normalized_auc, 3)

# Function to analyze hyperparameter
def analyze_hyperparameter(data, hyperparameter, hyperparameter_values, budgets):
    budget_columns = extract_budgets(data, budgets)
    strategies = data['args/strategy'].unique()
    
    auc_values_online = defaultdict(dict)
    auc_values_test = defaultdict(dict)
    aggregated_data = defaultdict(dict)

    if hyperparameter:
        for value in hyperparameter_values:
            for strategy in strategies:
                aggregated_accuracies = aggregate_accuracy(data, hyperparameter, value, strategy, budget_columns)
                aggregated_accuracies_test = aggregate_accuracy_test(data, hyperparameter, value, strategy, budgets)
                aggregated_data[value][strategy] = pd.DataFrame({
                    'budget': budgets,
                    'online': aggregated_accuracies,
                    'test': aggregated_accuracies_test
                })
                
                auc_values_online[value][strategy] = calculate_auc(aggregated_data[value][strategy]['online'], budgets)
                auc_values_test[value][strategy] = calculate_auc(aggregated_data[value][strategy]['test'], budgets)
    else:
        value = 'all'
        for strategy in strategies:
            aggregated_accuracies = aggregate_accuracy(data, None, None, strategy, budget_columns)
            aggregated_accuracies_test = aggregate_accuracy_test(data, None, None, strategy, budgets)
            aggregated_data[value][strategy] = pd.DataFrame({
                'budget': budgets,
                'online': aggregated_accuracies,
                'test': aggregated_accuracies_test
            })
            
            auc_values_online[value][strategy] = calculate_auc(aggregated_data[value][strategy]['online'], budgets)
            auc_values_test[value][strategy] = calculate_auc(aggregated_data[value][strategy]['test'], budgets)
    
    return auc_values_online, auc_values_test, aggregated_data, strategies

# Parameters
hyperparameter = 'args/ewc_lambda'
hyperparameter_values = [0.2, 0.6, 0.8]

# hyperparameter = 'args/buffer_percent'
# hyperparameter_values = [0.25, 0.5, 0.75]

# hyperparameter=None
# hyperparameter_values=[]

# Analyze hyperparameter
auc_values_online, auc_values_test, aggregated_data, strategies = analyze_hyperparameter(data, hyperparameter, hyperparameter_values, budgets)

# Plot results
fig, axs = plt.subplots(1, len(hyperparameter_values) if hyperparameter else 1, figsize=(15, 5))
if hyperparameter:
    for idx, value in enumerate(hyperparameter_values):
        for strategy in strategies:
            axs[idx].plot(budgets, aggregated_data[value][strategy]['online'], label=f'{strategy} Online Acc')
            axs[idx].plot(budgets, aggregated_data[value][strategy]['test'], label=f'{strategy} Test Acc')
            axs[idx].set_title(f"{hyperparameter}: {value}")
            axs[idx].set_xlabel('Budget')
            axs[idx].set_ylabel('Accuracy')
            axs[idx].legend()
else:
    value = 'all'
    for strategy in strategies:
        axs.plot(budgets, aggregated_data[value][strategy]['online'], label=f'{strategy} Online Acc')
        axs.plot(budgets, aggregated_data[value][strategy]['test'], label=f'{strategy} Test Acc')
        axs.set_title("All Data")
        axs.set_xlabel('Budget')
        axs.set_ylabel('Accuracy')
        axs.legend()
plt.tight_layout()
plt.show()

print('Online AUC')
print(auc_values_online)
print('Test AUC')
print(auc_values_test)