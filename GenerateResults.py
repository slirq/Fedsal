#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import pickle 
# Ensure output directories exist
if not os.path.exists("final_output"):
    os.makedirs("final_output")

if not os.path.exists("final_output/csv"):
    os.makedirs("final_output/csv")

# Function to load data and calculate statistics
def calculate_accuracy_statistics(base_dir, tud_list, methods):
    results = []
    for tud in tud_list:
        for k in methods:
            avg_acc = []
            for i in range(1, 6):
                file_path = f'{base_dir}{tud}/repeats/{i}_accuracy_{k}.csv'
                df = pd.read_csv(file_path)
                avg_acc.append((sum(df['test_acc']) / len(df)) * 100)
            mean_acc = np.mean(avg_acc)
            std_dev = np.std(avg_acc)
            results.append({'Method': k, 'Dataset': tud, 'Mean Accuracy': mean_acc, 'Std Deviation': std_dev})
            print(f'{k} avg accuracy on {tud} = {mean_acc:.2f} ± {std_dev:.2f}')
    return pd.DataFrame(results)

# Load and process data for multi-dataset scenarios
multi_base_dir = './outputs/seqLen10/multiDS-nonOverlap/'
multi_tud = ['molecules', 'biochem', 'mix']
multi_methods = ['selftrain_GC', 'fedavg_GC', 'fedsal', 'fedsalplus']

multi_results = calculate_accuracy_statistics(multi_base_dir, multi_tud, multi_methods)
multi_results.to_csv("final_output/csv/multi_dataset_results.csv", index=False)

# Load and process data for single-dataset scenarios
single_base_dir = './outputs/seqLen10/oneDS-nonOverlap/'
single_tud = ['IMDB-BINARY-10clients', 'NCI1-30clients', 'PROTEINS-10clients']
single_methods = ['selftrain_GC', 'fedavg_GC', 'fedsal']

single_results = calculate_accuracy_statistics(single_base_dir, single_tud, single_methods)
single_results.to_csv("final_output/csv/single_dataset_results.csv", index=False)

print('-' * 100)


def calculate_metrics(base_dir, tud_list, methods):
    results = []
    for tud in tud_list:
        for k in methods:
            avg_comm_times = []
            for i in range(1, 6):
                file = f'M_{i}_accuracy_{k}.csv'
                df = pd.read_csv(f'{base_dir}{tud}/repeats/{file}')
                avg_comm_times.append(df['communication_time'].mean())
            mean_comm_time = np.mean(avg_comm_times)
            results.append({'Method': k, 'Dataset': tud, 'Average Communication Time': mean_comm_time})
            print(f'{k} avg communication_time on {tud} = {mean_comm_time:.2f} seconds')
        print('-'*100)
    return pd.DataFrame(results)

# Directories and datasets for multi-dataset and single-dataset scenarios
multi_base_dir = './outputs/seqLen10/multiDS-nonOverlap/'
multi_tud = ['molecules', 'biochem', 'mix']
multi_methods = ['fedavg_GC',  'fedsal', 'fedsalplus'] 

single_base_dir = './outputs/seqLen10/oneDS-nonOverlap/'
single_tud = ['IMDB-BINARY-10clients', 'NCI1-30clients', 'PROTEINS-10clients']
single_methods = ['fedavg_GC', 'fedsal']

# Calculate metrics and save results
multi_results = calculate_metrics(multi_base_dir, multi_tud, multi_methods)
multi_results.to_csv("final_output/csv/multi_dataset_communication_times.csv", index=False)

single_results = calculate_metrics(single_base_dir, single_tud, single_methods)
single_results.to_csv("final_output/csv/single_dataset_communication_times.csv", index=False)


# Example plotting function for convergence speeds
def plot_convergence_speeds(base_dir, domains, file_suffix):
    y = [i for i in range(1, 201)]
    target_round = 50
    fig, axes = plt.subplots(1, len(domains), figsize=(18, 6), sharey=True)
    if file_suffix=='multi_speed':
         methods = ['fedavg_GC', 'fedsal', 'fedsalplus']
    else:
        methods = ['fedavg_GC', 'fedsal']
    for ax, tud in zip(axes, domains):
        name_tud = tud.split('-')[0]
        for k in methods:
            all_metrics = []
            for i in range(1, 6):
                file_path = f'{base_dir}{tud}/repeats/{i}_accuracy_{k}.pkl'
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                all_metrics.append(pd.Series(data['convergence_speed']))
            avg_convergence_speed = pd.concat(all_metrics, axis=1).mean(axis=1)
            smoothed_convergence_speed = gaussian_filter1d(avg_convergence_speed, sigma=10)
            if k in ['fedsal', 'fedsalplus']:
                ax.plot(y, smoothed_convergence_speed, label=f'{k}', linestyle='-', linewidth=2.5)
            else:
                ax.plot(y, smoothed_convergence_speed, label=f'{k}', linestyle='dashed', linewidth=2.5)


            # Print accuracy at communication round 50
            if len(smoothed_convergence_speed) >= target_round:
                accuracy_at_50 = smoothed_convergence_speed[target_round - 1] * 100
                print(f'Accuracy of {k} at round 50: {accuracy_at_50:.2f}%')

        ax.set_title(f'Convergence Speed on {name_tud}')
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Convergence Speed')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'final_output/{file_suffix}.png', dpi=200)
    plt.show()

# Plot for multi-dataset
plot_convergence_speeds(multi_base_dir, multi_tud, "multi_speed")
# Plot for single-dataset
plot_convergence_speeds(single_base_dir, single_tud, "single_speed")
