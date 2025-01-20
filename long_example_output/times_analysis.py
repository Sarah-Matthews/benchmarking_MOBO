from open_file import times_dataframes_sobo, times_dataframes_mobo, times_dataframes_bofire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to compute the average time taken per iteration across all trials
def compute_average_time_and_std(times_dataframes):
    total_time = 0
    total_iterations = 0
    all_times = []
    for trial, df in times_dataframes.items():
        total_time += df["Time_taken"].sum()
        total_iterations += len(df)
        all_times.extend(df["Time_taken"].tolist())
    average_time = total_time / total_iterations
    std_deviation = np.std(all_times)
    return average_time, std_deviation

# Calculate the average time for each method
average_time_sobo, std_sobo = compute_average_time_and_std(times_dataframes_sobo)
average_time_mobo, std_mobo = compute_average_time_and_std(times_dataframes_mobo)
average_time_bofire, std_bofire = compute_average_time_and_std(times_dataframes_bofire)


# Output the results
print(f"Average time per iteration (SOBO): {average_time_sobo:.2f} ± {std_sobo:.2f}")
print(f"Average time per iteration (MOBO): {average_time_mobo:.2f} ± {std_mobo:.2f}")
print(f"Average time per iteration (BOFIRE): {average_time_bofire:.2f} ± {std_bofire:.2f}")


# Given data
methods = ["BayBE SOBO", "BayBE MOBO", "BoFire"]
average_times = [average_time_sobo, average_time_mobo, average_time_bofire]  # Replace with your calculated averages
std_devs = [std_sobo, std_mobo, std_bofire]  # Replace with your calculated standard deviations

# Bar plot with error bars
plt.figure(figsize=(8, 6))
x_positions = np.arange(len(methods))
plt.bar(x_positions, average_times, yerr=std_devs, capsize=5, color=['blue', 'orange', 'green'], alpha=0.7, edgecolor='black')

# Formatting the plot
plt.xticks(x_positions, methods, fontsize=12)
plt.ylabel("Average Time per Iteration (s)", fontsize=12)
plt.xlabel("Methods", fontsize=12)
plt.title("Comparison of Average Time per Iteration", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()