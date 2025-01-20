from trials_analysis_v2 import *
from random_trials import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from setup_files_alt import *
from sklearn.preprocessing import MinMaxScaler
import pickle
'''
plt.figure(figsize=(10, 6))

# Loop through each trial and plot the data
for trial, df in cumulative_maxima_dataframes_sobo.items():
    # Extract Iteration and Cumulative Max YLD columns
    plt.plot(df['Iteration'], df['Cumulative Max YLD'], label=f'Trial {trial}')

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Cumulative Max YLD')
plt.title('Iteration vs Cumulative Max YLD for Different Trials (Baybe - SOBO)')

# Add legend to differentiate trials
plt.legend()

# Show the plot
plt.grid(True)
plt.xticks(range(int(min(df['Iteration'])), int(max(df['Iteration'])) + 1))
plt.show()

#mobo plots (yld & ton separate)

# Create a figure for Yield (Cumulative Max YLD)
plt.figure(figsize=(10, 6))

# Loop through each trial and plot the Cumulative Max YLD
for trial, df in cumulative_maxima_dataframes_mobo.items():
    # Extract Iteration and Cumulative Max YLD columns
    plt.plot(df['Iteration'], df['Cumulative Max YLD'], label=f'Trial {trial}')

# Add labels and title for Yield
plt.xlabel('Iteration')
plt.ylabel('Cumulative Max YLD')
plt.title('Iteration vs Cumulative Max YLD for Different Trials (Baybe - MOBO)')

# Add legend
plt.legend()

# Show the Yield plot
plt.grid(True)
plt.xticks(range(int(min(df['Iteration'])), int(max(df['Iteration'])) + 1))
plt.show()

# Create a figure for TON (Cumulative Max TON)
plt.figure(figsize=(10, 6))

# Loop through each trial and plot the Cumulative Max TON
for trial, df in cumulative_maxima_dataframes_mobo.items():
    # Extract Iteration and Cumulative Max TON columns
    plt.plot(df['Iteration'], df['Cumulative Max TON'], label=f'Trial {trial}')

# Add labels and title for TON
plt.xlabel('Iteration')
plt.ylabel('Cumulative Max TON')
plt.title('Iteration vs Cumulative Max TON for Different Trials (Baybe - MOBO)')

# Add legend
plt.legend()

# Show the TON plot
plt.grid(True)
plt.xticks(range(int(min(df['Iteration'])), int(max(df['Iteration'])) + 1))
plt.show()


#bofire maxima

# Plot Cumulative Max Yield
plt.figure(figsize=(10, 5))
for trial, df in cumulative_maxima_dataframes_bofire.items():
    plt.plot(df["Iteration"], df["Cumulative_Max_Yield"], label=f"Trial {trial}")

plt.title("Iteration vs Cumulative Max Yield for Different Trials (BoFire - MOBO)")
plt.xlabel("Iteration")
plt.ylabel("Cumulative Max Yield")
plt.legend()
plt.grid(True)
plt.xticks(range(int(min(df['Iteration'])), int(max(df['Iteration'])) + 1))
plt.show()

# Plot Cumulative Max TON
plt.figure(figsize=(10, 5))
for trial, df in cumulative_maxima_dataframes_bofire.items():
    plt.plot(df["Iteration"], df["Cumulative_Max_TON"], label=f"Trial {trial}")

plt.title("Iteration vs Cumulative Max TON for Different Trials (BoFire - MOBO)")
plt.xlabel("Iteration")
plt.ylabel("Cumulative Max TON")
plt.legend()
plt.grid(True)
plt.xticks(range(int(min(df['Iteration'])), int(max(df['Iteration'])) + 1))
plt.show()





#averaged trials graph:



# Function to compute average cumulative maxima across trials for a given dictionary of DataFrames
def compute_averages(cumulative_maxima_dataframes, column_name):
    # Concatenate all dataframes for the model
    all_trials_df = pd.concat(cumulative_maxima_dataframes.values(), ignore_index=True)
    
    # Group by 'Iteration' and compute the mean of the specified column
    averaged_df = all_trials_df.groupby("Iteration")[column_name].mean().reset_index()

    return averaged_df

# Compute average cumulative maxima for Yield (YLD) and TON for each model
# SOBO
avg_yld_sobo = compute_averages(cumulative_maxima_dataframes_sobo, "Cumulative Max YLD")


# MOBO
avg_yld_mobo = compute_averages(cumulative_maxima_dataframes_mobo, "Cumulative Max YLD")
avg_ton_mobo = compute_averages(cumulative_maxima_dataframes_mobo, "Cumulative Max TON")

# BOFIRE
avg_yld_bofire = compute_averages(cumulative_maxima_dataframes_bofire, "Cumulative_Max_Yield")
avg_ton_bofire = compute_averages(cumulative_maxima_dataframes_bofire, "Cumulative_Max_TON")

# Plot YLD graph
plt.figure(figsize=(10, 6))
plt.plot(avg_yld_sobo["Iteration"], avg_yld_sobo["Cumulative Max YLD"], label="BayBE (SOBO)", linestyle="--", marker="o")
plt.plot(avg_yld_mobo["Iteration"], avg_yld_mobo["Cumulative Max YLD"], label="BayBE (MOBO)", linestyle="--", marker="s")
plt.plot(avg_yld_bofire["Iteration"], avg_yld_bofire["Cumulative_Max_Yield"], label="BoFire (MOBO)", linestyle="--", marker="^")

plt.title("Iteration vs Average Cumulative Max Yield (YLD) for Different Models")
plt.xlabel("Iteration")
plt.ylabel("Average Cumulative Max Yield (YLD)")
plt.legend()
plt.grid(True)
plt.xticks(range(int(min(df['Iteration'])), int(max(df['Iteration'])) + 1))
plt.show()

# Plot TON graph
plt.figure(figsize=(10, 6))

plt.plot(avg_ton_mobo["Iteration"], avg_ton_mobo["Cumulative Max TON"], label="BayBE (MOBO)", linestyle="--", marker="s")
plt.plot(avg_ton_bofire["Iteration"], avg_ton_bofire["Cumulative_Max_TON"], label="BoFire (MOBO)", linestyle="--", marker="^")

plt.title("Iteration vs Average Cumulative Max TON for Different Models")
plt.xlabel("Iteration")
plt.ylabel("Average Cumulative Max TON")
plt.legend()
plt.grid(True)
plt.xticks(range(int(min(df['Iteration'])), int(max(df['Iteration'])) + 1))
plt.show()

'''

# Load the dictionary from the pickle file
with open(pickle_file_path, "rb") as pkl_file:
    loaded_results_dict_random = pickle.load(pkl_file)



# Function to compute average and standard deviation for cumulative maxima
def compute_stats(cumulative_maxima_dataframes, column_name):
    all_trials_df = pd.concat(cumulative_maxima_dataframes.values(), ignore_index=True)

    # Debugging: Check if the column exists
    if column_name not in all_trials_df.columns:
        raise KeyError(f"Column not found: {column_name}. Available columns: {all_trials_df.columns}")

    # Group by 'Iteration' and compute mean and standard deviation
    stats_df = all_trials_df.groupby("Iteration")[column_name].agg(['mean', 'std']).reset_index()
    stats_df.rename(columns={"mean": "Average", "std": "Standard Deviation"}, inplace=True)
    return stats_df

# Compute stats for Yield (YLD) and TON for SOBO, MOBO, and BoFire
yld_sobo_stats = compute_stats(cumulative_maxima_dataframes_sobo, "Cumulative Max YLD")


yld_mobo_stats = compute_stats(cumulative_maxima_dataframes_mobo, "Cumulative Max YLD")
ton_mobo_stats = compute_stats(cumulative_maxima_dataframes_mobo, "Cumulative Max TON")

yld_bofire_stats = compute_stats(cumulative_maxima_dataframes_bofire, "Cumulative_Max_Yield")
ton_bofire_stats = compute_stats(cumulative_maxima_dataframes_bofire, "Cumulative_Max_TON")

yld_random_stats = compute_stats(loaded_results_dict_random,"Cumulative_Max_Yield")
ton_random_stats = compute_stats(loaded_results_dict_random,"Cumulative_Max_TON")




#shaded region

# Plot Yield (YLD) with Standard Deviation as a Shaded Region
plt.figure(figsize=(10, 6))

# SOBO
plt.plot(yld_sobo_stats["Iteration"], yld_sobo_stats["Average"], label="BayBE (SOBO)", linestyle="--", marker="o")
plt.fill_between(
    yld_sobo_stats["Iteration"],
    yld_sobo_stats["Average"] - yld_sobo_stats["Standard Deviation"],
    yld_sobo_stats["Average"] + yld_sobo_stats["Standard Deviation"],
    alpha=0.2, label="SOBO ± Std Dev"
)

# MOBO
plt.plot(yld_mobo_stats["Iteration"], yld_mobo_stats["Average"], label="BayBE (MOBO)", linestyle="--", marker="s")
plt.fill_between(
    yld_mobo_stats["Iteration"],
    yld_mobo_stats["Average"] - yld_mobo_stats["Standard Deviation"],
    yld_mobo_stats["Average"] + yld_mobo_stats["Standard Deviation"],
    alpha=0.2, label="MOBO ± Std Dev"
)

# BoFire
plt.plot(yld_bofire_stats["Iteration"], yld_bofire_stats["Average"], label="BoFire", linestyle="--", marker="^")
plt.fill_between(
    yld_bofire_stats["Iteration"],
    yld_bofire_stats["Average"] - yld_bofire_stats["Standard Deviation"],
    yld_bofire_stats["Average"] + yld_bofire_stats["Standard Deviation"],
    alpha=0.2, label="BoFire ± Std Dev"
)

# random
plt.plot(yld_random_stats["Iteration"], yld_random_stats["Average"], label="Random search", linestyle="--", marker="*")
plt.fill_between(
    yld_random_stats["Iteration"],
    yld_random_stats["Average"] - yld_random_stats["Standard Deviation"],
    yld_random_stats["Average"] + yld_random_stats["Standard Deviation"],
    alpha=0.2, label="Random search ± Std Dev"
)

plt.title("Iteration vs Average Cumulative Max Yield (YLD) with Standard Deviation")
plt.xlabel("Iteration")
plt.ylabel("Average Cumulative Max Yield (YLD)")
plt.legend()
plt.grid(True)
plt.xticks(range(int(min(df['Iteration'])), int(max(df['Iteration'])) + 1))
plt.show()

# Plot TON with Standard Deviation as a Shaded Region
plt.figure(figsize=(10, 6))


# MOBO
plt.plot(ton_mobo_stats["Iteration"], ton_mobo_stats["Average"], label="BayBE (MOBO)", linestyle="--", marker="s")
plt.fill_between(
    ton_mobo_stats["Iteration"],
    ton_mobo_stats["Average"] - ton_mobo_stats["Standard Deviation"],
    ton_mobo_stats["Average"] + ton_mobo_stats["Standard Deviation"],
    alpha=0.2, label="BayBE (MOBO) ± Std Dev"
)

# BoFire
plt.plot(ton_bofire_stats["Iteration"], ton_bofire_stats["Average"], label="BoFire", linestyle="--", marker="^")
plt.fill_between(
    ton_bofire_stats["Iteration"],
    ton_bofire_stats["Average"] - ton_bofire_stats["Standard Deviation"],
    ton_bofire_stats["Average"] + ton_bofire_stats["Standard Deviation"],
    alpha=0.2, label="BoFire ± Std Dev"
)

# random
plt.plot(ton_random_stats["Iteration"], ton_random_stats["Average"], label="Random search", linestyle="--", marker="*")
plt.fill_between(
    ton_random_stats["Iteration"],
    ton_random_stats["Average"] - ton_random_stats["Standard Deviation"],
    ton_random_stats["Average"] + ton_random_stats["Standard Deviation"],
    alpha=0.2, label="Random search ± Std Dev"
)

plt.title("Iteration vs Average Cumulative Max TON with Standard Deviation")
plt.xlabel("Iteration")
plt.ylabel("Average Cumulative Max TON")
plt.legend()
plt.grid(True)
plt.xticks(range(int(min(df['Iteration'])), int(max(df['Iteration'])) + 1))
plt.show()
