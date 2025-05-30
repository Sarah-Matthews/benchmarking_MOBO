from trials_analysis_v2_corr import *
from random_trials import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from setup_file_final import *
from sklearn.preprocessing import MinMaxScaler
import pickle
from open_file_corr import *




modified_measurements_dataframes_mobo_with_init = {}

for trial, df in modified_measurements_dataframes_mobo.items():
    
    df['Cumulative Max YLD'] = df['Yield'].cummax()
    df['Cumulative Max TON'] = df['Ton'].cummax()
    
    
    modified_measurements_dataframes_mobo_with_init[trial] = df[['Iteration', 'Yield', 'Ton', 'Cumulative Max YLD', 'Cumulative Max TON']]


modified_measurements_dataframes_sobo_with_init = {}

for trial, df in modified_measurements_dataframes_sobo.items():
    
    df['Cumulative Max YLD'] = df['Yield'].cummax()

    modified_measurements_dataframes_sobo_with_init[trial] = df[['Iteration', 'Yield', 'Cumulative Max YLD']]



modified_measurements_dataframes_bofire_with_init = {}

for trial, df in modified_measurements_dataframes_bofire.items():
    
    df['Cumulative Max YLD'] = df['Yield'].cummax()
    df['Cumulative Max TON'] = df['Ton'].cummax()
    
    
    modified_measurements_dataframes_bofire_with_init[trial] = df[['Iteration', 'Yield', 'Ton', 'Cumulative Max YLD', 'Cumulative Max TON']]


initialised_points_per_trial = {}

for trial, df in modified_measurements_dataframes_bofire_with_init.items():
    
    initialised_points_per_trial[trial] = df.head(5)

print(initialised_points_per_trial)




with open('results_dict_random.pkl', "rb") as pkl_file:
    loaded_results_dict_random = pickle.load(pkl_file)



#function to compute average and standard deviation for cumulative maxima
def compute_stats(cumulative_maxima_dataframes, column_name):
    all_trials_df = pd.concat(cumulative_maxima_dataframes.values(), ignore_index=True)

    
    if column_name not in all_trials_df.columns:
        raise KeyError(f"Column not found: {column_name}. Available columns: {all_trials_df.columns}")

    
    stats_df = all_trials_df.groupby("Iteration")[column_name].agg(['mean', 'std']).reset_index()
    stats_df.rename(columns={"mean": "Average", "std": "Standard Deviation"}, inplace=True)
    return stats_df


yld_sobo_stats = compute_stats(cumulative_maxima_dataframes_sobo, "Cumulative Max YLD")


yld_mobo_stats = compute_stats(cumulative_maxima_dataframes_mobo, "Cumulative Max YLD")
ton_mobo_stats = compute_stats(cumulative_maxima_dataframes_mobo, "Cumulative Max TON")

yld_bofire_stats = compute_stats(cumulative_maxima_dataframes_bofire, "Cumulative_Max_Yield")
ton_bofire_stats = compute_stats(cumulative_maxima_dataframes_bofire, "Cumulative_Max_TON")

yld_random_stats = compute_stats(loaded_results_dict_random,"Cumulative_Max_Yield")
ton_random_stats = compute_stats(loaded_results_dict_random,"Cumulative_Max_TON")


initial_points_yld = compute_stats(initialised_points_per_trial, "Cumulative Max YLD")
initial_points_ton = compute_stats(initialised_points_per_trial, "Cumulative Max TON")


#shaded region
'''
# Plot Yield (YLD) with Standard Deviation as a Shaded Region
plt.figure(figsize=(10, 6))

# SOBO
plt.plot(yld_sobo_stats["Iteration"], yld_sobo_stats["Average"], label="BayBE (SOBO)", linestyle="--", marker="o", color = '#5D8B9E')
plt.fill_between(
    yld_sobo_stats["Iteration"],
    yld_sobo_stats["Average"] - yld_sobo_stats["Standard Deviation"],
    yld_sobo_stats["Average"] + yld_sobo_stats["Standard Deviation"],
    alpha=0.2,
    color = '#5D8B9E'
)

# MOBO
plt.plot(yld_mobo_stats["Iteration"], yld_mobo_stats["Average"], label="BayBE (MOBO)", linestyle="--", marker="s", color = '#ff7f0e')
plt.fill_between(
    yld_mobo_stats["Iteration"],
    yld_mobo_stats["Average"] - yld_mobo_stats["Standard Deviation"],
    yld_mobo_stats["Average"] + yld_mobo_stats["Standard Deviation"],
    alpha=0.2, 
    color = '#ff7f0e'
)

# BoFire
plt.plot(yld_bofire_stats["Iteration"], yld_bofire_stats["Average"], label="BoFire", linestyle="--", marker="^", color = '#2ca02c')
plt.fill_between(
    yld_bofire_stats["Iteration"],
    yld_bofire_stats["Average"] - yld_bofire_stats["Standard Deviation"],
    yld_bofire_stats["Average"] + yld_bofire_stats["Standard Deviation"],
    alpha=0.2, 
    color = '#2ca02c'
)

#random
plt.plot(yld_random_stats["Iteration"], yld_random_stats["Average"], label="Random search", linestyle="--", marker="*", color = '#D76B6B')
plt.fill_between(
    yld_random_stats["Iteration"],
    yld_random_stats["Average"] - yld_random_stats["Standard Deviation"],
    yld_random_stats["Average"] + yld_random_stats["Standard Deviation"],
    alpha=0.2, 
    color = '#D76B6B'
)



#plt.title("Iteration vs Average Cumulative Max Yield (YLD) with Standard Deviation")
plt.xlabel("Iteration", fontsize = 14)
plt.ylabel("Cumulative maximum yield", fontsize = 14)
plt.legend()
plt.minorticks_on()
plt.grid(True)
plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   # Lighter gridlines for y-axis
plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.xticks(range(int(min(yld_sobo_stats['Iteration']))-1, int(max(yld_sobo_stats['Iteration'])) + 1, 5))
plt.show()

# Plot TON with Standard Deviation as a Shaded Region
plt.figure(figsize=(10, 6))


# MOBO
plt.plot(ton_mobo_stats["Iteration"], ton_mobo_stats["Average"], label="BayBE (MOBO)", linestyle="--", marker="s", color = '#ff7f0e')
plt.fill_between(
    ton_mobo_stats["Iteration"],
    ton_mobo_stats["Average"] - ton_mobo_stats["Standard Deviation"],
    ton_mobo_stats["Average"] + ton_mobo_stats["Standard Deviation"],
    alpha=0.2, 
    color = '#ff7f0e'
)

# BoFire
plt.plot(ton_bofire_stats["Iteration"], ton_bofire_stats["Average"], label="BoFire", linestyle="--", marker="^", color = '#2ca02c')
plt.fill_between(
    ton_bofire_stats["Iteration"],
    ton_bofire_stats["Average"] - ton_bofire_stats["Standard Deviation"],
    ton_bofire_stats["Average"] + ton_bofire_stats["Standard Deviation"],
    alpha=0.2, 
    color = '#2ca02c'
)

# random
plt.plot(ton_random_stats["Iteration"], ton_random_stats["Average"], label="Random search", linestyle="--", marker="*", color = '#D76B6B')
plt.fill_between(
    ton_random_stats["Iteration"],
    ton_random_stats["Average"] - ton_random_stats["Standard Deviation"],
    ton_random_stats["Average"] + ton_random_stats["Standard Deviation"],
    alpha=0.2, 
    color = '#D76B6B'
)

#plt.title("Iteration vs Average Cumulative Max TON with Standard Deviation")
plt.xlabel("Iteration",fontsize = 14)
plt.ylabel("Cumulative maximum TON", fontsize = 14)
plt.legend()
plt.minorticks_on()
plt.grid(True)
plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   # Lighter gridlines for y-axis
plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.xticks(range(int(min(yld_sobo_stats['Iteration']))-1, int(max(yld_sobo_stats['Iteration'])) + 1, 5))
plt.show()



#new graphs with 5 initialised points included

'''
'''

yld_mobo_stats_full_init = compute_stats(modified_measurements_dataframes_mobo_with_init, "Cumulative Max YLD")
ton_mobo_stats_full_init = compute_stats(modified_measurements_dataframes_mobo_with_init, "Cumulative Max TON")

yld_sobo_stats_full_init = compute_stats(modified_measurements_dataframes_sobo_with_init, "Cumulative Max YLD")


yld_bofire_stats_full_init = compute_stats(modified_measurements_dataframes_bofire_with_init, "Cumulative Max YLD")
ton_bofire_stats_full_init = compute_stats(modified_measurements_dataframes_bofire_with_init, "Cumulative Max TON")




# Plot Yield (YLD) with Standard Deviation as a Shaded Region
plt.figure(figsize=(10, 6))


# SOBO
plt.plot(yld_sobo_stats_full_init["Iteration"], yld_sobo_stats_full_init["Average"], label="BayBE (SOBO)", linestyle="--", marker="o")
plt.fill_between(
    yld_sobo_stats_full_init["Iteration"].iloc[5:],  # Exclude first 5 points
    yld_sobo_stats_full_init["Average"].iloc[5:] - yld_sobo_stats_full_init["Standard Deviation"].iloc[5:],
    yld_sobo_stats_full_init["Average"].iloc[5:] + yld_sobo_stats_full_init["Standard Deviation"].iloc[5:],
    alpha=0.2, label="SOBO ± Std Dev"
)

# MOBO
plt.plot(yld_mobo_stats_full_init["Iteration"], yld_mobo_stats_full_init["Average"], label="BayBE (MOBO)", linestyle="--", marker="s")
plt.fill_between(
    yld_mobo_stats_full_init["Iteration"].iloc[5:],  # Exclude first 5 points
    yld_mobo_stats_full_init["Average"].iloc[5:] - yld_mobo_stats_full_init["Standard Deviation"].iloc[5:],
    yld_mobo_stats_full_init["Average"].iloc[5:] + yld_mobo_stats_full_init["Standard Deviation"].iloc[5:],
    alpha=0.2, label="MOBO ± Std Dev"
)

# BoFire
plt.plot(yld_bofire_stats_full_init["Iteration"], yld_bofire_stats_full_init["Average"], label="BoFire", linestyle="--", marker="^")
plt.fill_between(
    yld_bofire_stats_full_init["Iteration"].iloc[5:],  # Exclude first 5 points
    yld_bofire_stats_full_init["Average"].iloc[5:] - yld_bofire_stats_full_init["Standard Deviation"].iloc[5:],
    yld_bofire_stats_full_init["Average"].iloc[5:] + yld_bofire_stats_full_init["Standard Deviation"].iloc[5:],
    alpha=0.2, label="BoFire ± Std Dev"
)

# initial
plt.plot(initial_points_yld["Iteration"], initial_points_yld["Average"], label="Initialised points", linestyle="--", marker="o")
#plt.fill_between(
    #initial_points_yld["Iteration"].iloc[:5],
    #initial_points_yld["Average"].iloc[:5] - initial_points_yld["Standard Deviation"].iloc[:5],
    #initial_points_yld["Average"].iloc[:5] + initial_points_yld["Standard Deviation"].iloc[:5],
    #alpha=0.2, label="Initialised points"
#)


plt.title("Iteration vs Average Cumulative Max Yield (YLD) with Standard Deviation")
plt.xlabel("Iteration")
plt.ylabel("Average Cumulative Max Yield (YLD)")
plt.legend()
plt.minorticks_on()
plt.grid(True)
plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   # Lighter gridlines for y-axis
plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.xticks(range(int(min(yld_mobo_stats_full_init['Iteration']))-1, int(max(yld_mobo_stats_full_init['Iteration'])) + 1, 5))
plt.show()



# Adjust iterations for initial points
initial_points_yld_iterations = [-4, -3, -2, -1, 0]
iterations = yld_sobo_stats_full_init["Iteration"].iloc[:-5].to_numpy()  # Convert to numpy array
iterations_with_zero = np.insert(iterations, 0, 0)  # Prepend 0

# Plot Yield (YLD) with Standard Deviation as a Shaded Region
plt.figure(figsize=(10, 6))

# SOBO
plt.plot(iterations_with_zero, yld_sobo_stats_full_init["Average"].iloc[4:], label="BayBE (SOBO)", linestyle="--", marker="o", markevery=slice(1, None))
plt.fill_between(
    yld_sobo_stats_full_init["Iteration"].iloc[:-5],  
    yld_sobo_stats_full_init["Average"].iloc[5:] - yld_sobo_stats_full_init["Standard Deviation"].iloc[5:],
    yld_sobo_stats_full_init["Average"].iloc[5:] + yld_sobo_stats_full_init["Standard Deviation"].iloc[5:],
    alpha=0.2, 
)

# MOBO
plt.plot(iterations_with_zero, yld_mobo_stats_full_init["Average"].iloc[4:], label="BayBE (MOBO)", linestyle="--", marker="s",markevery=slice(1, None))
plt.fill_between(
    yld_mobo_stats_full_init["Iteration"].iloc[:-5],  
    yld_mobo_stats_full_init["Average"].iloc[5:] - yld_mobo_stats_full_init["Standard Deviation"].iloc[5:],
    yld_mobo_stats_full_init["Average"].iloc[5:] + yld_mobo_stats_full_init["Standard Deviation"].iloc[5:],
    alpha=0.2, 
)

# BoFire
plt.plot(iterations_with_zero, yld_bofire_stats_full_init["Average"].iloc[4:], label="BoFire", linestyle="--", marker="^", markevery=slice(1, None))
plt.fill_between(
    yld_bofire_stats_full_init["Iteration"].iloc[:-5], 
    yld_bofire_stats_full_init["Average"].iloc[5:] - yld_bofire_stats_full_init["Standard Deviation"].iloc[5:],
    yld_bofire_stats_full_init["Average"].iloc[5:] + yld_bofire_stats_full_init["Standard Deviation"].iloc[5:],
    alpha=0.2, 
)

# Initial points
plt.plot(initial_points_yld_iterations, initial_points_yld["Average"], label="Initialised points", linestyle="--", marker="x", color="black")
#plt.fill_between(
    #initial_points_yld_iterations,
    #initial_points_yld["Average"] - initial_points_yld["Standard Deviation"],
    #initial_points_yld["Average"] + initial_points_yld["Standard Deviation"],
    #alpha=0.2,  color="grey"
#)


#plt.axvline(x=0, linestyle=":", color="black", label="Initialisation Ends")

#plt.title("Iteration vs Average Cumulative Max Yield (YLD) with Standard Deviation")
plt.xlabel("Iteration", fontsize=13)
plt.ylabel("Yield (%)", fontsize=13)
plt.legend(loc="lower right")
plt.minorticks_on()
plt.grid(True)
plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   # Lighter gridlines for y-axis
plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.xticks(range(-5, int(max(yld_mobo_stats_full_init['Iteration'])) -4, 5))
plt.show()




# Plot ton with Standard Deviation as a Shaded Region - adding initial points
plt.figure(figsize=(10, 6))


# MOBO
plt.plot(iterations_with_zero, ton_mobo_stats_full_init["Average"].iloc[4:], label="BayBE (MOBO)", linestyle="--", marker="s", markevery=slice(1, None), color='#ff7f0e')
plt.fill_between(
    ton_mobo_stats_full_init["Iteration"].iloc[:-5],  # Exclude first 5 points
    ton_mobo_stats_full_init["Average"].iloc[5:] - ton_mobo_stats_full_init["Standard Deviation"].iloc[5:],
    ton_mobo_stats_full_init["Average"].iloc[5:] + ton_mobo_stats_full_init["Standard Deviation"].iloc[5:],
    alpha=0.2, color='#ff7f0e'
)

# BoFire
plt.plot(iterations_with_zero, ton_bofire_stats_full_init["Average"].iloc[4:], label="BoFire", linestyle="--", marker="^", markevery=slice(1, None), color='#2ca02c')
plt.fill_between(
    ton_bofire_stats_full_init["Iteration"].iloc[:-5],  # Exclude first 5 points
    ton_bofire_stats_full_init["Average"].iloc[5:] - ton_bofire_stats_full_init["Standard Deviation"].iloc[5:],
    ton_bofire_stats_full_init["Average"].iloc[5:] + ton_bofire_stats_full_init["Standard Deviation"].iloc[5:],
    alpha=0.2, color='#2ca02c'
)

# init
plt.plot(initial_points_yld_iterations, initial_points_ton["Average"], label="Initialised points", linestyle="--", marker="x", color = 'black')
#plt.fill_between(
    #initial_points_yld_iterations,
    #initial_points_yld["Average"].iloc[:5] - initial_points_yld["Standard Deviation"].iloc[:5],
    #initial_points_yld["Average"].iloc[:5] + initial_points_yld["Standard Deviation"].iloc[:5],
    #alpha=0.2, color='grey'
#)


#plt.title("Iteration vs Average Cumulative Max Yield (YLD) with Standard Deviation")
plt.xlabel("Iteration", fontsize=13)
plt.ylabel("Turnover number", fontsize=13)
plt.legend(loc="lower right")
plt.minorticks_on()
plt.grid(True)
plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   
plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.xticks(range(-5, int(max(yld_mobo_stats_full_init['Iteration'])) -4, 5))
plt.show()

'''

'''

#function to plot cumulative max values vs. iteration
def plot_cumulative_max(data_dict, title, ylabel, key):
    plt.figure(figsize=(10, 6))
    
    for trial, df in data_dict.items():
        
        df_filtered = df.iloc[5:]
        plt.plot(df_filtered['Iteration'], df_filtered[key], label=f'Trial {trial}')
    
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot for MOBO - Cumulative Max Yield
plot_cumulative_max(modified_measurements_dataframes_mobo_with_init, 
                    "BayBE - Cumulative Max Yield vs. Iteration", 
                    "Cumulative Max Yield", 
                    'Cumulative Max YLD')

# Plot for MOBO - Cumulative Max Ton
plot_cumulative_max(modified_measurements_dataframes_mobo_with_init, 
                    "BayBE - Cumulative Max Ton vs. Iteration", 
                    "Cumulative Max Ton", 
                    'Cumulative Max TON')

# Plot for BOFIRE - Cumulative Max Yield
plot_cumulative_max(modified_measurements_dataframes_bofire_with_init, 
                    "BOFIRE - Cumulative Max Yield vs. Iteration", 
                    "Cumulative Max Yield", 
                    'Cumulative Max YLD')

# Plot for BOFIRE - Cumulative Max Ton
plot_cumulative_max(modified_measurements_dataframes_bofire_with_init, 
                    "BOFIRE - Cumulative Max Ton vs. Iteration", 
                    "Cumulative Max Ton", 
                    'Cumulative Max TON')





# function to compute average iterations to convergence (i.e. before maximum value reached)
def compute_avg_convergence(data_dict, key):
    iterations_to_convergence = []

    for trial, df in data_dict.items():
        
        df_filtered = df.iloc[5:].reset_index(drop=True)
        
        
        max_value = df_filtered[key].max()
        
        
        first_occurrence = df_filtered[df_filtered[key] == max_value]['Iteration'].min()
        
        if not np.isnan(first_occurrence):
            iterations_to_convergence.append(first_occurrence)

    
    avg_iterations = np.mean(iterations_to_convergence) if iterations_to_convergence else None
    std_iterations = np.std(iterations_to_convergence)
    return avg_iterations, std_iterations


mobo_avg_iter_yld, std_baybe_yld = compute_avg_convergence(modified_measurements_dataframes_mobo_with_init, 'Cumulative Max YLD')
mobo_avg_iter_ton, std_baybe_ton  = compute_avg_convergence(modified_measurements_dataframes_mobo_with_init, 'Cumulative Max TON')

bofire_avg_iter_yld, std_bofire_yld = compute_avg_convergence(modified_measurements_dataframes_bofire_with_init, 'Cumulative Max YLD')
bofire_avg_iter_ton, std_bofire_ton = compute_avg_convergence(modified_measurements_dataframes_bofire_with_init, 'Cumulative Max TON')

print(f"BayBE - Avg Iterations to Reach Max Yield: {mobo_avg_iter_yld, std_baybe_yld}")
print(f"BayBE - Avg Iterations to Reach Max Ton: {mobo_avg_iter_ton, std_baybe_ton}")

print(f"BOFIRE - Avg Iterations to Reach Max Yield: {bofire_avg_iter_yld, std_bofire_yld}")
print(f"BOFIRE - Avg Iterations to Reach Max Ton: {bofire_avg_iter_ton, std_bofire_ton}")

def finding_maxima(dict, key):
    max_values = []
    for trial, df in dict.items():
        
        df_filtered = df.iloc[5:].reset_index(drop=True)
        max_value = df_filtered[key].max()
        max_values.append(max_value)
    

    mean_value = np.mean(max_values)


    std_dev_value = np.std(max_values)


    return max_values, mean_value, std_dev_value

max_values_yld_baybe, mean_value_yld_baybe, std_dev_value_yld_baybe = finding_maxima(modified_measurements_dataframes_mobo_with_init, 'Cumulative Max YLD')
max_values_ton_baybe, mean_value_ton_baybe, std_dev_value_ton_baybe = finding_maxima(modified_measurements_dataframes_mobo_with_init, 'Cumulative Max TON')
max_values_yld_bofire, mean_value_yld_bofire, std_dev_value_yld_bofire = finding_maxima(modified_measurements_dataframes_bofire_with_init, 'Cumulative Max YLD')
max_values_ton_bofire, mean_value_ton_bofire, std_dev_value_ton_bofire = finding_maxima(modified_measurements_dataframes_bofire_with_init, 'Cumulative Max TON')

print(f'Baybe average maximum yield {mean_value_yld_baybe} with std. {std_dev_value_yld_baybe}')
print(f'Baybe average maximum ton {mean_value_ton_baybe} with std. {std_dev_value_ton_baybe}')
print(f'BoFire average maximum yield {mean_value_yld_bofire} with std. {std_dev_value_yld_bofire}')
print(f'BoFire average maximum ton {mean_value_ton_bofire} with std. {std_dev_value_ton_bofire}')

print(f'Baybe maximum yield values: {max_values_yld_baybe} ')
print(f'Baybe maximum ton values: {max_values_ton_baybe} ')
print(f'BoFire maximum yield values: {max_values_yld_bofire} ')
print(f'BoFire maximum ton values: {max_values_ton_bofire} ')

'''
