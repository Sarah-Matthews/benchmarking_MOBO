import pickle
import matplotlib.pyplot as plt
import pandas as pd


try:
    with open('benchmark_results.pkl', 'rb') as f:
        results_runs = pickle.load(f)
except FileNotFoundError:
    print("Benchmark results not found. Please run the benchmark first.")
    


sobo_data = {}
mobo_data = {}
bofire_data = {}

# Iterate through the models in results_runs
for model_name, model_results in results_runs.items():

    if "sobo" in model_name.lower():
        sobo_data[model_name] = model_results
    elif "mobo" in model_name.lower():
        mobo_data[model_name] = model_results
    else:
        bofire_data[model_name] = model_results


print("SOBO Results:")
print(sobo_data)
print("Length of sobo_data:", len(sobo_data))

print("MOBO Results:")
print(mobo_data)

print("Bofire MOBO Results:")
print(bofire_data)

key = list(sobo_data.keys())[0]  # Access the first key
value = sobo_data[key]
x_df = value[0]
measurements_sobo = x_df[0]
cumulative_yld_sobo = x_df[2]
iteration = cumulative_yld_sobo['Iteration']
yld = cumulative_yld_sobo['Cumulative Max YLD']


key = list(mobo_data.keys())[0]  # Access the first key
value = mobo_data[key]
x_df = value[0]
measurements_mobo = x_df[0]

cumulative_max_df_mobo = x_df[3]
print('max:', cumulative_max_df_mobo)

iteration_mobo = cumulative_max_df_mobo['Iteration']
yld_mobo = cumulative_max_df_mobo['Cumulative Max YLD']
ton_mobo = cumulative_max_df_mobo['Cumulative Max TON']

key = list(bofire_data.keys())[0]  # Access the first key
value = bofire_data[key]
x_df = value[0]
measurements_bofire = x_df[0]
print("1st",measurements_bofire)

df_bofire = x_df[0]
print("bofire results incl. cumulative maxima", df_bofire)
full_df_bofire = x_df[1]
print('full bofire results (incl. random trials):', full_df_bofire)

#iteration_mobo = cumulative_max_df_mobo['Iteration']
#yld_mobo = cumulative_max_df_mobo['Cumulative Max YLD']
#ton_mobo = cumulative_max_df_mobo['Cumulative Max TON']

# Plot for SOBO: Cumulative YLD vs Iteration
plt.figure(figsize=(10, 6))
plt.plot(iteration, yld, label='SOBO Cumulative YLD', color='b')
plt.xlabel('Iteration')
plt.ylabel('Cumulative YLD')
plt.title('SOBO: Cumulative YLD vs Iteration')
plt.legend()
plt.grid(True)
plt.show()

# Create a plot
plt.figure(figsize=(8, 6))
# Plot Cumulative Max YLD
plt.plot(iteration_mobo, yld_mobo, marker='o', linestyle='-', color='b', label='Cumulative Max YLD')

# Plot Cumulative Max TON
plt.plot(iteration_mobo, ton_mobo, marker='s', linestyle='-', color='r', label='Cumulative Max TON')

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Iteration vs Cumulative Max YLD and Cumulative Max TON')

# Add a legend to differentiate the two lines
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

iteration_bofire = df_bofire['Iteration']
yld_bofire = df_bofire['Cumulative_Max_Yield']
ton_bofire = df_bofire['Cumulative_Max_TON']
print('iteration',iteration_bofire)
print('yld bofire', yld_bofire)
print('ton bpfire', ton_bofire)

# Create a plot
plt.figure(figsize=(8, 6))
# Plot Cumulative Max YLD
plt.plot(iteration_bofire, yld_bofire, marker='o', linestyle='-', color='b', label='Cumulative Max YLD')

# Plot Cumulative Max TON
plt.plot(iteration_bofire, ton_bofire, marker='s', linestyle='-', color='r', label='Cumulative Max TON')

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Iteration vs Cumulative Max YLD and Cumulative Max TON (Bofire)')

# Add a legend to differentiate the two lines
plt.legend()

# Show the plot
plt.grid(True)
plt.show()