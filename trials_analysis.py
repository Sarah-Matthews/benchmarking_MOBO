import pickle
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import pickle

import pandas as pd
import pickle

# Load the data
try:
    with open('benchmark_results_multi_trial.pkl', 'rb') as f:
        results_runs = pickle.load(f)
except FileNotFoundError:
    print("Benchmark results not found. Please run the benchmark first.")
    results_runs = pd.DataFrame()  # Fallback to an empty DataFrame

if results_runs.empty:
    print("No data found in the benchmark file.")
else:
    # Check unique trials in the data
    print(f"Unique trials: {results_runs['Trials'].unique()}")
    
    # Separate data for each model type and trial
    sobo_data = {}
    mobo_data = {}
    bofire_data = {}

    # Iterate through the rows and filter based on model names and trial numbers
    for _, row in results_runs.iterrows():
        model_name = row['Models']  # Assuming 'Models' column contains the model names
        trial = row['Trials']
        results = row['Results']

        # Ensure each trial is added under the respective model's dictionary
        if 'sobo' in model_name.lower():
            if trial not in sobo_data:
                sobo_data[trial] = {}
            sobo_data[trial][model_name] = results
        elif 'mobo' in model_name.lower():
            if trial not in mobo_data:
                mobo_data[trial] = {}
            mobo_data[trial][model_name] = results
        elif 'bofire' in model_name.lower():
            if trial not in bofire_data:
                bofire_data[trial] = {}
            bofire_data[trial][model_name] = results

    # Output the separate data dictionaries for each model type and trial
    print("SOBO Data:")
    for trial, models in sobo_data.items():
        print(f"Trial {trial}:")
        for model, result in models.items():
            print(f"  {model}: {result}")

    print("\nMOBO Data:")
    for trial, models in mobo_data.items():
        print(f"Trial {trial}:")
        for model, result in models.items():
            print(f"  {model}: {result}")

    print("\nBOFIRE Data:")
    for trial, models in bofire_data.items():
        print(f"Trial {trial}:")
        for model, result in models.items():
            print(f"  {model}: {result}")
