import pickle
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'benchmark_results_multi_trial_long_trial.pkl'

try:
    with open(file_path, 'rb') as f:
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

    sobo_data_list = []
    mobo_data_list = []
    bofire_data_list=[]

    # Iterate through the rows and filter based on model names and trial numbers
    for _, row in results_runs.iterrows():
        model_name = row['Models']  # Assuming 'Models' column contains the model names
        trial = row['Trials']
        results = row['Results']

        

        # Assuming results is a tuple (campaign_measurements, dataframe_str)
        campaign_measurements = results[0]
        dataframe_str = results[1]  # This may need further parsing if the dataframe is serialized

        
        # Check if dataframe_str is already a DataFrame
        if isinstance(dataframe_str, pd.DataFrame):
            dataframe = dataframe_str
            #print(f"Dataframe: {dataframe.head()}")  # Print first few rows to verify
        else:
            # If it's not a DataFrame, try to convert it (e.g., from a string or another format)
            try:
                dataframe = pd.read_csv(pd.compat.StringIO(dataframe_str))  # or other formats
            except Exception as e:
                print(f"Error converting dataframe: {e}")
                dataframe = None

        
        
        # Ensure each trial is added under the respective model's dictionary
        if 'sobo' in model_name.lower():
            if trial not in sobo_data:
                sobo_data[trial] = {}
            sobo_data[trial][model_name] = {'campaign_measurements': campaign_measurements, 'dataframe': dataframe}
        elif 'mobo' in model_name.lower():
            if trial not in mobo_data:
                mobo_data[trial] = {}
            mobo_data[trial][model_name] = {'campaign_measurements': campaign_measurements, 'dataframe': dataframe}
        elif 'bofire' in model_name.lower():
            if trial not in bofire_data:
                bofire_data[trial] = {}
            bofire_data[trial][model_name] = {'campaign_measurements': campaign_measurements, 'dataframe': dataframe}

    # Output the separate data dictionaries for each model type and trial
    print("SOBO Data:")
    for trial, models in sobo_data.items():
        print(f"Trial {trial}:")
        for model, data in models.items():
            print(f"  {model}:")
            print(f"    Campaign Measurements: {data['campaign_measurements']}")
            print(f"    Dataframe: {data['dataframe']}")

            sobo_data_list.append({
            'Trial': trial,
            'Model': model,
            'Campaign Measurements': data['campaign_measurements'],
            'Cumulative Maxima Dataframe': data['dataframe']
        })

    print("\nMOBO Data:")
    for trial, models in mobo_data.items():
        print(f"Trial {trial}:")
        for model, data in models.items():
            print(f"  {model}:")
            print(f"    Campaign Measurements: {data['campaign_measurements']}")
            print(f"    Dataframe: {data['dataframe']}")

            mobo_data_list.append({
            'Trial': trial,
            'Model': model,
            'Campaign Measurements': data['campaign_measurements'],
            'Cumulative Maxima Dataframe': data['dataframe']
        })

    print("\nBOFIRE Data:")
    for trial, models in bofire_data.items():
        print(f"Trial {trial}:")
        for model, data in models.items():
            print(f"  {model}:")
            print(f"    Campaign Measurements: {data['dataframe']}")
            print(f"    Dataframe: {data['campaign_measurements']}")

            bofire_data_list.append({
            'Trial': trial,
            'Model': model,
            'Campaign Measurements': data['dataframe'],
            'Cumulative Maxima Dataframe': data['campaign_measurements']
        })





#extracting the cumulative maximum dataframes for each trial to plot

cumulative_maxima_dataframes_sobo = {}
cumulative_maxima_dataframes_mobo = {}
cumulative_maxima_dataframes_bofire = {}

for entry in sobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Cumulative Maxima Dataframe']
    
    # Store the dataframe in the dictionary with trial as the key
    cumulative_maxima_dataframes_sobo[trial] = dataframe

for entry in mobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Cumulative Maxima Dataframe']
    
    # Store the dataframe in the dictionary with trial as the key
    cumulative_maxima_dataframes_mobo[trial] = dataframe


# Loop over bofire_data_list to store the required dataframe
for entry in bofire_data_list:
    trial = entry['Trial']
    dataframe = entry['Cumulative Maxima Dataframe']  # Ensure this is the correct dataframe
    
    # Check if 'Campaign Measurements' is correctly referenced as a dataframe
    if isinstance(dataframe, pd.DataFrame):
        print(f"Dataframe for Trial {trial}:\n{dataframe.head()}")
    else:
        print(f"Warning: Data for Trial {trial} is not a DataFrame!")

    # Store the dataframe under the trial key
    cumulative_maxima_dataframes_bofire[trial] = dataframe

print('bofire_max',cumulative_maxima_dataframes_bofire)