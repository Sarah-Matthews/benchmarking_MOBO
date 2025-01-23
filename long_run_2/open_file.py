import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    with open('long_trial_results.pkl', 'rb') as f:
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
        times_df = results[2]
        
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
            sobo_data[trial][model_name] = {'campaign_measurements': campaign_measurements, 'dataframe': dataframe, 'times': times_df}
        elif 'mobo' in model_name.lower():
            if trial not in mobo_data:
                mobo_data[trial] = {}
            mobo_data[trial][model_name] = {'campaign_measurements': campaign_measurements, 'dataframe': dataframe, 'times': times_df}
        elif 'bofire' in model_name.lower():
            if trial not in bofire_data:
                bofire_data[trial] = {}
            bofire_data[trial][model_name] = {'campaign_measurements': campaign_measurements, 'dataframe': dataframe, 'times': times_df}

    # Output the separate data dictionaries for each model type and trial
    print("SOBO Data:")
    
    for trial, models in sobo_data.items():
        print(f"Trial {trial}:")
        for model, data in models.items():
            print(f"  {model}:")
            print(f"    Campaign Measurements: {data['campaign_measurements']}")
            print(f"    Dataframe: {data['dataframe']}")
            print(f"    times Dataframe: {data['times']}")

            sobo_data_list.append({
            'Trial': trial,
            'Model': model,
            'Campaign Measurements': data['campaign_measurements'],
            'Cumulative Maxima Dataframe': data['dataframe'],
            'Times Dataframe': data['times']
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
            'Cumulative Maxima Dataframe': data['dataframe'],
            'Times Dataframe': data['times']
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
            'Cumulative Maxima Dataframe': data['campaign_measurements'],
            'Times Dataframe': data['times']
        })


#extracting the measurements dataframes for each trial to plot

measurements_dataframes_sobo = {}
measurements_dataframes_mobo = {}
measurements_dataframes_bofire = {}

for entry in sobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Campaign Measurements']
    
    # Store the dataframe in the dictionary with trial as the key
    measurements_dataframes_sobo[trial] = dataframe

for entry in mobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Campaign Measurements']
    
    # Store the dataframe in the dictionary with trial as the key
    measurements_dataframes_mobo[trial] = dataframe


# Loop over bofire_data_list to store the required dataframe
for entry in bofire_data_list:
    trial = entry['Trial']
    dataframe = entry['Campaign Measurements']  # Ensure this is the correct dataframe
    
    measurements_dataframes_bofire[trial] = dataframe

print(measurements_dataframes_bofire)

modified_measurements_dataframes_mobo = {}

for trial, dataframe in measurements_dataframes_mobo.items():
    # Create a new DataFrame with the required columns
    modified_dataframe = pd.DataFrame({
        'Iteration': range(1, len(dataframe) + 1),  
        'Yield': dataframe['yld'].values,          
        'Ton': dataframe['ton'].values            
    })
    
    # Add the modified DataFrame to the new dictionary
    modified_measurements_dataframes_mobo[trial] = modified_dataframe



modified_measurements_dataframes_sobo = {}

for trial, dataframe in measurements_dataframes_sobo.items():
    # Create a new DataFrame with the required columns
    modified_dataframe = pd.DataFrame({
        'Iteration': range(1, len(dataframe) + 1),  
        'Yield': dataframe['yld'].values,                   
    })
    
    # Add the modified DataFrame to the new dictionary
    modified_measurements_dataframes_sobo[trial] = modified_dataframe

#print(measurements_dataframes_bofire)
modified_measurements_dataframes_bofire = {}

for trial, dataframe in measurements_dataframes_bofire.items():
    # Create a new DataFrame with the required columns
    modified_dataframe = pd.DataFrame({
        'Iteration': range(1, len(dataframe) + 1),  
        'Yield': dataframe['Yield'].values,          
        'Ton': dataframe['TON'].values            
    })
    
    # Add the modified DataFrame to the new dictionary
    modified_measurements_dataframes_bofire[trial] = modified_dataframe
#ßprint(modified_measurements_dataframes_bofire)

'''Extracting yield and ton values into a 2d array where each row is a different repeat (note this will 
include the initialisation pts for bofire)'''
yld_values_list_mobo = []  
ton_values_list_mobo = []  
# Loop through each entry in the dictionary
for trial, dataframe in measurements_dataframes_mobo.items():
    # Extract yield and ton (TON) values
    yld_values_mobo = dataframe['yld'].values 
    ton_values_mobo = dataframe['ton'].values  
    
    
    # Append the values for this trial
    yld_values_list_mobo.append(yld_values_mobo)
    ton_values_list_mobo.append(ton_values_mobo)

# Convert lists of arrays into 2D numpy arrays
yld_values_array_mobo = np.array(yld_values_list_mobo)  
ton_values_array_mobo = np.array(ton_values_list_mobo)  


print("Yield Values Array (shape:", yld_values_array_mobo.shape, "):")
#print(yld_values_array_mobo)

print("\nTon Values Array (shape:", ton_values_array_mobo.shape, "):")
#print(ton_values_array_mobo)



yld_values_list_bofire = []  
ton_values_list_bofire = []  
# Loop through each entry in the dictionary
for trial, dataframe in measurements_dataframes_bofire.items():
    # Extract yield and ton (TON) values
    yld_values_bofire = dataframe['Yield'].values 
    ton_values_bofire = dataframe['TON'].values  
    
    # Append the values for this trial
    yld_values_list_bofire.append(yld_values_bofire)
    ton_values_list_bofire.append(ton_values_bofire)

# Convert lists of arrays into 2D numpy arrays
yld_values_array_bofire = np.array(yld_values_list_bofire)  
ton_values_array_bofire = np.array(ton_values_list_bofire)  


print("Yield Values Array (shape:", yld_values_array_bofire.shape, "):")
#print(yld_values_array_bofire)

print("\nTon Values Array (shape:", ton_values_array_bofire.shape, "):")
#print(ton_values_array_bofire)




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
    #if isinstance(dataframe, pd.DataFrame):
        #print(f"Dataframe for Trial {trial}:\n{dataframe.head()}")
    #else:
        #print(f"Warning: Data for Trial {trial} is not a DataFrame!")

    # Store the dataframe under the trial key
    cumulative_maxima_dataframes_bofire[trial] = dataframe

#print('bofire_max',cumulative_maxima_dataframes_bofire)



#extracting the cumulative maximum dataframes for each trial to plot

times_dataframes_sobo = {}
times_dataframes_mobo = {}
times_dataframes_bofire = {}

for entry in sobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Times Dataframe']
    
    # Store the dataframe in the dictionary with trial as the key
    times_dataframes_sobo[trial] = dataframe

for entry in mobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Times Dataframe']
    
    # Store the dataframe in the dictionary with trial as the key
    times_dataframes_mobo[trial] = dataframe


# Loop over bofire_data_list to store the required dataframe
for entry in bofire_data_list:
    trial = entry['Trial']
    dataframe = entry['Times Dataframe']  # Ensure this is the correct dataframe
    
    # Check if 'Campaign Measurements' is correctly referenced as a dataframe
    #if isinstance(dataframe, pd.DataFrame):
        #print(f"Dataframe for Trial {trial}:\n{dataframe.head()}")
    #else:
        #print(f"Warning: Data for Trial {trial} is not a DataFrame!")

    # Store the dataframe under the trial key
    times_dataframes_bofire[trial] = dataframe

#print('sobo_iter times:',times_dataframes_sobo)
#print('mobo_iter times:',times_dataframes_mobo)
#print('bofire_iter times:',times_dataframes_bofire)