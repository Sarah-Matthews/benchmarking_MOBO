from open_file import *


'''pareto_dataframes_sobo = {}
pareto_dataframes_mobo = {}
pareto_dataframes_bofire = {}

for entry in sobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Campaign Measurements']
    
    # Store the dataframe in the dictionary with trial as the key
    pareto_dataframes_sobo[trial] = dataframe

for entry in mobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Campaign Measurements']
    
    # Store the dataframe in the dictionary with trial as the key
    pareto_dataframes_mobo[trial] = dataframe


# Loop over bofire_data_list to store the required dataframe
for entry in bofire_data_list:
    trial = entry['Trial']
    dataframe = entry['Campaign Measurements']  # Ensure this is the correct dataframe
    
    # Check if 'Campaign Measurements' is correctly referenced as a dataframe
    if isinstance(dataframe, pd.DataFrame):
        print(f"Dataframe for Trial {trial}:\n{dataframe.head()}")
    else:
        print(f"Warning: Data for Trial {trial} is not a DataFrame!")

    # Store the dataframe under the trial key
    pareto_dataframes_bofire[trial] = dataframe

print(pareto_dataframes_mobo)
'''


cumulative_maxima_dataframes_mobo 
cumulative_maxima_dataframes_bofire