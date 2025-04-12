import pickle
import pandas as pd
import matplotlib.pyplot as plt


#file loads the benchmarkiing results and separates out by model & trial

try:
    with open('benchmark_results_multi_trial.pkl', 'rb') as f:
        results_runs = pickle.load(f)
except FileNotFoundError:
    print("Benchmark results not found. Please run the benchmark first.")
    results_runs = pd.DataFrame()  # Fallback to an empty DataFrame

if results_runs.empty:
    print("No data found in the benchmark file.")
else:
    
    print(f"Unique trials: {results_runs['Trials'].unique()}")
    
    
    sobo_data = {}
    mobo_data = {}
    bofire_data = {}

    

    sobo_data_list = []
    mobo_data_list = []
    bofire_data_list=[]

    
    for _, row in results_runs.iterrows():
        model_name = row['Models']  
        trial = row['Trials']
        results = row['Results']

        

        
        campaign_measurements = results[0]
        dataframe_str = results[1]  
        times_df = results[2]
        
       
        if isinstance(dataframe_str, pd.DataFrame):
            dataframe = dataframe_str
            
        else:
           
            try:
                dataframe = pd.read_csv(pd.compat.StringIO(dataframe_str))  # or other formats
            except Exception as e:
                print(f"Error converting dataframe: {e}")
                dataframe = None

        
        
        
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

    #output the separate data dictionaries for each model type and trial
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





#extracting the cumulative maximum dataframes for each trial to plot

cumulative_maxima_dataframes_sobo = {}
cumulative_maxima_dataframes_mobo = {}
cumulative_maxima_dataframes_bofire = {}

for entry in sobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Cumulative Maxima Dataframe']
    
    
    cumulative_maxima_dataframes_sobo[trial] = dataframe

for entry in mobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Cumulative Maxima Dataframe']
    
    
    cumulative_maxima_dataframes_mobo[trial] = dataframe



for entry in bofire_data_list:
    trial = entry['Trial']
    dataframe = entry['Cumulative Maxima Dataframe'] 
    
    
    if isinstance(dataframe, pd.DataFrame):
        print(f"Dataframe for Trial {trial}:\n{dataframe.head()}")
    else:
        print(f"Warning: Data for Trial {trial} is not a DataFrame!")

    
    cumulative_maxima_dataframes_bofire[trial] = dataframe

print('bofire_max',cumulative_maxima_dataframes_bofire)



#extracting the cumulative maximum dataframes for each trial to plot

times_dataframes_sobo = {}
times_dataframes_mobo = {}
times_dataframes_bofire = {}

for entry in sobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Times Dataframe']
    
    
    times_dataframes_sobo[trial] = dataframe

for entry in mobo_data_list:
    trial = entry['Trial']
    dataframe = entry['Times Dataframe']
    
    
    times_dataframes_mobo[trial] = dataframe



for entry in bofire_data_list:
    trial = entry['Trial']
    dataframe = entry['Times Dataframe']  
    
    
    if isinstance(dataframe, pd.DataFrame):
        print(f"Dataframe for Trial {trial}:\n{dataframe.head()}")
    else:
        print(f"Warning: Data for Trial {trial} is not a DataFrame!")

    
    times_dataframes_bofire[trial] = dataframe

print('sobo_iter times:',times_dataframes_sobo)
print('mobo_iter times:',times_dataframes_mobo)
print('bofire_iter times:',times_dataframes_bofire)