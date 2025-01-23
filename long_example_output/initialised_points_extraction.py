from open_file import measurements_dataframes_bofire, measurements_dataframes_sobo, measurements_dataframes_mobo

#modify the value in df.head(x) depending on how many random trials the experiments were initialised with

initilialised_points_sobo = {}
initilialised_points_mobo = {}
initilialised_points_bofire = {}

# Loop through each trial in the measurements dictionary
for trial, df in measurements_dataframes_sobo.items():
    # Extract the first x rows for the current trial
    initilialised_points_sobo[trial] = df.head(10)  

print('first trial sobo:', initilialised_points_sobo.get(1))

for trial, df in measurements_dataframes_mobo.items():
    # Extract the first x rows for the current trial
    initilialised_points_mobo[trial] = df.head(10)  

print('first trial mobo:', initilialised_points_mobo.get(1))

for trial, df in measurements_dataframes_bofire.items():
    # Extract the first x rows for the current trial
    initilialised_points_bofire[trial] = df.head(10)  

print('first trial bofire:', initilialised_points_bofire.get(1))

