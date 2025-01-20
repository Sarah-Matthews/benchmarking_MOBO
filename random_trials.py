from setup_files_alt import *
from sklearn.preprocessing import MinMaxScaler
import pickle

'''Using the random strategy model from bofire to initialise points for all 3 models!'''

def random_point(domain):
    random_strategy_model = RandomStrategyModel(domain=domain)
    random_strategy = strategies.map(random_strategy_model)
    candidates = random_strategy.ask(1)

    results = evaluate_candidates(candidates)

    return results
'''

iterations = 30

results_df = pd.DataFrame(columns=["Iteration","Catalyst Loading", "Residence Time", "Temperature", 
                                       "Catalyst", "Yield", "valid_Yield", "TON", "valid_TON"])

for i in range(0, iterations):

    random_experiment = random_point(domain=domain_bofire)
    results_df = pd.concat([results_df, random_experiment], ignore_index=True)
    results_df.loc[results_df.index[-1], "Iteration"] = i+1
    results_df["Cumulative_Max_Yield"] = results_df["Yield"].cummax()
    results_df["Cumulative_Max_TON"] = results_df["TON"].cummax()

print(results_df[["Iteration","Yield", "Cumulative_Max_Yield", "TON", "Cumulative_Max_TON"]])

'''

# Number of iterations per trial
iterations = 30
# Number of trials
trials = 30

# Dictionary to store results for each trial
results_dict_random = {}

for trial in range(1, trials + 1):
    # Create a new DataFrame for each trial
    results_df = pd.DataFrame(columns=["Iteration", "Yield", "TON", "Cumulative_Max_Yield", "Cumulative_Max_TON"])
    
    for i in range(0, iterations):
        # Simulate a random experiment
        random_experiment = random_point(domain=domain_bofire)
        results_df = pd.concat([results_df, random_experiment], ignore_index=True)
        
        # Update the Iteration column for the most recent row
        results_df.loc[results_df.index[-1], "Iteration"] = i + 1
        
        # Update cumulative metrics
        results_df["Cumulative_Max_Yield"] = results_df["Yield"].cummax()
        results_df["Cumulative_Max_TON"] = results_df["TON"].cummax()
    
    # Store the DataFrame in the dictionary with the trial number as the key
    results_dict_random[trial] = results_df
    pickle_file_path = "results_dict_random.pkl"

# Save the dictionary to the pickle file
    with open(pickle_file_path, "wb") as pkl_file:
        pickle.dump(results_dict_random, pkl_file)

# Print results for each trial
for trial, df in results_dict_random.items():
    print(f"\nTrial {trial} Results:")
    print(df[["Iteration", "Yield", "Cumulative_Max_Yield", "TON", "Cumulative_Max_TON"]])
