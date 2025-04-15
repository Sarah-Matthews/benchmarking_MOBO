from setup_file_final import *
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


iterations = 35

trials = 15


results_dict_random = {}

for trial in range(1, trials + 1):
    
    results_df = pd.DataFrame(columns=["Iteration", "Yield", "TON", "Cumulative_Max_Yield", "Cumulative_Max_TON"])
    
    for i in range(0, iterations):
        
        random_experiment = random_point(domain=domain_bofire)
        results_df = pd.concat([results_df, random_experiment], ignore_index=True)
        
        
        results_df.loc[results_df.index[-1], "Iteration"] = i + 1
        
        
        results_df["Cumulative_Max_Yield"] = results_df["Yield"].cummax()
        results_df["Cumulative_Max_TON"] = results_df["TON"].cummax()
    
    
    results_dict_random[trial] = results_df
    pickle_file_path = "results_dict_random.pkl"


    with open(pickle_file_path, "wb") as pkl_file:
        pickle.dump(results_dict_random, pkl_file)


for trial, df in results_dict_random.items():
    print(f"\nTrial {trial} Results:")
    print(df[["Iteration", "Yield", "Cumulative_Max_Yield", "TON", "Cumulative_Max_TON"]])
