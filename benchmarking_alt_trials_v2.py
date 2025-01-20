#from benchmarking_files_v2.setup_files_alt import *
#from benchmarking_files_v2.initialising_points import *
#from benchmarking_files_v2.baybe_models import *
from setup_files_alt import *
from initialising_points import *
from baybe_models import *
from bofire_setup import run_mobo_optimization

def run_benchmark_baybe(num_trials=5):

    """
    Main function to run the benchmark for SOBO and MOBO models.
    """
    import copy

    # Define models and their campaigns
    models_to_run = {
        "SOBO": {
            "model_fn": Models.run_sobo_loop,
            "campaign": campaign_sobo,
            "use_campaign": True,
        },

        "MOBO": {
            "model_fn": Models.run_mobo_loop,
            "campaign": campaign_mobo,
            "use_campaign": True,
        },
         #"Bofire": {
            #"model_fn": run_mobo_optimization,  
            #"mobo_strategy": mobo_strategy_bofire,  
            #"use_campaign": False,  
        #},
    }

    #initial_conditions = df_random_initialised_point_renamed
    #bofire_initial_conditions = random_initialised_points

    emulator = get_pretrained_reizman_suzuki_emulator(case=1)

    #no. iterations for each optimisation run
    iterations = 5

    #df to hold full data from several runs
    all_results = []
    all_results_new = {
        "Trials": [],
        "Models": [],
        "Results": [],
    }
    for trial in range(1, num_trials+1):
        print(f"Starting trial {trial}/{num_trials}")

        bofire_initial_conditions = initialise_random_point(domain=domain_bofire)
        df_random_initialised_point = pd.DataFrame(bofire_initial_conditions)
        initial_conditions = df_random_initialised_point.rename(columns=name_map)

        trial_results = {model_name: [] for model_name in models_to_run.keys()}


        for model_name, model_config in models_to_run.items():
            print(f"Running {model_name} for Trial {trial}")
            if model_name=="Bofire":
                results = run_mobo_optimization(emulator = emulator,mobo_strategy = mobo_strategy_bofire,  bofire_initial_conditions = bofire_initial_conditions, experimental_budget=iterations-1)
            else:
                campaign = copy.deepcopy(model_config["campaign"])
                results = model_config["model_fn"](emulator, campaign, iterations, initial_conditions)

            '''
            for result in results:  # Assuming `results` is iterable
                #result["Trial"] = trial
                #result["Model"] = model_name
                all_results.append(result)
            '''

            all_results_new["Trials"].append(trial)
            all_results_new["Models"].append(model_name)
            all_results_new["Results"].append(results)
        
    results_df = pd.DataFrame(all_results_new)

    with open('benchmark_results_multi_trial.pkl', 'wb') as f:
        pickle.dump(results_df, f)
    return results_df
    
    
'''
#results_runs_baybe = run_benchmark_baybe()
#bofire_results_exp = bofire_mobo_loop(emperimental_budget=10, initial_conditions=bofire_initial_conditions, mobo_strategy_data_model=mobo_strategy_bofire)
#print(bofire_results_exp)
bofire_results_exp =bofire_mobo_loop(experimental_budget=10, initial_conditions=bofire_initial_conditions,mobo_strategy_data_model=mobo_strategy_bofire)
print(bofire_results_exp)
'''


#run = run_mobo_optimization(mobo_strategy = mobo_strategy_bofire,  bofire_initial_conditions = bofire_initial_conditions, experimental_budget=5)


results_runs_baybe = run_benchmark_baybe(num_trials=2)