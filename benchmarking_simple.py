from bofire_setup import run_mobo_optimization
from baybe_models import Models
from setup_files_alt import *
from initialising_points import *
from setup_files_alt import *


print(bofire_initial_conditions)
print(initial_conditions)

iterations = 5
emulator = get_pretrained_reizman_suzuki_emulator(case=1)

def run_optimization_workflow(iterations, 
                              bofire_initial_conditions, 
                              emulator, 
                              campaign_sobo, 
                              campaign_mobo,
                              initial_conditions_df):
    """
    Runs three optimization functions in sequence for a specified number of iterations.
    
    Parameters:
        iterations (int): Number of iterations to run each function.
        bofire_initial_conditions: Initial conditions for `run_mobo_optimization`.
        emulator: Emulator object for `run_sobo_loop` and `run_mobo_loop`.
        campaign: Campaign object for `run_sobo_loop` and `run_mobo_loop`.
        initial_conditions_df: Initial conditions DataFrame for `run_sobo_loop` and `run_mobo_loop`.
    """
    # Step 1: Run MOBO Optimization
    #run_mobo_optimization(emulator=emulator, mobo_strategy=mobo_strategy_bofire,bofire_initial_conditions=bofire_initial_conditions, experimental_budget=iterations)
    
    # Step 2: Run SOBO Loop
    Models.run_sobo_loop(
        emulator=emulator, 
        campaign=campaign_sobo, 
        iterations=iterations, 
        initial_conditions_df=initial_conditions_df
    )
    
    # Step 3: Run MOBO Loop
    Models.run_mobo_loop(
        emulator=emulator, 
        campaign=campaign_mobo, 
        iterations=iterations, 
        initial_conditions_df=initial_conditions_df
    )

results = run_optimization_workflow(iterations=5, bofire_initial_conditions=bofire_initial_conditions, emulator=emulator, campaign_sobo=campaign_sobo, campaign_mobo=campaign_mobo, initial_conditions_df=initial_conditions)
