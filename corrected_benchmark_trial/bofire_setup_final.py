
import bofire 
import botorch
import torch
import summit
import numpy as np
import pandas as pd
import time
import os
import pickle
from setup_file_final import evaluate_candidates
import multiprocessing
import importlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.exceptions
import bofire.strategies.api as strategies
from bofire.data_models.domain.api import Domain
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
    CategoricalInput,
    CategoricalMolecularInput,
    CategoricalDescriptorInput,)

from pprint import pprint as pp
from bofire.data_models.objectives.api import MinimizeObjective, MaximizeObjective
from bofire.data_models.strategies.api import MoboStrategy
from bofire.data_models.acquisition_functions.api import qEHVI, qLogEHVI
from bofire.data_models.strategies.api import SoboStrategy
from bofire.data_models.acquisition_functions.api import qEI
from bofire.data_models.strategies.api import (
    RandomStrategy as RandomStrategyModel,
)
import pathlib
from initialising_points_final import bofire_initial_conditions
#from benchmarking_alt import bofire_initial_conditions


def run_mobo_optimization(emulator: summit.benchmarks.experimental_emulator.ReizmanSuzukiEmulator, mobo_strategy,  bofire_initial_conditions, experimental_budget=5):
    """
    Runs a multi-objective Bayesian optimization loop.
    
    Parameters:
        mobo_strategy: The optimization strategy object with ask and tell methods.
        evaluate_candidates: Function to evaluate candidate experiments.
        bofire_initial_conditions: Initial conditions for optimization.
        experimental_budget: Number of iterations for the optimization loop.
        
    Returns:
        results_df: DataFrame containing experiment results and cumulative metrics.
        results: Final experiments stored in the strategy object.
    """


   
    temperature_feature = ContinuousInput(
    key="Temperature", bounds=[30.0, 110.0], unit="°C"
    )


    catalyst_loading_feature = ContinuousInput(
        key="Catalyst Loading", bounds=[0.5, 2.0], unit="%"
    )


    residence_time_feature = ContinuousInput(
        key="Residence Time", bounds=[1 * 60, 10 * 60], unit="minutes"
    )


    catalyst_feature = CategoricalMolecularInput(
    key="Catalyst",
    categories=list(available_catalysts.values())
    
)

    # gather all individual features
    input_features = Inputs(
        features=[
            temperature_feature,
            catalyst_loading_feature,
            residence_time_feature,
            catalyst_feature,
        ]
    )

    name_map = {
        "Catalyst Loading": "catalyst_loading",
        "Residence Time": "t_res",
        "Temperature": "temperature",
        "Catalyst": "catalyst",
        "Yield": "yld",
    }
    candidates = pd.DataFrame(
        {
            "Catalyst Loading": [],
            "Residence Time": [],
            "Temperature": [],
            "Catalyst": [
],
        }
    ).rename(columns=name_map)

    conditions = summit.DataSet.from_df(candidates)
    results = emulator.run_experiments(conditions, rtn_std=True).rename(
        columns=dict(zip(name_map.values(), name_map.keys())),
    )
    experiments = pd.DataFrame(
        {
            "Catalyst Loading": results["Catalyst Loading"],
            "Residence Time": results["Residence Time"],
            "Temperature": results["Temperature"],
            "Catalyst": results["Catalyst"],
            "Yield": results["Yield"],
            "valid_Yield": 1,
        }
    )


    max_objective = MaximizeObjective(w=1.0)

    max_objective_2 = MaximizeObjective(w=1.0, bounds=[0, 100])


    yield_feature = ContinuousOutput(key="Yield", objective=max_objective)
    ton_feature = ContinuousOutput(key="TON", objective=max_objective_2)
    
    output_features = Outputs(features=[yield_feature, ton_feature])
    domain = Domain(
        inputs=input_features,
        outputs=output_features,
    )
    

    #qExpectedImprovement = qEHVI()
    qLogExpectedImprovement = qLogEHVI()
    mobo_strategy_data_model = MoboStrategy(
        domain=domain,
        acquisition_function=qLogExpectedImprovement,
    )

    # map the strategy data model to the actual strategy that has functionality
    mobo_strategy = strategies.map(mobo_strategy_data_model)



    # initialize variables
    candidates = bofire_initial_conditions

    print('initial conditions', candidates)

    experiments = evaluate_candidates(candidates)
    
    mobo_strategy.tell(experiments, replace=True, retrain=True)

    results_df = pd.DataFrame(columns=["Iteration","Catalyst Loading", "Residence Time", "Temperature", 
                                       "Catalyst", "Yield", "valid_Yield", "TON", "valid_TON"])
    
    times_df_bofire = pd.DataFrame(columns=["Iteration", "Time_taken"])

    i = 0
    done = False

    while not done:
        print('Starting Bofire loop')
        i += 1
        t1 = time.time()

   
        new_candidate = mobo_strategy.ask(1)
        new_experiment = evaluate_candidates(new_candidate)
        
        
        mobo_strategy.tell(new_experiment)

        # add new results to dataframe
        results_df = pd.concat([results_df, new_experiment], ignore_index=True)

        # find cumulative max for Yield and TON
        results_df["Cumulative_Max_Yield"] = results_df["Yield"].cummax()
        results_df["Cumulative_Max_TON"] = results_df["TON"].cummax()
        results_df.loc[results_df["Iteration"].isna(), "Iteration"] = i

        time_taken = time.time() - t1
        print(f"Iteration {i} took {time_taken:.2f} seconds")

        times_df_bofire = pd.concat(
        [times_df_bofire, pd.DataFrame({"Iteration": [i], "Time_taken": [time_taken]})],
        ignore_index=True
    )

        print("Current Results:")
        print(results_df[["Yield", "Cumulative_Max_Yield", "TON", "Cumulative_Max_TON"]])

        if i > experimental_budget:
            done = True

    results = mobo_strategy.experiments
    print("Final experiments:")
    print(results)

    print("Results including cumulative max metrics:")
    print(results_df[["Yield", "Cumulative_Max_Yield", "TON", "Cumulative_Max_TON"]])

    return results_df, results, times_df_bofire
