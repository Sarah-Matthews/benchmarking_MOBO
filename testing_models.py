

import json
import summit
from summit.benchmarks import get_pretrained_reizman_suzuki_emulator
from summit.benchmarks.experimental_emulator import ReizmanSuzukiEmulator 
from summit.utils.dataset import DataSet
import pandas as pd
import numpy as np
import os
import pathlib

from baybe.acquisition.acqfs import ExpectedImprovement, qLogExpectedImprovement, ProbabilityOfImprovement
import matplotlib.pyplot as plt
from baybe.parameters import CategoricalParameter, NumericalContinuousParameter, SubstanceParameter, NumericalDiscreteParameter
from baybe.targets import NumericalTarget, TargetTransformation
from baybe.recommenders import RandomRecommender, SequentialGreedyRecommender, TwoPhaseMetaRecommender, BotorchRecommender
from baybe.surrogates import GaussianProcessSurrogate
from baybe import Campaign
from baybe import objectives
from baybe.objectives import SingleTargetObjective, DesirabilityObjective, base
from baybe.objectives.base import Objective
#from baybe.acquisition import ExpectedImprovement # see others at https://emdgroup.github.io/baybe/_autosummary/baybe.acquisition.acqfs.html#module-baybe.acquisition.acqfs
#from baybe.acquisition import debotorchize
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
#from baybe.acquisition.acqfs import ExpectedImprovement
from botorch import acquisition
import torch
import json
import random
from botorch import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from baybe.acquisition.acqfs import qLogExpectedImprovement, ProbabilityOfImprovement

#Baybe set-up

emulator = get_pretrained_reizman_suzuki_emulator(case=1)

catalyst_smiles = pd.read_csv(pathlib.Path.cwd() / pathlib.Path("suzuki_miyaura_catalysts.csv"))
ligand_smiles = pd.read_csv(pathlib.Path.cwd() / pathlib.Path("suzuki_miyaura_ligands.csv"))

name_map = {
    "Catalyst Loading": "catalyst_loading",
    "Residence Time": "t_res",
    "Temperature": "temperature",
    "Catalyst": "catalyst",
    "Yield": "yld",
}
candidates = pd.DataFrame(
    {
        "Catalyst Loading": [] ,
        "Residence Time": [],
        "Temperature": [],
        "Catalyst": [],
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

#Transforming categorical variable (catalyst) into chemical variable = need smiles representation for each catalyst



available_catalysts = {
    "P1-L1": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L1']['smiles'].values[0]}",
    "P1-L2": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L2']['smiles'].values[0]}",
    "P1-L3": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L3']['smiles'].values[0]}",
    "P1-L4": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L4']['smiles'].values[0]}",
    "P1-L5": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L5']['smiles'].values[0]}",
    "P1-L6": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L6']['smiles'].values[0]}",
    "P1-L7": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L7']['smiles'].values[0]}",
    "P2-L1": f"{catalyst_smiles[catalyst_smiles['name'] == 'P2']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L1']['smiles'].values[0]}",
}

#defining parameter space
parameters = [
    SubstanceParameter(
        name="catalyst",
        data=available_catalysts,
        encoding="MORDRED"
    ),
    NumericalContinuousParameter(
        name="catalyst_loading",
        bounds=(0.5,2.0),
    ),
    NumericalContinuousParameter(
        name="temperature",
        bounds=(30,110),
    ),
    NumericalContinuousParameter(
        name="t_res",
        bounds=(60,600),
    )
]


#defining search space
searchspace = SearchSpace.from_product(parameters)

target_1 = NumericalTarget(name="yld", mode=f"MAX", bounds=(0,100), transformation="LINEAR")

objective = SingleTargetObjective(target=target_1)
recommender = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=BotorchRecommender(
            sequential_continuous=True,
            surrogate_model=GaussianProcessSurrogate(),
            acquisition_function = qLogExpectedImprovement() ,
            allow_repeated_recommendations=False,
            allow_recommending_already_measured=False,
        )
    )

campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=recommender,
)


def perform_df_experiment(data_df: pd.DataFrame, emulator: ReizmanSuzukiEmulator, objective) -> dict:
    conditions = DataSet.from_df(data_df)
    #print(conditions)
    
    emulator_output = emulator.run_experiments(conditions, return_std=True)
    
    result_df = data_df.copy()

    for target in objective.targets:
        target_name = target.name  # Get the name of the target 
        
        # Find the column corresponding to the target_name in the emulator_output
        if target_name in emulator_output.columns:
            target_value = emulator_output[target_name].values[0]  
            result_df[target_name] = target_value # Add the target to the result DataFrame
        else:
            raise ValueError(f"Target column '{target_name}' not found in emulator output.")


    return result_df

#this function should be general (i.e. mobo & sobo) due to the dynamic target value extraction
def perform_df_experiment_multi(data_df: pd.DataFrame, emulator: ReizmanSuzukiEmulator, objective) -> dict:
    conditions = DataSet.from_df(data_df)
    #print(conditions)
    
    #emulator = get_pretrained_reizman_suzuki_emulator(case=1)
    emulator_output = emulator.run_experiments(conditions, return_std=True)
    
    result_df = data_df.copy()

    for target in objective.targets:
        target_name = target.name  # Get the name of the target 
        
        # Find the column corresponding to the target_name in the emulator_output
        if target_name in emulator_output.columns:
            target_values = emulator_output[target_name].values 
            
            #result_df[target_name] = emulator_output[target_name].values # Add the target to the result DataFrame
            result_df[target_name] = pd.to_numeric(target_values, errors='coerce')
        else:
            raise ValueError(f"Target column '{target_name}' not found in emulator output.")

    #print(result_df)

    return result_df


sample_dataset = {
        "catalyst": ["P1-L3"], "t_res": [600], "temperature": [30],"catalyst_loading": [0.498],
    }

initial_conditions_df = pd.DataFrame(sample_dataset)

parameter_columns = [param.name for param in searchspace.parameters]
data_df = pd.DataFrame(columns=parameter_columns)

        #print(f"Initial conditions - randomly generated: {initial_conditions_df}")
   
target_measurement = perform_df_experiment_multi(initial_conditions_df, emulator, objective=objective)
        #target_measurement = evaluate_candidates(initial_conditions_df)

campaign.add_measurements(target_measurement)
 
iterations = 10

for i in range(1, iterations+1):
    print(campaign)
    print(f"Running experiment {i }/{iterations}")
        
        
    recommended_conditions = campaign.recommend(batch_size=1)
            #print(f"Recommended conditions: {recommended_conditions}")

    data_df = pd.concat([data_df, recommended_conditions], ignore_index=True)

    target_measurement = perform_df_experiment(recommended_conditions, emulator, objective=objective)
            #target_measurement = evaluate_candidates(candidates=recommended_conditions)
    campaign.add_measurements(target_measurement)
    print('measurements in campaign!',campaign.measurements)
      
            #print(f"Iteration {i} took {(time.time() - t1):.2f} seconds")
        
            #eval_df_sobo = evaluate_candidates(target_measurement)
    new_yld = target_measurement['yld'].values[0]
    print(new_yld)

    campaign.measurements,