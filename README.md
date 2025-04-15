# Benchmarking

## Corrected_benchmark_trial Contents


File contents falls into the major categories below:

### 1. Required Inputs

- **suzuki_miyaura_catalysts.csv**  
  File containing the name and SMILES codes for the precatalyst scaffolds (P1 and P2).  
  Required for the definition of Molecular or Substance parameters within BoFire and BayBE search spaces.

- **suzuki_miyaura_ligands.csv**  
  File containing the name and SMILES codes for the available ligands (L1–L7).  
  Required for the definition of Molecular or Substance parameters within BoFire and BayBE search spaces.

### 2. Set-Up Files

- **Setup_file_final.py**  
  Defines the search spaces, objectives, strategies/recommenders, and ultimately the campaign or strategy  
  which serves as the framework for BayBE and BoFire optimisation experiments.

- **baybe_models_final.py**  
  Defines the functions for a single optimisation loop for BayBE single-objective and multi-objective optimisation –  
  including obtaining recommendations, measurements from the emulator, and updating the campaign.

- **bofire_setup_final.py**  
  Defines a function for a single optimisation loop for BoFire multi-objective optimisation –  
  including obtaining recommendations, measurements from the emulator, and updating the strategy.

- **Initialising_points_final.py**  
  Uses the BoFire random strategy model to select random points from the domain and convert into the formats  
  expected by BayBE and BoFire as an initial dataset for each repeat trial.

- **Random_trials.py**  
  Uses the BoFire random strategy model to carry out an optimisation experiment of 35 iterations and 15 repeat trials  
  for strategy performance comparison.

### 3. Running the Benchmark Procedure

- **run_benchmark_trials_final.py**  
  Defines a main function ('run_benchmark') to run the benchmark for the BayBE and BoFire SOBO and MOBO models.  
  Within each trial, the function initialises all three models from the same initial dataset and updates the pre-defined campaign/strategy.
  Then, the function carries out 35 iterations of the BO loop, consisting of getting recommendations, evaluating recommendations using the emulator, and finally updating the campaign/strategy.
  The function returns a pickle file with all results, including a table of objective values against iteration number and iteration time.

### 4. Benchmark Results

- **benchmark_results_multi_trial.pkl**  
  Example pickle file output from the `run_benchmark` function.

- **results_dict_random.pkl**  
  Example pickle file output from the `run_benchmark` function.

### 5. Analysis and Plotting

- **trials_analysis_v2_corr.py**  
  Loads the benchmarking results from the pickle file and separates them out by model and trial.

- **trials_analysis_graphs_corr.py**  
  Creates plots of cumulative maximum objective value against iteration for all models, including the random strategy. All values are averaged across 15 trials with standard deviation calculated.  
  The file includes different variations of these graphs with initial data points included.

- **times_analysis_corr.py**  
  Computes the average time taken per iteration across all trials for each model and plots the values.

- **mobo_plots.py**  
  Responsible for creating multi-objective plots.

- **pymoo_pareto.py**  
  Uses the Pymoo Python library to create Pareto plots and calculate and plot hypervolume indicator values for each MO model.

     
   
