import pickle

# Replace 'file_path.pkl' with the actual path to your .pkl file
file_path = 'benchmark_results_multi_trial_long_trial.pkl'

# Open the file in read-binary mode
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Print or inspect the data
print(data)