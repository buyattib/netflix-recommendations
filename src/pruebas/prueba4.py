import numpy as np

data_dir = "./data"
weighted_projections_dir = data_dir + "/weightedProjections/gt3"
recommendations_dir = data_dir + "/recommendations/weightedProjections/gt3"

probs = np.load(recommendations_dir + "/all_probs.npy")
heats = np.load(recommendations_dir + "/all_heats.npy")

print(probs)
print(heats)