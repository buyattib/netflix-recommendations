import numpy as np

data_dir = "./data"
weighted_projections_dir = data_dir + "/weightedProjections/gt3"

probs = np.load(weighted_projections_dir + "/probs.npy")
heats = np.load(weighted_projections_dir + "/heats.npy")