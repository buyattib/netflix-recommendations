import numpy as np
from ..functions import make_recommendation

data_dir = "./data"
weighted_projections_dir = data_dir + "/weightedProjections/lt3"
recommendations_dir = data_dir + "/recommendations/weightedProjections/lt3"

incidence = np.load(weighted_projections_dir + "/incidence_90perc_edges.npy")
probs = np.load(weighted_projections_dir + "/probs.npy")

all_recommendations, recommendations_newmovie = make_recommendation(probs, incidence)

# np.save(recommendations_dir + "/all_probs.npy", all_recommendations)
np.save(recommendations_dir + "/probs.npy", recommendations_newmovie)