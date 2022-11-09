import numpy as np
from ..functions import make_recommendation

data_dir = "./data"
weighted_projections_dir = data_dir + "/weightedProjections/lt3"
recommendations_dir = data_dir + "/recommendations/weightedProjections/lt3"

incidence = np.load(weighted_projections_dir + "/incidence_90perc_edges.npy")
heats = np.load(weighted_projections_dir + "/heats.npy")

all_recommendations, recommendations_newmovie = make_recommendation(heats, incidence)

# np.save(recommendations_dir + "/all_heats.npy", all_recommendations)
np.save(recommendations_dir + "/heats.npy", recommendations_newmovie)