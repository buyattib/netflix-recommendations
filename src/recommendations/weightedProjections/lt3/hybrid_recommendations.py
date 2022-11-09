import numpy as np
import os

from ..functions import make_recommendation

data_dir = "./data"
weighted_projections_dir = data_dir + "/weightedProjections/lt3"
recommendations_dir = data_dir + "/recommendations/weightedProjections/lt3/hybrid"

hybrid_files = os.listdir(weighted_projections_dir + "/hybrid")

incidence = np.load(weighted_projections_dir + "/incidence_90perc_edges.npy")

for file in hybrid_files:
    l = float(file.strip("hybrid_").strip(".npy"))
    h = np.load(weighted_projections_dir + f"/hybrid/{file}")
    all_recommendations, recommendations_newmovie = make_recommendation(h, incidence)
    np.save(recommendations_dir + f"/{file}", recommendations_newmovie)

