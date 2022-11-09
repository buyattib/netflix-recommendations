import numpy as np
import igraph as ig

# data_dir = "./data"
# networks_dir = data_dir + "/networks"
# weighted_projections_dir = data_dir + "/weightedProjections/gt3"

# incidence = np.load(weighted_projections_dir + "/incidence_90perc_edges.npy")

def similarity_matrix(incidence):
    users_degrees = np.sum(incidence, axis=1)
    pre_matrix = np.matmul(incidence, incidence.T)
    normalization = 1/np.sqrt(users_degrees)
    first_division = np.divide(pre_matrix, normalization)
    users_similarity = np.divide(first_division.T, normalization).T
    return users_similarity