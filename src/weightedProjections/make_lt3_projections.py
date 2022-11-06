import numpy as np
import igraph as ig

from functions import degree_normalized_projection, heatS, probS, make_hybrid

data_dir = "./data"
networks_dir = data_dir + "/networks"
weighted_projections_dir = data_dir + "/weightedProjections/lt3"

#tomo la red sampleada de 5000 usuarios de grados < 1500 con el 90% de enlaces
g_bip = ig.Graph().Read_GraphMLz(networks_dir + "/lt3_sample_90perc_edges.graphmlz")
incidence_tuple = g_bip.get_incidence()
incidence_matrix = np.array(incidence_tuple[0])

degree_norm_matrix, objects_degree = degree_normalized_projection(incidence_matrix)
probS_matrix = probS(degree_norm_matrix, objects_degree)
heatS_matrix = heatS(probS_matrix)

np.save(weighted_projections_dir + "/incidence.npy", incidence_matrix)
np.save(weighted_projections_dir + "/probs.npy", probS_matrix)
np.save(weighted_projections_dir + "/heats.npy", heatS_matrix)

lambdas = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
for l in lambdas:
    hybrid_matrix = make_hybrid(degree_norm_matrix, objects_degree, alpha=l)
    np.save(weighted_projections_dir + f"/hybrid/hybrid_{l}.npy", hybrid_matrix)
