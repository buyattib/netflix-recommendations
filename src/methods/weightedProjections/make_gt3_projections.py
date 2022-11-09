import numpy as np
import igraph as ig

from functions import degree_normalized_projection, heatS, probS, make_hybrid

##Correr por un lado probs y heats y por otro hybrid

data_dir = "./data"
networks_dir = data_dir + "/networks"
weighted_projections_dir = data_dir + "/weightedProjections/gt3"

#tomo la red sampleada de 5000 usuarios de grados < 1500 con el 90% de enlaces
g_bip = ig.Graph().Read_GraphMLz(networks_dir + "/gt3_sample_90perc_edges.graphmlz")
incidence_tuple = g_bip.get_incidence()
incidence_matrix = np.array(incidence_tuple[0])

# #calculo la matriz comun a todas las proyecciones y luego la uso para calcular probs y heats
degree_norm_matrix, objects_degree = degree_normalized_projection(incidence_matrix)
# probS_matrix = probS(degree_norm_matrix, objects_degree)
# heatS_matrix = heatS(probS_matrix)

# #guardo
# np.save(weighted_projections_dir + "/incidence_90perc_edges.npy", incidence_matrix)
# np.save(weighted_projections_dir + "/probs.npy", probS_matrix)
# np.save(weighted_projections_dir + "/heats.npy", heatS_matrix)

# #calculo las proyecciones hibridas con distintos parametros lambda
lambdas = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
for l in lambdas:
    hybrid_matrix = make_hybrid(degree_norm_matrix, objects_degree, alpha=l)
    np.save(weighted_projections_dir + f"/hybrid/hybrid_{l}.npy", hybrid_matrix)
