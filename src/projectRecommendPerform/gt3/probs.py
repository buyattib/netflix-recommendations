import numpy as np
import igraph as ig
from ..projections import degree_normalized_projection, probS
from ..recommendations import make_recommendation
from ..metrics import calculate_metrics
import time

#paths
data_dir = "./data"
networks_dir = data_dir + "/networks/gt3"

N = 1

recovery = np.zeros(N)
precision = np.zeros(N)
recall = np.zeros(N)
ep = np.zeros(N)
er = np.zeros(N)
personalization = np.zeros(N)
novelty = np.zeros(N)

for l in range(N):
    print(f"Iteracion {l}")
    t0 = time.time()

    #defino los paths a los datos
    path_90perc = networks_dir + f"/samplesTrainTestDates/90perc/net_90perc_{l}.graphmlz"
    path_10perc  = networks_dir + f"/samplesTrainTestDates/10perc/edges_10perc_{l}.npy"

    #cargo la red con el 90% de enlaces y el 10% de enlaces eliminados
    g_bip = ig.Graph().Read_GraphMLz(path_90perc)
    deleted_edges = np.load(path_10perc)

    n_users = len([i for i, u in enumerate(g_bip.vs) if not u["type"]])
    n_movies = len([i for i, u in enumerate(g_bip.vs) if u["type"]])

    #busco la matriz de incidencia de la red bipartita
    incidence_tuple = g_bip.get_incidence()
    incidence_matrix = np.array(incidence_tuple[0])

    # #busco la incidencia pesada por fechas
    weighted_incidence = np.load(networks_dir + f"/weightedIncidence/wi_{l}.npy")

    #calculo la matriz de pesos que tienen en comun todos los metodos
    degree_norm_matrix, objects_degree = degree_normalized_projection(incidence_matrix)
    #calculo probs
    probS_matrix = probS(degree_norm_matrix, objects_degree)

    #hago la recomendacion
    recommendations = make_recommendation(probS_matrix, incidence_matrix)
    recommendations_weighted = make_recommendation(probS_matrix, weighted_incidence)

    #calculo el diccionario de grados de peliculas
    movies = [i for i, u in enumerate(g_bip.vs) if u["type"]]
    degrees = g_bip.degree(movies)
    degrees_dict = {i : k for i, k in zip(movies, degrees)}

    #calculo las metricas
    L = 20
    r, p_r_tuple, h, I = calculate_metrics(recommendations, deleted_edges, degrees_dict, L)

    tf = time.time()
    print(f"Tiempo total de la iteracion {l}: {tf-t0}")

    recovery[l] = r

    precision[l] = p_r_tuple[0]
    recall[l] = p_r_tuple[1]
    ep[l] = p_r_tuple[2]
    er[l] = p_r_tuple[3]

    personalization[l] = h
    
    novelty[l] = I

