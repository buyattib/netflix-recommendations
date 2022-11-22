import numpy as np
import igraph as ig
from ..projections import degree_normalized_projection, probS
from ..recommendations import make_recommendation
from ..metrics import calculate_metrics
import time

def reescale_array(array):
    x, y = array.shape
    a_max = np.max(array, axis=1).reshape((x, 1))
    a_min = np.min(array, axis=1).reshape((x, 1))
    dif = (a_max-a_min).reshape((x, 1))
    reescaled_array = 9*(array - a_min)/dif + 1 
    return reescaled_array

#paths
data_dir = "./data"
networks_dir = data_dir + "/networks/gt3"

N = 10

#las filas son las diferentes metricas segun: recovery, precision, recall, ep, er, personalization, novelty
results = np.zeros((7, N))
weighted_results = np.zeros((7, N))

for L in [20]:
    for l in range(N):
        print(f"Red {l}")
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
        weighted_incidence_original = np.load(networks_dir + f"/weightedIncidence/wi_{l}.npy")
        weighted_incidence = np.power(weighted_incidence_original, 2)

        #calculo la matriz de pesos que tienen en comun todos los metodos
        degree_norm_matrix, objects_degree = degree_normalized_projection(incidence_matrix)
        #calculo probs
        probS_matrix = probS(degree_norm_matrix, objects_degree)

        #hago la recomendacion
        recommendations = make_recommendation(probS_matrix, incidence_matrix)
        recommendations_weighted = make_recommendation(probS_matrix, incidence_matrix, weighted_incidence)

        #calculo el diccionario de grados de peliculas
        movies = [i for i, u in enumerate(g_bip.vs) if u["type"]]
        degrees = g_bip.degree(movies)
        degrees_dict = {i : k for i, k in zip(movies, degrees)}

        #calculo las metricas
        metrics = calculate_metrics(recommendations, deleted_edges, degrees_dict, L)
        metrics_weighted = calculate_metrics(recommendations_weighted, deleted_edges, degrees_dict, L)

        tf = time.time()
        print(f"Tiempo total de la iteracion {l}: {tf-t0}")

        for i in range(7):
            results[i, l] = metrics[i]
            weighted_results[i, l] = metrics_weighted[i]

print(results)
print(weighted_results)

results_mean = np.mean(results, axis=1)
results_std = np.std(results, axis=1)

weighted_results_mean = np.mean(weighted_results, axis=1)
weighted_results_std = np.std(weighted_results, axis=1)

print("\n")
print(results_mean)
print(results_std)

print("\n")
print(weighted_results_mean)
print(weighted_results_std)