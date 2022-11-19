import numpy as np
import igraph as ig
from ..projections import degree_normalized_projection, probS
from ..recommendations import make_recommendation
from ..metrics import recovery2, precision_and_recall2, personalization2, novelty2
import time

#paths
data_dir = "./data"
networks_dir = data_dir + "/networks/gt3"

l=0

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

weighted_incidence = np.load(networks_dir + f"/weightedIncidence/wi_{l}.npy")


#calculo la matriz de pesos que tienen en comun todos los metodos
degree_norm_matrix, objects_degree = degree_normalized_projection(incidence_matrix)
#calculo probs
probS_matrix = probS(degree_norm_matrix, objects_degree)

#hago la recomendacion
recommendations = make_recommendation(probS_matrix, incidence_matrix)
recommendations_weighted = make_recommendation(probS_matrix, weighted_incidence)

t1 = time.time()

print(f"Post recomendations: {t1-t0}s")

movies = [i for i, u in enumerate(g_bip.vs) if u["type"]]
degrees = g_bip.degree(movies)
degrees_dict = {i : k for i, k in zip(movies, degrees)}

t2 = time.time()
print(f"Movies degrees dict: {t2-t1}s")

L = 20

r = recovery2(recommendations, deleted_edges)

t3 = time.time()
print(f"Recovery: {t3-t2}s")

pres, rec, ep, er = precision_and_recall2(recommendations, deleted_edges, L)
I = novelty2(recommendations, degrees_dict, L, deleted_edges)

t4 = time.time()
print(f"Pres and recov + novelty: {t4-t3}s")

h = personalization2(recommendations, L)

t5 = time.time()
print(f"Personalization: {t5-t4}s")

print(r)
print(pres, rec, ep, er)
print(I)
print(h)
