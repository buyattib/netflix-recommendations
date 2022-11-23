# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:13:13 2022

@author: Usuario
"""

import numpy as np
import igraph as ig 
from itertools import repeat
import matplotlib.pyplot as plt
import time
import os

#%% funciones -- tenes que pararte en C:\Users\Usuario\Desktop\Redes\Proyecto final\netflix-recommendations\src\projectRecommendPerform

from projections import degree_normalized_projection, probS
from recommendations import make_recommendation
from metrics import  calculate_metrics#(recommendations, deleted_edges, movies_degrees, L, max_user)

#%%

r = []
epec = [] # precision
erec = [] # recovery
h = [] # personalizacion
i_nov = [] # novedad

L_recom = np.arange(5, 41)

N = 1 #cantidad de redes


data_dir = 'C:\\Users\\Usuario\\Desktop\\Redes\\Proyecto final\\netflix-recommendations\\data'
networks_dir = data_dir + r"\networks\gt3"

max_user = 5000

for l in range(N):
    print(f"Red {l}")
    t0 = time.time()

    #defino los paths a los datos
    path_90perc = networks_dir + f"\\samplesTrainTestDates\\90perc\\net_90perc_{l}.graphmlz"
    path_10perc  = networks_dir + f"\\samplesTrainTestDates\\10perc\\edges_10perc_{l}.npy"

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
    weighted_incidence = np.power(weighted_incidence_original, 4)

    #calculo la matriz de pesos que tienen en comun todos los metodos
    degree_norm_matrix, objects_degree = degree_normalized_projection(incidence_matrix)
    #calculo probs
    probS_matrix = probS(degree_norm_matrix, objects_degree)

    #hago la recomendacion
    recommendations = make_recommendation(probS_matrix, incidence_matrix)
    recommendations_weighted = make_recommendation(probS_matrix, incidence_matrix, weighted_incidence)

    
    recommendation = recommendations
    deleted = deleted_edges

    movie_degree_dict = {}
    degree_ = g_bip.degree()
    for i in range(len(g_bip.degree())):
        if i >= n_users:
            movie_degree_dict[i] = degree_[i]
           
    epec_L = [] # son listas donde en la posicion 0 va a estar el precision de los usuarios de grado mas bajo
    erec_L = [] # recovery
    h_L = [] # personalizacion
    i_L = [] # novedad
    for L in L_recom:
        print('va por L = ', L)
        if L == min(L_recom):
            r_avg, precision, recall, ep, er, avg_h, I = calculate_metrics(recommendation, deleted, movie_degree_dict, L, saltear_r=False)
        else:
            nada, precision, recall, ep, er, avg_h, I = calculate_metrics(recommendation, deleted, movie_degree_dict, L, saltear_r=True)
        epec_L.append(ep)
        erec_L.append(er)
        h_L.append(h)
        i_L.append(I) 
    t1 = time.time()
    print('tardo', t1-t0)
    r.append(r_avg)
    epec.append(epec_L)
    erec.append(erec_L)
    h.append(h_L)
    i_nov.append(i_L)

r_average = np.nanmean(r)
r_std = np.nanstd(r)
pr_average = np.nanmean(epec, axis=0)
pr_std = np.nanstd(epec, axis=0)
rec_average = np.nanmean(erec, axis=0)
rec_std = np.nanstd(erec, axis=0)
h_average = np.nanmean(h, axis=0)
h_std = np.nanstd(h, axis=0)
i_average = np.nanmean(i_nov, axis=0)
i_std = np.nanstd(i_nov, axis=0)

plt.fig(1)

plt.hlines(r_average, min(L_recom), max(L_recom), fmt='o-', markersize=15, linewidth = 3, elinewidth = 3,label = 'r') #color = 'darkcyan')
plt.errorbar(L_recom, pr_average, yerr = pr_std, fmt='o-', markersize=15, linewidth = 3, elinewidth = 3, label = 'pre') #color = 'darkcyan')
plt.errorbar(L_recom, rec_average, yerr = rec_std, fmt='o-', markersize=15, linewidth = 3, elinewidth = 3, label = 'rec')#color = 'darkcyan')
plt.errorbar(L_recom, h_average, yerr = h_std, fmt='o-', markersize=15, linewidth = 3, elinewidth = 3, label = 'h')#color = 'darkcyan')
plt.errorbar(L_recom, i_average, yerr = i_std, fmt='o-', markersize=15, linewidth = 3, elinewidth = 3, color = 'i')
plt.legend(fontsize = 14)
plt.xlabel("Largo de lista recomendación", fontsize = 16);
plt.ylabel("Métricas", fontsize = 16);
plt.xticks(fontsize=15);
plt.yticks(fontsize=15);
    
#%%
print('hola')