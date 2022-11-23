# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:27:22 2022

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


def get_deleted_edges_byuser(deleted_edges, max_users): #max users es el numero de usuarios contando desde 1 (no desde 0)
    users_deleted_movies = [[] for i in range(max_users+1)]
    for edge in deleted_edges:
        user, movie = (edge[0], edge[1]) if edge[0] < max_users+1 else (edge[1], edge[0])
        users_deleted_movies[user].append(movie)
    
    return users_deleted_movies

def calculate_metrics(recommendations, deleted_edges, movies_degrees, L, max_users, users):
    '''

    Parameters
    ----------
    recommendations : matriz de recomendacion con los usuarios en las filas y las peliculas en las columnas
    deleted_edges : TYPE
        DESCRIPTION.
    movies_degrees : TYPE
        DESCRIPTION.
    L : largo de la fila de recomendación
    max_users :  es el numero total de usuarios contando desde 1 (no desde 0)
    users : es un diccionario que como llave tiene el numero de fila en el
            que aparece en recommendations y el valor es la numeracion original del usuario 

    Returns
    -------
    r_avg : TYPE
        DESCRIPTION.
    precision : TYPE
        DESCRIPTION.
    recall : TYPE
        DESCRIPTION.
    ep : TYPE
        DESCRIPTION.
    er : TYPE
        DESCRIPTION.
    avg_h : TYPE
        DESCRIPTION.
    I : TYPE
        DESCRIPTION.

    '''
    r = []
    precision = []
    recall = []
    users_avg_information = []

    n_users, n_movies = recommendations.shape
    n_deleted_edges = len(deleted_edges)
    users_deleted_movies = get_deleted_edges_byuser(deleted_edges, max_users)
    
    recommendations_indices_L = recommendations[:, :L]-max_users
    counts_matrix = np.zeros((n_users, n_movies))

    for user in range(n_users):
        user_recommendation = recommendations[user]
        user_recommendation_L = user_recommendation[:L]
        user_deleted_movies = np.array(users_deleted_movies[users[user]])
        
        #recovery
        k_user = len(np.where(user_recommendation == -1)[0])
        deleted_movies_rankings = np.where(user_deleted_movies[:, None] == user_recommendation[None, :])[1]+1
        r_user = deleted_movies_rankings/(n_movies-k_user)
        r.append(r_user)

        #precision and recall
        n_user_deleted_movies = len(user_deleted_movies)
        d_user = sum(np.in1d(user_recommendation_L, user_deleted_movies))
        precision.append(d_user/L)
        recall.append(d_user/n_user_deleted_movies)

        #novelty
        user_recommendation_information = [np.log2(max_users/movies_degrees[movie]) for movie in user_recommendation_L if movies_degrees[movie] != 0]
        user_avg_information = np.mean(user_recommendation_information)
        users_avg_information.append(user_avg_information)

        #personalization
        counts_matrix[user, recommendations_indices_L[user]] = 1


    r_avg = np.mean(np.concatenate(r))
    
    precision = np.mean(precision)
    recall = np.mean(recall)
    ep = precision*max_users*n_movies/n_deleted_edges
    er = recall*n_movies/L

    I = np.mean(users_avg_information)

    q = np.matmul(counts_matrix, counts_matrix.T)
    h = 1-q/L
    avg_h = np.sum(np.triu(h, k=0))/(n_users*(n_users-1)/2)

    return r_avg, (precision, recall, ep, er), avg_h, I

#%%
r = []
epec = [] # precision
erec = [] # recovery
h = [] # personalizacion
i_nov = [] # novedad

L = 20

N = 1 #cantidad de redes

#paths
#os.getcwd() me da el path de donde estoy parada
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

    user_degree_dict = {}
    movie_degree_dict = {}
    degree_ = g_bip.degree()
    for i in range(len(g_bip.degree())):
        if i < n_users:
            user_degree_dict[i] = degree_[i]
        if i >= n_users:
            movie_degree_dict[i] = degree_[i]
            
    #degrees = np.unique(list(user_degree_dict.values()))
    degrees = np.arange(0, 1501)
    degree_users = [[] for i in repeat(None, max(degrees)+1)] #queremos guardar en la lista 6 los usuarios de grado 6 y asi
    for user in user_degree_dict:
        grado = user_degree_dict[user]
        degree_users[grado].append(user) 
    r_deg = [] # son listas donde en la posicion 0 va a estar el r de los usuarios de grado mas bajo
    epec_deg = [] # precision
    erec_deg = [] # recovery
    h_deg = [] # personalizacion
    i_deg = [] # novedad
    for degree in degrees:
        users_same_degree = degree_users[degree]
        if users_same_degree != []:
            dict_users = {}
            for indice, u in enumerate(users_same_degree):
                dict_users[indice] = u
            recomm_same_degree = recommendation[[users_same_degree]]
            r_avg, (precision, recall, ep, er), avg_h, I = calculate_metrics(recomm_same_degree, deleted, movie_degree_dict, L, max_user, dict_users)
        else: 
            r_avg, (precision, recall, ep, er), avg_h, I = np.nan, (np.nan, np.nan, np.nan, np.nan), np.nan, np.nan 
        r.append(r_avg)
        epec_deg.append(ep)
        erec_deg.append(er)
        h_deg.append(h)
        i_deg.append(I) 
    
    r.append(r_deg)
    epec.append(epec_deg)
    erec.append(erec_deg)
    h.append(h_deg)
    i_nov.append(i_deg)
    t1 = time.time()
    print('tardo', t1-t0)


r_average = np.nanmean(r, axis=0)
r_std = np.nanstd(r, axis=0)
pr_average = np.nanmean(epec, axis=0)
pr_std = np.nanstd(epec, axis=0)
rec_average = np.nanmean(erec, axis=0)
rec_std = np.nanstd(erec, axis=0)
h_average = np.nanmean(h, axis=0)
h_std = np.nanstd(h, axis=0)
i_average = np.nanmean(i_nov, axis=0)
i_std = np.nanstd(i_nov, axis=0)

plt.fig(1)

plt.errorbar(degrees, r_average, yerr = r_std, fmt='o-', markersize=15, linewidth = 3, elinewidth = 3,label = 'r') #color = 'darkcyan')
plt.errorbar(degrees, pr_average, yerr = pr_std, fmt='o-', markersize=15, linewidth = 3, elinewidth = 3, label = 'pre') #color = 'darkcyan')
plt.errorbar(degrees, rec_average, yerr = rec_std, fmt='o-', markersize=15, linewidth = 3, elinewidth = 3, label = 'rec')#color = 'darkcyan')
plt.errorbar(degrees, h_average, yerr = h_std, fmt='o-', markersize=15, linewidth = 3, elinewidth = 3, label = 'h')#color = 'darkcyan')
plt.errorbar(degrees, i_average, yerr = i_std, fmt='o-', markersize=15, linewidth = 3, elinewidth = 3, color = 'i')
plt.legend(fontsize = 14)
plt.xlabel("Grado", fontsize = 16);
plt.ylabel("Métricas", fontsize = 16);
plt.xticks(fontsize=15);
plt.yticks(fontsize=15);
    
   