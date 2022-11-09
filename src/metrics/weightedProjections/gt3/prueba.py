# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:28:27 2022

@author: Usuario
"""

import numpy as np
from metrics import recovery, precision_and_recall,  personalization, novelty
import tqdm.auto as tqdm #no funciono
import time
#%%

recomm_heat_matrix = np.load("new_heats.npy")
recomm_prob_matrix = np.load("new_probs.npy")
#%%
delete = np.load("gt3_sample_10perc_edges.npy")

#%%
import igraph as ig
#data_dir = "./data"
#networks_dir = data_dir + "/networks"
#weighted_projections_dir = data_dir + "/weightedProjections/gt3"

#tomo la red sampleada de 5000 usuarios de grados < 1500 con el 90% de enlaces
g_bip = ig.Graph().Read_GraphMLz("gt3_sample_90perc_edges.graphmlz")

#%%
degree_dict = {}
for i in range(len(g_bip.degree())):
    if i >4999:
        degree_dict[i] = g_bip.degree()[i]
#%%
degree_dict
#%%
start = time.process_time()

h_heat = personalization(recomm_heat_matrix, 20)

print(time.process_time() - start)
#%%
start = time.process_time()

h_prob = personalization(recomm_prob_matrix, 20)


print(time.process_time() - start)
#%%
start = time.process_time()

p_h, r_h, ep_h, er_h = precision_and_recall(recomm_heat_matrix, delete, 20)

print(time.process_time() - start)
#%%
start = time.process_time()

r_h = recovery(recomm_heat_matrix, delete)

print(time.process_time() - start)
#%%
p_p, r_p, ep_p, er_p = precision_and_recall(recomm_prob_matrix[:10], delete, 20)

r_p = recovery(recomm_prob_matrix, delete)
#%%
i_p = novelty(recomm_prob_matrix, degree_dict, delete, 20)
i_h = novelty(recomm_heat_matrix, degree_dict, delete, 20)
#%%
print(h_heat, h_prob)
print(p_h, r_h, ep_h, er_h)
print(p_p, r_p, ep_p, er_p)
print(r_h, r_p)

#%%
print(i_p,i_h)
        

#%% 

tupla_enlaces = delete#[0:10]

objetos_borrados_usuario_0 = [objetos for (usuario, objetos) in tupla_enlaces if usuario == 0]

print(objetos_borrados_usuario_0)

recomend_usuario_0 = recomm_prob_matrix[0]

for objetos in objetos_borrados_usuario_0:   
    print(objetos)
    #print(np.where(recomend_usuario_0 == objetos)[0][0])
    
#%%
recomendacion = np.array([np.array([2, 5, 1, 3, 4, 4999, 4999])])

enlaces_borrados_prueba = np.array([np.array([0, 1]), np.array([0, 3])])

r_prueba = recovery(recomendacion, enlaces_borrados_prueba)

print(r_prueba)

#%%

recomendacion = np.array([np.array([2, 5, 1, 3, 4, 4999, 4999]), np.array([4, 5, 6, 2, 3, 1, 4999])])

enlaces_borrados_prueba = np.array([np.array([0, 1]), np.array([0, 3]), np.array([1, 5])])

r_prueba = recovery(recomendacion, enlaces_borrados_prueba)

print(r_prueba)

#%%
recomendacion = np.array([np.array([2, 5, 1, 3, 4, 4999, 4999])])

enlaces_borrados_prueba = np.array([np.array([0, 1]), np.array([0, 3])])

p_prueba, rr_prueba, ep_prueba, er_prueba = precision_and_recall(recomendacion, enlaces_borrados_prueba, 3)

print(p_prueba, rr_prueba, ep_prueba, er_prueba)

#%%

recomendacion = np.array([np.array([2, 5, 1, 3, 4, 4999, 4999]), np.array([4, 5, 6, 2, 3, 1, 4999])])

enlaces_borrados_prueba = np.array([np.array([0, 1]), np.array([0, 3]), np.array([1, 5])])

p_prueba, rr_prueba, ep_prueba, er_prueba = precision_and_recall(recomendacion, enlaces_borrados_prueba, 3)

print(p_prueba, rr_prueba, ep_prueba, er_prueba)

#%%

recomendacion = np.array([np.array([2, 5, 1, 3, 4, 4999, 4999]), np.array([4, 5, 6, 2, 3, 1, 4999]), np.array([1, 6, 4, 2, 3, 5, 4999])])

h_prueba = personalization(recomendacion, 3)

print(h_prueba)

#%%

recomendacion = np.array([np.array([2, 1, 4999, 4999]), np.array([3, 4, 4999, 4999]), np.array([1, 4, 3, 4999])])

dict_grados_objetos = {1: 1, 2: 2, 3: 1, 4: 1}

enlaces_borrados_prueba = np.array([np.array([0, 1]), np.array([0, 2]), np.array([1, 3]), np.array([2, 1])])

n_prueba = novelty(recomendacion, dict_grados_objetos, enlaces_borrados_prueba, 2)

print(n_prueba)