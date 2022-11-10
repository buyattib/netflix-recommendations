# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:41:33 2022

@author: Usuario
"""

#bajo las librerias y data

import numpy as np
import numpy_groupies as npg
import matplotlib.pyplot as plt
import pandas as pd
from itertools import repeat

data_dir = "./data"
edges_dir = data_dir + "/edges"

edges_by_users = np.load(edges_dir + "/all_edges.npy")

#armo un array de usuarios y otros de sus grados
users, degrees = np.unique(edges_by_users[:,0], return_counts=True)


max_user = np.max(users)

unique_degrees = np.unique(degrees)

#%%

ratings, counts = np.unique(edges_by_users[:,2], return_counts=True)
#%%

puntajes_usuarios = [[] for i in repeat(None, int(max_user)+1)]

for (user, movie, rating) in edges_by_users:
    puntajes_usuarios[int(user)].append(rating)
    
#%%

puntajes_usuarios_limpios = list(filter(None, puntajes_usuarios)) 

#%%

#puntajes_usuarios = []

#for i in users:
#    list_puntajes = [rating for user, movie, rating in edges_by_users if user == i]
#    puntajes_usuarios.append(list_puntajes)

#%%

dict_degrees = {}

for i in range(len(users)):
    dict_degrees[users[i]] = degrees[i]
    
#%%

df_degrees = pd.DataFrame.from_dict(dict_degrees, orient ='index', columns = ['degree'])
df_degrees['puntajes'] = puntajes_usuarios_limpios

#%%
df_degree_0_300 = df_degrees[df_degrees['degree'] < 300]
df_degree_300_1500 = df_degrees[df_degrees['degree'] >= 300]
df_degree_300_800 = df_degree_300_1500[df_degree_300_1500['degree']<=800]
df_degree_800_1500 = df_degrees[df_degrees['degree'] > 800]

#len(df_degree_800_1500)+len(df_degree_300_800)+len(df_degree_0_300) == len(df_degrees)
#dio igual, tiene pinta de estar bien
#%%
puntajes_0_300 = np.concatenate(df_degree_0_300['puntajes'].to_numpy())
puntajes_300_800 = np.concatenate(df_degree_300_800['puntajes'].to_numpy())
puntajes_800_1500 = np.concatenate(df_degree_800_1500['puntajes'].to_numpy())

#%%

#Hacemos el histograma

counts, bins = np.histogram(puntajes_0_300 , density=True, bins=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]))
max_count = np.max(counts)
bin_width = np.diff(bins[:2])[0]

#Lo graficamos
plt.figure(figsize=(10,8))

plt.bar(bins[:-1], counts, alpha=0.6, width=bin_width, align="edge", edgecolor="k");
#plt.vlines(y2h_avg_clustering, 0, max_count, linewidth=3, color="r", label="Clustering medio de la red y2h");

plt.legend(loc="upper right");
plt.xlabel("Puntajes");
plt.ylabel("Densidad");
# plt.xticks(bins[::2]);


#%%

counts, bins = np.histogram(puntajes_300_800 , density=True, bins=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]))
max_count = np.max(counts)
bin_width = np.diff(bins[:2])[0]

#Lo graficamos
plt.figure(figsize=(10,8))

plt.bar(bins[:-1], counts, alpha=0.6, width=bin_width, align="edge", edgecolor="k");
#plt.vlines(y2h_avg_clustering, 0, max_count, linewidth=3, color="r", label="Clustering medio de la red y2h");

plt.legend(loc="upper right");
plt.xlabel("Puntajes");
plt.ylabel("Densidad");
# plt.xticks(bins[::2]);
#%%

counts, bins = np.histogram(puntajes_800_1500 , density=True, bins=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]))
max_count = np.max(counts)
bin_width = np.diff(bins[:2])[0]

#Lo graficamos
plt.figure(figsize=(10,8))

plt.bar(bins[:-1], counts, alpha=0.6, width=bin_width, align="edge", edgecolor="k");
#plt.vlines(y2h_avg_clustering, 0, max_count, linewidth=3, color="r", label="Clustering medio de la red y2h");

plt.legend(loc="upper right");
plt.xlabel("Puntajes");
plt.ylabel("Densidad");
# plt.xticks(bins[::2]);

#%%

counts, bins = np.histogram(np.concatenate([puntajes_0_300, puntajes_800_1500, puntajes_300_800]) , density=True, bins=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]))
max_count = np.max(counts)
bin_width = np.diff(bins[:2])[0]

#Lo graficamos
#plt.figure(figsize=(10,8))

plt.bar(bins[:-1], counts, alpha=0.6, width=bin_width, align="edge", edgecolor="k");
#plt.vlines(y2h_avg_clustering, 0, max_count, linewidth=3, color="r", label="Clustering medio de la red y2h");

#plt.legend(loc="upper right");
plt.xlabel("Puntajes", fontsize = 16);
plt.ylabel("Densidad", fontsize = 16);
plt.xticks(fontsize=15);
plt.yticks(fontsize=15);
# plt.xticks(bins[::2]);