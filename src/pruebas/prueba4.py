import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

data_dir = "./data"
networks_dir = data_dir + "/networks"

g_bip_gt3 = ig.Graph().Read_GraphMLz(networks_dir + "/gt3_sample_90perc_edges.graphmlz")
g_bip_lt3 = ig.Graph().Read_GraphMLz(networks_dir + "/lt3_sample_90perc_edges.graphmlz")

users_gt3 = np.where(np.array(g_bip_gt3.vs["type"]) == False)[0]
users_lt3 = np.where(np.array(g_bip_lt3.vs["type"]) == False)[0]

users_degree_gt3 = g_bip_gt3.degree(users_gt3)
users_degree_lt3 = g_bip_lt3.degree(users_lt3)

print(np.unique(users_degree_gt3, return_counts=True))
print(np.unique(users_degree_lt3, return_counts=True))