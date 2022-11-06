import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

data_dir = "./data"
networks_dir = data_dir + "/networks"

g_bip_gt3 = ig.Graph().Read_GraphMLz(networks_dir + "/sample_gt3.graphmlz")
g_bip_lt3 = ig.Graph().Read_GraphMLz(networks_dir + "/sample_lt3.graphmlz")

users_gt3 = np.where(np.array(g_bip_gt3.vs["type"]) == False)[0]
users_lt3 = np.where(np.array(g_bip_lt3.vs["type"]) == False)[0]

users_degree_gt3 = g_bip_gt3.degree(users_gt3)
users_degree_lt3 = g_bip_lt3.degree(users_lt3)

users_bins_gt3 = np.arange(0, np.max(users_degree_gt3), 1)
users_bins_lt3 = np.arange(0, np.max(users_degree_lt3), 1)

users_degree_dist_gt3 = np.histogram(users_degree_gt3, bins=users_bins_gt3)[0]
users_degree_dist_lt3 = np.histogram(users_degree_lt3, bins=users_bins_lt3)[0]

plt.plot(users_degree_dist_gt3)
plt.show();

plt.plot(users_degree_dist_lt3)
plt.show();