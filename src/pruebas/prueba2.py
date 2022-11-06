import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

data_dir = "./data"
networks_dir = data_dir + "/networks"

g_bip_gt3 = ig.Graph().Read_GraphMLz(networks_dir + "/gt3_sample_90perc_edges.graphmlz")
g_bip_lt3 = ig.Graph().Read_GraphMLz(networks_dir + "/lt3_sample_90perc_edges.graphmlz")

print(g_bip_gt3.vs.attributes())
print(g_bip_gt3.vs["original_name"][:5])
print(g_bip_gt3.vs["id"][:5])
print(list(g_bip_gt3.vs)[:5])