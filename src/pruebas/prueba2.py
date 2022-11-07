import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

data_dir = "./data"
networks_dir = data_dir + "/networks"

g_bip_gt3 = ig.Graph().Read_GraphMLz(networks_dir + "/gt3_sample_90perc_edges.graphmlz")
drop_edges = np.load(networks_dir + "/gt3_sample_10perc_edges.npy")


# print(len(g_bip_gt3.vs))
# print(g_bip_gt3.vs.attributes())
# print(g_bip_gt3.vs["id"][:10])
# print(g_bip_gt3.vs["original_name"][:10])
# print(g_bip_gt3.vs["type"][:10])

# print(len(np.where(np.array(g_bip_gt3.vs["type"]) == False)[0]))
# print(np.where(np.array(g_bip_gt3.vs["type"]) == True)[0])


# print(len(g_bip_gt3.es))

# print([(e.source, e.target) for e in g_bip_gt3.es][:10])

# print(g_bip_gt3.get_eid(0, 9388))

# g_bip_gt3 = ig.Graph().Read_GraphMLz(networks_dir + "/gt3_sample_90perc_edges.graphmlz")
# g_bip_lt3 = ig.Graph().Read_GraphMLz(networks_dir + "/lt3_sample_90perc_edges.graphmlz")

# print(g_bip_gt3.vs.attributes())
# print(g_bip_gt3.vs["original_name"][:5])
# print(g_bip_gt3.vs["id"][:5])
# print(list(g_bip_gt3.vs)[:5])