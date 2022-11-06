import numpy as np
import igraph as ig

data_dir = "./data"
networks_dir = data_dir + "/networks"

g_bip = ig.Graph().Read_GraphMLz(networks_dir + "/sample_lt3.graphmlz")

n_edges = len(g_bip.es)
edges_ids = np.arange(0, n_edges, 1)
edges_sample_indexes = np.random.choice(edges_ids, size=int(0.1*n_edges), replace=False)
edges = np.array([(edge.source, edge.target) for edge in g_bip.es])
drop_edges = edges[edges_sample_indexes]
drop_edges_tuples = [(e[0], e[1]) for e in drop_edges]

g_bip.delete_edges(drop_edges_tuples)

g_bip.write_graphmlz(f=networks_dir + "/lt3_sample_90perc_edges.graphmlz")
np.save(networks_dir + "/lt3_sample_10perc_edges.npy", drop_edges)