import igraph as ig
import numpy as np

types = np.array([0,0,0,0,1,1,1,1])
edges_ids = np.array([ (0,4), (0,5), (1,4), (2,6), (3,6) ])

g_bip = ig.Graph().Bipartite(types, [])
g_bip.add_edges(edges_ids)

g_bip.vs["name"] = [f"pepe_{i}" for i in range(len(g_bip.vs))]

# print(list(g_bip.es))
# print([(e.source, e.target) for e in g_bip.es])

# g_bip.delete_edges([4])

# print(list(g_bip.vs))

# print(list(g_bip.es))

# print(list(g_bip.vs))
# print([v["name"] for v in g_bip.vs])

# g_bip.delete_vertices([0,5])

# print(list(g_bip.vs))
# print([v["name"] for v in g_bip.vs])