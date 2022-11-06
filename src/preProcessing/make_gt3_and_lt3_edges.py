import numpy as np

data_dir = "./data"
edges_dir = data_dir + "/edges"

edges = np.load(edges_dir + "/all_edges.npy")
gt3 = edges[edges[:,2] >= 3][:,:2]
lt3 = edges[edges[:,2] < 3][:,:2]

np.save(edges_dir + "/gt3.npy", gt3)
np.save(edges_dir + "/lt3.npy", lt3)