import numpy as np

data_dir = "./data"
edges_dir = data_dir + "/edges"

edges = np.load(edges_dir + "/edges_low_degree.npy")
gt3 = edges[edges[:,2] > 3][:, [0,1,3]]
lt3 = edges[edges[:,2] < 3][:, [0,1,3]]

print(len(gt3))
print(len(np.unique(gt3[:,0])))
print(len(np.unique(gt3[:,1])))

print(len(lt3))
print(len(np.unique(lt3[:,0])))
print(len(np.unique(lt3[:,1])))

np.save(edges_dir + "/gt3.npy", gt3)
np.save(edges_dir + "/lt3.npy", lt3)