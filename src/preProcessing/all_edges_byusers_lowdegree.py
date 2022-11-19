import numpy as np
import pandas as pd

data_dir = "./data"
edges_dir = data_dir + "/edges"

edges = np.load(edges_dir + "/all_edges.npy")
users, users_degrees = np.unique(edges[:, 0], return_counts=True)

users_to_max = np.arange(0, np.max(users)+1)
degrees_to_max = np.zeros(len(users_to_max))
degrees_to_max[users.astype(int)] = users_degrees

degree_column = np.array([degrees_to_max[int(i)] for i in edges[:,0]])
edges = np.array([edges[:,0], edges[:,1], edges[:,2], edges[:,3], degree_column]).T

df = pd.DataFrame(edges, columns=["users", "movies", "ratings", "dates", "user_degree"])
df = df[df["user_degree"] < 1500]

edges = df[["users", "movies", "ratings", "dates"]].to_numpy()

print(len(edges))
print(len(np.unique(edges[:,0])))
print(len(np.unique(edges[:,1])))

np.save(edges_dir + "/edges_low_degree.npy", edges)
