import numpy as np
import pandas as pd

data_dir = "./data"
edges_dir = data_dir + "/edges"

complete_edges_path = edges_dir + "/all_edges_bymovies.npy"
complete_edges = np.load(complete_edges_path)

df = pd.DataFrame(complete_edges, columns=["movies", "users", "rating"])
df = df[["users", "movies", "rating"]]
df.sort_values("users", inplace=True)

edges_by_users = df.to_numpy()
np.save(edges_dir + "/all_edges.npy", edges_by_users)