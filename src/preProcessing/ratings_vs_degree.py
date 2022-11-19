import numpy as np
import numpy_groupies as npg
import matplotlib.pyplot as plt

data_dir = "./data"
edges_dir = data_dir + "/edges"

edges_by_users = np.load(edges_dir + "/edges_low_degree.npy")
users, degrees = np.unique(edges_by_users[:,0], return_counts=True)

unique_degrees = np.unique(degrees)

users_mean_ratings = npg.aggregate(edges_by_users[:,0].astype(int), edges_by_users[:,2], func='mean', fill_value=0)
users_mean_ratings = users_mean_ratings[users.astype(int)]

users_degrees_ratings = np.array([users, degrees, users_mean_ratings]).T
# np.save(edges_dir + "/users_degrees_meanratings.npy", users_degrees_ratings)

ratings_mean = npg.aggregate(degrees.astype(int), users_mean_ratings, func='mean', fill_value=0)
ratings_mean = ratings_mean[unique_degrees.astype(int)]

plt.plot(unique_degrees[:1500], ratings_mean[:1500], linestyle="--")
plt.show()