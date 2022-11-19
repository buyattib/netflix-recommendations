import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

def make_igraph_ids(edges):
    #A los nodos les damos el atributo name que va a ser el id original, distinto al id que usa igraph (es la enumeracion eso).
    users_names = np.unique(edges[:,0])
    movies_names = np.unique(edges[:,1])
    
    #nro de usuarios y peliculas
    n_users = len(users_names)
    n_movies = len(movies_names)

    #Enumeracion desde 0 hasta el total.
    users_igraph_ids = np.arange(n_users)
    movies_igraph_ids = np.arange(n_users, n_users+n_movies)

    #Tipos para el grafo bipartito: los users son 0's y las movies son 1's.
    types = np.concatenate([np.full(shape=n_users, fill_value=0), np.full(shape=n_movies, fill_value=1)])

    #Mapeo entre el enumerado (id igraph) y los nombres (ids originales)
    users_id_name_map = {name: id for name, id in zip(users_names, users_igraph_ids)}
    movies_id_name_map = {name: id for name, id in zip(movies_names, movies_igraph_ids)}

    #Armo los arrays de edges pero con los ids de igraph
    edges_users_igraph = np.array([users_id_name_map[name] for name in edges[:,0]])
    edges_movies_igraph = np.array([movies_id_name_map[name] for name in edges[:,1]])

    #Ahora los users van de 0 a n_users y las movies van de n_users a n_users+n_movies
    edges_ids = np.array([edges_users_igraph, edges_movies_igraph]).T
    return edges_ids, types, users_id_name_map, movies_id_name_map

data_dir = "./data"
networks_dir = data_dir + "/networks"
edges_dir = data_dir + "/edges"


# ##Distribucion de grado de usuarios red total
# edges = np.load(edges_dir + "/gt3.npy")

# edges_ids, types, users_id_name_map, movies_id_name_map = make_igraph_ids(edges)
# users, users_degrees = np.unique(edges_ids[:, 0], return_counts=True)
# degrees, degrees_dist = np.unique(users_degrees, return_counts=True)

# lt10_users = np.sum(degrees_dist[:10])
# print(lt10_users)

# print(len(np.unique(edges[:,0])))
# print(len(np.unique(edges[:,1])))
# print(len((edges)))

# plt.plot(degrees_dist[:999]/len(users))
# plt.xlabel("Grado")
# plt.ylabel("Densidad")
# plt.show();


# ##Distribucion de grado de peliculas red total
# edges = np.load(edges_dir + "/gt3.npy")

# edges_ids, types, users_id_name_map, movies_id_name_map = make_igraph_ids(edges)
# movies, movies_degrees = np.unique(edges_ids[:, 1], return_counts=True)
# degrees, degrees_dist = np.unique(movies_degrees, return_counts=True)

# plt.plot(degrees_dist/len(movies))
# plt.xlabel("Grado")
# plt.ylabel("Densidad")
# plt.show();


# ##Distribucion de grado de peliculas samples
# for l in range(10):
#     sample = ig.Graph().Read_GraphMLz(networks_dir + f"/gt3/samples/sample_{l}.graphmlz")
#     sample_movies = np.array([i for i, v in enumerate(sample.vs) if v["type"] == True])
#     sample_movies_degrees = sample.degree(sample_movies)
#     sample_degrees_m, sample_degrees_dist_m = np.unique(sample_movies_degrees, return_counts=True)
#     n_sample_movies = len(sample_movies)

#     plt.plot(sample_degrees_dist_m/n_sample_movies)
#     plt.xlabel("Grado")
#     plt.ylabel("Densidad")
#     plt.title(f"sample {l}")
#     plt.show();


# ##Distribucion de grado de usuarios
# for l in range(10):
#     sample = ig.Graph().Read_GraphMLz(networks_dir + f"/gt3/samples/sample_{l}.graphmlz")
#     sample_users = np.array([i for i, v in enumerate(sample.vs) if v["type"] == False])
#     sample_users_degrees = sample.degree(sample_users)
#     sample_degrees_u, sample_degrees_dist_u = np.unique(sample_users_degrees, return_counts=True)
#     n_sample_users = len(sample_users)

#     plt.plot(sample_degrees_dist_u/n_sample_users)
#     plt.xlabel("Grado")
#     plt.ylabel("Densidad")
#     plt.title(f"sample {l}")
#     plt.show();
