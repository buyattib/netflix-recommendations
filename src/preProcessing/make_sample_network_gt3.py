import numpy as np
import igraph as ig

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
edges_dir = data_dir + "/edges"
networks_dir = data_dir + "/networks"

#enlaces con ratings mayores o menores a 3
edges = np.load(edges_dir + "/gt3.npy")

#armo los enlaces segun ids para igraph
edges_ids, types, users_id_name_map, movies_id_name_map = make_igraph_ids(edges)

#armo las redes
g_bip = ig.Graph().Bipartite(types, [])
g_bip.add_edges(edges_ids)

#pongo el atributo original_name en los vertices para guardar los ids originales de usuarios y peliculas
for name in users_id_name_map:
    id = users_id_name_map[name]
    g_bip.vs[id]["original_name"] = name

for name in movies_id_name_map:
    id = movies_id_name_map[name]
    g_bip.vs[id]["original_name"] = name

#busco todos los usuarios y separo los de grado mayor a 1500
all_users = np.where(types == 0)[0]
users_degrees = np.array(g_bip.degree(all_users))
users_high_degree = np.where(users_degrees >= 1500)[0]

#elimino los usuarios de grado mayor a 1500
g_bip.delete_vertices(users_high_degree)

#busco los usuarios que quedaron de grado menor a 1500 y de ellos tomo un sample de 5000 usuarios
n_users_low_degree = len(np.where(np.array(g_bip.vs["type"]) == False)[0])
users_low_degree = np.arange(0, n_users_low_degree)
users_low_degree_sample = np.random.choice(users_low_degree, size=5000, replace=False)

#tomo todos los usuarios que quedaron fuera del sample de 5000 para eliminarlos
drop_users_sample = np.setdiff1d(users_low_degree, users_low_degree_sample)

#los elimino
g_bip.delete_vertices(drop_users_sample)

#guardo la red
g_bip.write_graphmlz(f=networks_dir + "/sample_gt3.graphmlz")