import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import time

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

def make_giant_component(graph):
    graph_copy = graph.copy()
    giant_component_nodes = sorted(list(graph_copy.connected_components(mode='weak')), key=len, reverse=True)[0]
    giant_component = graph_copy.induced_subgraph(giant_component_nodes)
    return giant_component

def degree_normalized_projection(g_bip, which):
        incidence = np.array(g_bip.get_incidence()[0])
        user_degrees = np.maximum(np.sum(incidence, axis=1), 1)
        object_degrees = np.maximum(np.sum(incidence, axis=0), 1)
        if which == 0:
            normalized_incidence = np.divide(incidence, object_degrees)
            I = incidence.T
        elif which == 1:
            normalized_incidence = np.divide(incidence.T, user_degrees)
            I = incidence

        weights_matrix = np.matmul(normalized_incidence, I)
        return weights_matrix

def projection(g_bip, which):
        w = degree_normalized_projection(g_bip, which=which)
        projection = g_bip.bipartite_projection(which=which)
        
        for edge in projection.es:
            i, j = edge.source, edge.target
            if i != j:
                edge["weight"] = w[i][j]
        
        return projection

def make_vertex_color(labels):
        return [plt.get_cmap('Set1')(i) for i in labels]

def plot_projection(g, communities=None, title="", layout=None, giant_component=True, vertex_size=0.05, figsize=(38, 16)):
        if communities:
            g.vs["color"] = make_vertex_color(communities)
        
        if giant_component:
            g = make_giant_component(g)

        if not layout:
            layout = g.layout("fr")

        fig, axs = plt.subplots(figsize=figsize);
        ig.plot(
            g, 
            target=axs, 
            vertex_size=vertex_size,
            layout=layout,
            edge_width=0.5*np.array(g.es["weight"])
        )
        fig.suptitle(f"{title}", fontsize=16)
        plt.show();

def louvain_labels(G, weights=None):
        # louvain_obj = G.community_multilevel(weights=weights, resolution=0.2)
        louvain_obj = G.community_leiden(weights=weights, resolution_parameter=0.1, n_iterations=1000)
        louvain_labels = louvain_obj.membership
        return louvain_labels
    
def modularity(G, community_labels, directed=False, weights=None):
        Q = G.modularity(community_labels, directed=directed, weights=weights)
        return Q

data_dir = "./data"
gt3_path = data_dir + "/networks/gt3/samples/sample_0.graphmlz"

g_bip = ig.Graph().Read_GraphMLz(gt3_path)

g = projection(g_bip, which=1)

weights = g.es["weight"]
mean_w = np.mean(weights)

print(f"mean: {mean_w}")
print(f"min: {np.min(weights)}")
print(f"max: {np.max(weights)}")

print(len(weights))

t0 = time.time()
delete_edges = np.where(np.array(g.es["weight"]) < 1)[0]
print(f"eliminados: {len(delete_edges)}")
g.delete_edges(delete_edges)
print(f"restantes: {len(g.es)}")
tf = time.time()
print(tf-t0)

print("eliminados")

communities_louvain = louvain_labels(g, weights=g.es["weight"])
mod = modularity(g, communities_louvain, weights=g.es["weight"])

print(np.unique(communities_louvain))
print(f"modularidad: {mod}")

# t0 = time.time()
# plot_projection(g, communities=communities_louvain)
# tf = time.time()
# print(tf-t0)


# weights = g_users.es["weight"]
# mean_w = np.mean(weights)
# max_w = np.max(weights)
# min_w = np.min(weights)

# # bins = np.linspace(1, max_w, 20)
# plt.hist(g_users.es["weight"], bins=20)
# plt.show()