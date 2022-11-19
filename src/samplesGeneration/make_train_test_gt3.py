import numpy as np
import igraph as ig

data_dir = "./data"
networks_dir = data_dir + "/networks/gt3"

def get_edges_from_users(g):
    users_edges = [[] for i in range(5000)]
    for edge in g.es:
        user = edge.source if edge.source < 5000 else edge.target
        users_edges[user].append(edge)
    return users_edges

def delete_user_edges_random(user_edges, user_degree, delete_number):
    delete_edges_indexes = np.random.choice(np.arange(0, user_degree), size=delete_number, replace=False)
    delete_edges = []
    for i in delete_edges_indexes:
        edge = user_edges[i]
        delete_edges.append((edge.source, edge.target))
    return delete_edges

def delete_user_edges_dates(user_edges, delete_number):
    edges_dates = np.array([edge["date"] for edge in user_edges])
    sorted_indexes = np.argsort(edges_dates)[::-1]
    sorted_indexes_delete = sorted_indexes[:delete_number]
    delete_edges = []
    for i in sorted_indexes_delete:
        edge = user_edges[i]
        delete_edges.append((edge.source, edge.target))
    return delete_edges

def delete_edges(g):
    g_bip_copy_random = g.copy()
    g_bip_copy_dates = g.copy()

    users_edges = get_edges_from_users(g)

    delete_edges_random = []
    delete_edges_dates = []

    for user_edges in users_edges:
        degree = len(user_edges)
        delete_number = 1 if degree < 10 else int(0.1*degree)

        edges_random = delete_user_edges_random(user_edges, degree, delete_number)
        edges_dates = delete_user_edges_dates(user_edges, delete_number)
        
        delete_edges_random += edges_random
        delete_edges_dates += edges_dates
    
    g_bip_copy_random.delete_edges(delete_edges_random)
    g_bip_copy_dates.delete_edges(delete_edges_dates)

    result = {
        "random_delete_graph": g_bip_copy_random,
        "dates_delete_graph": g_bip_copy_dates,
        "random_delete_edges": np.array(delete_edges_random), 
        "dates_delete_edges": np.array(delete_edges_dates)
    }

    return result


for l in range(10):
    print(l)
    g_bip = ig.Graph().Read_GraphMLz(networks_dir + f"/samples/sample_{l}.graphmlz")

    n_edges_original = len(g_bip.es)
    n_movies_original = len([i for i, u in enumerate(g_bip.vs) if u["type"]])
    n_users_original = len([i for i, u in enumerate(g_bip.vs) if not u["type"]])

    result = delete_edges(g_bip)

    g_bip_copy_random = result["random_delete_graph"]
    g_bip_copy_dates = result["dates_delete_graph"]
    deleted_edges_random = result["random_delete_edges"]
    deleted_edges_dates = result["dates_delete_edges"]

    print("Comparacion entre cantidad de enlaces restantes y eliminados con los valores iniciales.")
    print(len(g_bip_copy_random.es)/n_edges_original)
    print(len(g_bip_copy_dates.es)/n_edges_original)
    print(len(deleted_edges_random)/n_edges_original)
    print(len(deleted_edges_dates)/n_edges_original)

    print("Comparacion entre cantidad de usuarios y peliculas restantes con los valores iniciales.")
    print(len(g_bip_copy_random.vs)/(n_movies_original+n_users_original))
    print(len(g_bip_copy_dates.vs)/(n_movies_original+n_users_original))

    print("Comparacion entre cantidad de usuarios y peliculas eliminados con los valores iniciales.")
    print(len(np.unique(deleted_edges_random[:,0]))/n_users_original)
    print(len(np.unique(deleted_edges_random[:,1]))/n_movies_original)
    
    print(len(np.unique(deleted_edges_dates[:,0]))/n_users_original)
    print(len(np.unique(deleted_edges_dates[:,1]))/n_movies_original)

    print("\n")


    g_bip_copy_random.write_graphmlz(f=networks_dir + f"/samplesTrainTest/90perc/net_90perc_{l}.graphmlz")
    np.save(networks_dir + f"/samplesTrainTest/10perc/edges_10perc_{l}.npy", deleted_edges_random)

    g_bip_copy_dates.write_graphmlz(f=networks_dir + f"/samplesTrainTestDates/90perc/net_90perc_{l}.graphmlz")
    np.save(networks_dir + f"/samplesTrainTestDates/10perc/edges_10perc_{l}.npy", deleted_edges_dates)