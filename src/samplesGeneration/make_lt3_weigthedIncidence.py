import numpy as np
import igraph as ig
import time

#paths
data_dir = "./data"
networks_dir = data_dir + "/networks/lt3"

for l in range(10):
    print(l)
    #defino los paths a los datos
    path_net = networks_dir + f"/samples/sample_{l}.graphmlz"
    path_90perc = networks_dir + f"/samplesTrainTestDates/90perc/net_90perc_{l}.graphmlz"
    deleted_10perc = np.load(networks_dir + f"/samplesTrainTestDates/10perc/edges_10perc_{l}.npy")

    t0 = time.time()

    g_bip = ig.Graph().Read_GraphMLz(path_net)
    g_bip_90perc = ig.Graph().Read_GraphMLz(path_90perc)

    n_users = len([i for i, u in enumerate(g_bip_90perc.vs) if not u["type"]])
    n_movies = len([i for i, u in enumerate(g_bip_90perc.vs) if u["type"]])

    users_deleted_edges_dates = [[] for i in range(n_users)]
    for edge in deleted_10perc:
        user = edge[0] if edge[0] < n_users else edge[1]
        edge_id = g_bip.get_eid(edge[0], edge[1])
        date = g_bip.es[edge_id]["date"]
        users_deleted_edges_dates[user].append(date)
    
    mean_users_deleted_dates = np.array([np.mean(dates) for dates in users_deleted_edges_dates])

    t1 = time.time()
    print(f"Fechas eliminadas promedio tarda: {t1-t0}")

    adjacency_matrix_dates = np.array(list(g_bip_90perc.get_adjacency(attribute="date")))
    incidence_matrix_dates = adjacency_matrix_dates[:n_users, n_users:]
    weighted_incidence = np.zeros((n_users, n_movies))

    t2 = time.time()
    print(f"La adyacencia pesada tarda: {t2-t1}")

    for user in range(n_users):
        row = incidence_matrix_dates[user]
        weighted_incidence[user] = row/mean_users_deleted_dates[user]

    t3 = time.time()
    print(f"La incidencia pesada tarda: {t3-t2}")

    np.save(networks_dir + f"/weightedIncidence/wi_{l}.npy", weighted_incidence)
