import numpy as np
import igraph as ig
import time

#paths
data_dir = "./data"
networks_dir = data_dir + "/networks/gt3"

for l in range(10):
    print(l)
    #defino los paths a los datos
    path_90perc = networks_dir + f"/samplesTrainTestDates/90perc/net_90perc_{l}.graphmlz"

    g_bip = ig.Graph().Read_GraphMLz(path_90perc)

    n_users = len([i for i, u in enumerate(g_bip.vs) if not u["type"]])
    n_movies = len([i for i, u in enumerate(g_bip.vs) if u["type"]])

    t0 = time.time()

    incidence_matrix_dates = np.array(list(g_bip.get_adjacency(attribute="date")))
    incidence_matrix_dates = incidence_matrix_dates[:n_users, n_users:]
    weighted_incidence = np.zeros((n_users, n_movies))
    for user in range(n_users):
        row = incidence_matrix_dates[user]
        weighted_incidence[user] = row/np.max(row)

    t1 = time.time()
    print(t1-t0)
    np.save(networks_dir + f"/weightedIncidence/wi_{l}.npy", weighted_incidence)
