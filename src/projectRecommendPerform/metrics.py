import numpy as np

def get_deleted_edges_byuser(deleted_edges, n_users):
    users_deleted_movies = [[] for i in range(n_users)]
    for edge in deleted_edges:
        user, movie = (edge[0], edge[1]) if edge[0] < n_users else (edge[1], edge[0])
        users_deleted_movies[user].append(movie)
    
    return users_deleted_movies

def calculate_metrics(recommendations, deleted_edges, movies_degrees, L, saltear_r = False):
    r = []
    precision = []
    recall = []
    users_avg_information = []

    n_users, n_movies = recommendations.shape
    n_deleted_edges = len(deleted_edges)
    users_deleted_movies = get_deleted_edges_byuser(deleted_edges, n_users)
    
    recommendations_indices_L = recommendations[:, :L]-n_users
    counts_matrix = np.zeros((n_users, n_movies))

    for user in range(n_users):
        user_recommendation = recommendations[user]
        user_recommendation_L = user_recommendation[:L]
        user_deleted_movies = np.array(users_deleted_movies[user])
        
        #recovery
        if saltear_r == False:
            k_user = len(np.where(user_recommendation == -1)[0])
            deleted_movies_rankings = np.where(user_deleted_movies[:, None] == user_recommendation[None, :])[1]+1
            r_user = deleted_movies_rankings/(n_movies-k_user)
            r.append(r_user)


        #precision and recall
        n_user_deleted_movies = len(user_deleted_movies)
        d_user = sum(np.in1d(user_recommendation_L, user_deleted_movies))
        precision.append(d_user/L)
        recall.append(d_user/n_user_deleted_movies)

        #novelty
        user_recommendation_information = [np.log2(n_users/movies_degrees[movie]) for movie in user_recommendation_L if movies_degrees[movie] != 0]
        user_avg_information = np.mean(user_recommendation_information)
        users_avg_information.append(user_avg_information)

        #personalization
        counts_matrix[user, recommendations_indices_L[user]] = 1
    r_avg = 0
    if saltear_r == False:
        r_avg = np.mean(np.concatenate(r))
    
    precision = np.mean(precision)
    recall = np.mean(recall)
    ep = precision*n_users*n_movies/n_deleted_edges
    er = recall*n_movies/L

    I = np.mean(users_avg_information)

    q = np.matmul(counts_matrix, counts_matrix.T)
    h = 1-q/L
    avg_h = np.sum(np.triu(h, k=0))/(n_users*(n_users-1)/2)

    return r_avg, precision, recall, ep, er, avg_h, I
