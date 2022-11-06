import numpy as np

def make_recommendation(weights_matrix, incidence_matrix):
    n_users, n_movies = incidence_matrix.shape
    recommendations = np.zeros((n_users, n_movies)) #guardo la lista de recomendaciones completas
    recommendations_newmovie = (-1) * np.ones((n_users, n_movies)) #guardo la lista de recomendaciones sacando los objetos ya recolectados por el usuario

    for i in range(n_users):
        init_user_i = incidence_matrix[i, :] #vector de 1's y 0's segun si el usuario i ya vio la pelicula o no
        seen_movies = np.where(init_user_i != 0)[0] #indices de las peliculas vistas por el usuario i
        movies_scores_i = np.matmul(weights_matrix, init_user_i) #vector con los scores de las peliculas para el usuario i
        recommended_movies_i = np.argsort(movies_scores_i)[::-1] #vector con los indices de las peliculas ordenadas desde el maximo al minimo score para el usuario i
        recommended_new_movies_i = recommended_movies_i[~np.in1d(recommended_movies_i, seen_movies)] #al vector de peliculas ordenado le saco las que ya fueron vistas por el usuario i
        recommendations[i, :] = recommended_movies_i #guardo el vector como la fila i de la matriz probS_recommendations, corresponde al usuario i.
        recommendations_newmovie[i, :len(recommended_new_movies_i)] = recommended_new_movies_i #guardo el vector de peliculas ordenadas sin las que ya vio en las primeras n_movies-k_i columnas dejando -1's en las k_i columnas siguientes

    return recommendations, recommendations_newmovie