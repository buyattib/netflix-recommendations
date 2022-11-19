# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def get_deleted_edges_byuser(deleted_edges, n_users):
    users_deleted_movies = [[] for i in range(n_users)]
    for edge in deleted_edges:
        user, movie = (edge[0], edge[1]) if edge[0] < n_users else (edge[1], edge[0])
        users_deleted_movies[user].append(movie)
    
    return users_deleted_movies

def calculate_metrics(recommendations, deleted_edges, movies_degrees, L):
    r = []
    precision = []
    recall = []
    users_avg_information = []

    n_users, n_movies = recommendations.shape
    n_deleted_edges = len(deleted_edges)
    users_deleted_movies = get_deleted_edges_byuser(deleted_edges, n_users)

    for user in range(n_users):
        user_recommendation = recommendations[user]
        user_recommendation_L = user_recommendation[:L]
        recommendations_indices_L = recommendations[:, :L]-n_users
        
        counts_matrix = np.zeros((n_users, n_movies))

        user_deleted_movies = np.array(users_deleted_movies[user])
        n_user_deleted_movies = len(user_deleted_movies)
        k_user = len(np.where(user_recommendation == -1)[0])

        #recovery
        deleted_movies_rankings = np.where(user_deleted_movies[:, None] == user_recommendation[None, :])[1]+1
        r_user = deleted_movies_rankings/(n_movies-k_user)
        r.append(r_user)

        #precision and recall
        d_user = sum(np.in1d(user_recommendation_L, user_deleted_movies))
        precision.append(d_user/L)
        recall.append(d_user/n_user_deleted_movies)

        #novelty
        user_recommendation_information = [np.log2(n_users/movies_degrees[movie]) for movie in user_recommendation_L if movies_degrees[movie] != 0]
        user_avg_information = np.mean(user_recommendation_information)
        users_avg_information.append(user_avg_information)

        #personalization
        counts_matrix[user, recommendations_indices_L[user]] = 1


    r_avg = np.mean(np.concatenate(r))
    
    precision = np.mean(precision)
    recall = np.mean(recall)
    ep = precision*n_users*n_movies/n_deleted_edges
    er = recall*n_movies/L

    I = np.mean(users_avg_information)

    q = np.matmul(counts_matrix, counts_matrix.T)
    h = 1-q/L
    avg_h = np.sum(h-np.diag(np.diag(h)))/(2*(n_users*n_movies-n_users))

    return r_avg, (precision, recall, ep, er), avg_h, I



def recovery2(recommendations, deleted_edges):
    r = []
    n_users, n_movies = recommendations.shape
    users_deleted_movies = get_deleted_edges_byuser(deleted_edges, n_users)
    #n_users_deleted_edges = len(np.unique(deleted_edges[:,0])) #nro de usuarios a los q se le elimino al menos un enlace
    for user in range(n_users):
        user_deleted_movies = np.array(users_deleted_movies[user])
        if len(user_deleted_movies) == 0:
            pass
        user_recommendation = recommendations[user]
        k_user = len(np.where(user_recommendation == -1)[0])
        deleted_movies_rankings = np.where(user_deleted_movies[:, None] == user_recommendation[None, :])[1]+1
        r_user = deleted_movies_rankings/(n_movies-k_user)
        r.append(r_user)
    
    r_avg = np.mean(np.concatenate(r))
    return r_avg

def novelty2(recommendations, movies_degrees, L, deleted_edges):
    users_deleted_edges = np.unique(deleted_edges[:,0])
    n_users = recommendations.shape[0]
    users_avg_information = []
    for user in users_deleted_edges:
        user_recommendation = recommendations[user, :L]
        user_recommendation_information = [np.log2(n_users/movies_degrees[movie]) for movie in user_recommendation if movies_degrees[movie] != 0]
        user_avg_information = np.mean(user_recommendation_information)
        users_avg_information.append(user_avg_information)
    
    I = np.mean(users_avg_information)
    return I

def precision_and_recall2(recommendations, deleted_edges, L):
    p = []
    r = []
    n_users, n_movies = recommendations.shape
    n_deleted_edges = len(deleted_edges)
    users_deleted_movies = get_deleted_edges_byuser(deleted_edges, n_users)
    for user in range(n_users):
        user_recommendation = recommendations[user, :L]
        user_deleted_movies = users_deleted_movies[user]
        d_user = sum(np.in1d(user_recommendation, user_deleted_movies))
        n_deleted_movies_user = len(user_deleted_movies)
        if len(n_deleted_movies_user) == 0:
            pass
        p.append(d_user/L)
        r.append(d_user/n_deleted_movies_user)
    
    pres = np.mean(p)
    rec = np.mean(r)
    ep = pres*n_users*n_movies/n_deleted_edges
    er = rec*n_movies/L

    return pres, rec, ep, er    

def personalization2(recommendations, L):
    n_users, n_movies = recommendations.shape
    recommendations_indices_L = recommendations[:, :L]-n_users
    counts_matrix = np.zeros((n_users, n_movies))
    for user in range(n_users):
        counts_matrix[user, recommendations_indices_L[user]] = 1

    q = np.matmul(counts_matrix, counts_matrix.T)
    h = 1-q/L
    avg_h = np.sum(h-np.diag(np.diag(h)))/(2*n_users*n_movies-n_users)

    return avg_h





def recovery(matriz_recomendaciones, enlaces_borrados):
    r = [] #esto va a ser el promedio de los r_i de cada usuario
    #num_usuarios = matriz_recomendaciones.shape[0] #el numero de filas es el número de usuarios
    num_usuarios, num_objetos = matriz_recomendaciones.shape
    enlaces_borrados_por_usuario = get_deleted_edges_byuser(enlaces_borrados, num_usuarios)
    num_usuarios_con_enlace_borrados = 0
    for i in range(len(matriz_recomendaciones)):
        r_i = []
        #print(i)
        recomend_usuario_i = matriz_recomendaciones[i] #esta es la tira de recomendaciones del usuario i
        #objetos_borrados_usuario_i = [objetos for (usuario, objetos) in enlaces_borrados if usuario == i]
        objetos_borrados_usuario_i = enlaces_borrados_por_usuario[i]
        enlaces_borrados_de_i = len(objetos_borrados_usuario_i) 
        if enlaces_borrados_de_i != 0:
            num_usuarios_con_enlace_borrados += 1
            for objetos in objetos_borrados_usuario_i:   
                place = 1 + np.where(recomend_usuario_i == objetos)[0][0] #posicion en la q fue recomendada la peli
                #print(place)
                #print(num_objetos)
                k_i = len(np.where(recomend_usuario_i == 4999)[0]) #+ enlaces_borrados_de_i
                #print(np.where(recomend_usuario_i == 4999))
                #print(k_i)
                r_i.append(place/(num_objetos-k_i))
                #print(place/(num_objetos-k_i))
        #print(r_i)
        r_average_i = np.mean(r_i)
        #print(r_average_i)
        r.append(r_average_i)
    return np.mean(r)


def precision_and_recall(matriz_recomendaciones, enlaces_borrados, L): #L es el largo de la lista de recomendacion a considerar
    p = []
    r = []
    num_usuarios, num_objetos = matriz_recomendaciones.shape
    peliculas_borradas_usuarios = get_deleted_edges_byuser(enlaces_borrados, num_usuarios)
    for i in range(num_usuarios):
        d_i = 0
        recomend_usuario_i = matriz_recomendaciones[i][0:L] #esta es la tira de recomendaciones del usuario i de largo L
        # objetos_borrados_usuario_i = [objetos for (usuario, objetos) in enlaces_borrados if usuario == i]
        objetos_borrados_usuario_i = peliculas_borradas_usuarios[i]
        enlaces_borrados_de_i = len(objetos_borrados_usuario_i)
        for objetos in objetos_borrados_usuario_i:
            if objetos in recomend_usuario_i:
                d_i += 1
        if enlaces_borrados_de_i != 0:
            p.append(d_i/L)
            r.append(d_i/enlaces_borrados_de_i)
    return np.mean(p), np.mean(r), np.mean(p)*num_usuarios*num_objetos/len(enlaces_borrados), np.mean(r)*num_objetos/L #los ultimos dos devuelven comparando contra un modelo nulo


def personalization(matriz_recomendaciones, L):
    h = []
    for i in range(len(matriz_recomendaciones)):
        for j in range(i+1, len(matriz_recomendaciones)):
            #q_ij = sum(matriz_recomendaciones[i][0:L] == matriz_recomendaciones[j][0:L]) #numero de items en comun en el top L de recomendaciones del usuario i y j.
            q_ij = sum(np.in1d(matriz_recomendaciones[i][0:L], matriz_recomendaciones[j][0:L]))
            h_ij = 1 - q_ij/L
            h.append(h_ij)
    return(np.mean(h))



def novelty(matriz_recomendaciones, dict_grados_objetos,enlaces_borrados , L): #dict_grados_objetos tiene que ser un diccionario que tiene como llave el objeto y como valor su grado
    num_usuarios = matriz_recomendaciones.shape[0]
    usuarios_enlaces_borrados = np.unique(enlaces_borrados.T[0])
    I  = []
    dict_info_objetos = {}
    for j in dict_grados_objetos:
        if dict_grados_objetos[j] != 0:
            dict_info_objetos[j] = np.log2(num_usuarios/dict_grados_objetos[j])
    for i in range(len(matriz_recomendaciones)):
        recomend_usuario_i = matriz_recomendaciones[i][0:L] #esta es la tira de recomendaciones del usuario i de largo L
        if i in usuarios_enlaces_borrados: #solo me quedo con los que al menos les eliminamos un link
            I_i = 0
            for objeto in recomend_usuario_i:
                if objeto != -4999:
                    I_i += dict_info_objetos[objeto]/L
            I.append(I_i)
    return np.mean(I)

