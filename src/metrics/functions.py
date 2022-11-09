# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


def recovery(matriz_recomendaciones, enlaces_borrados):
    r = [] #esto va a ser el promedio de los r_i de cada usuario
    #num_usuarios = matriz_recomendaciones.shape[0] #el numero de filas es el número de usuarios
    num_objetos = matriz_recomendaciones.shape[1]
    num_usuarios_con_enlace_borrados = 0
    for i in range(len(matriz_recomendaciones)):
        r_i = []
        #print(i)
        recomend_usuario_i = matriz_recomendaciones[i] #esta es la tira de recomendaciones del usuario i
        objetos_borrados_usuario_i = [objetos for (usuario, objetos) in enlaces_borrados if usuario == i]
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
    num_usuarios = matriz_recomendaciones.shape[0]
    num_objetos = matriz_recomendaciones.shape[1]
    for i in range(len(matriz_recomendaciones)):
        d_i = 0
        recomend_usuario_i = matriz_recomendaciones[i][0:L] #esta es la tira de recomendaciones del usuario i de largo L
        objetos_borrados_usuario_i = [objetos for (usuario, objetos) in enlaces_borrados if usuario == i]
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

