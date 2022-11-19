import numpy as np

#Calcula la matriz de pesos común a probS y a heatS.
#El argumento incidence es la matriz de incidencia del grafo bipartito y debe tener a los usuarios en las filas y a los objetos en las columnas.
def degree_normalized_projection(incidence):
    #np.maximum pone 1 en los lugares donde hay ceros para no tener problemas al dividir.
    user_degrees = np.maximum(np.sum(incidence, axis=1), 1) #el grado de los nodos que estan en las filas (users)
    object_degrees = np.maximum(np.sum(incidence, axis=0), 1) #el grado de los nodos que estan en las columnas (objects)

    #proyeccion sobre los nodos que estan en las columnas (objects)
    normalized_incidence = np.divide(incidence.T, user_degrees)
    weights_matrix = np.matmul(normalized_incidence, incidence)
    return weights_matrix, object_degrees

def probS(weights_matrix, object_degrees):
    probS_matrix = np.divide(weights_matrix, object_degrees)
    return probS_matrix

def heatS(weights_matrix, object_degrees):
    probS_weights = probS(weights_matrix, object_degrees)
    return probS_weights.T

def make_hybrid(weights_matrix, object_degrees, alpha):
    pre_hybrid = np.divide(weights_matrix, np.power(object_degrees, alpha))
    hybrid = np.divide(pre_hybrid.T, np.power(object_degrees, 1-alpha)).T
    return hybrid