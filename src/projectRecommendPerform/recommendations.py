import numpy as np

def make_recommendation(weights_matrix, incidence_matrix):
    n_users = incidence_matrix.shape[0]
    scores = np.matmul(weights_matrix, incidence_matrix.T).T
    scores[np.where(incidence_matrix == 1)] = -1
    recommendations_newmovie = np.argsort(-scores)+n_users

    degrees = np.sum(incidence_matrix, axis=1)
    for i in range(n_users):
        ki = degrees[i]
        if ki != 0:
            recommendations_newmovie[i, -ki:] = -1

    return recommendations_newmovie