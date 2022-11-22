import numpy as np

def make_recommendation(weights_matrix, incidence_matrix, incidence_weights=None):
    n_users = incidence_matrix.shape[0]
    degrees = np.sum(incidence_matrix, axis=1)
    viewed_indices = np.where(incidence_matrix == 1)

    if incidence_weights is not None:
        incidence_matrix = incidence_weights

    scores = np.matmul(weights_matrix, incidence_matrix.T).T
    scores[viewed_indices] = -1
    recommendations_newmovie = np.argsort(-scores)+n_users

    for i in range(n_users):
        ki = degrees[i]
        if ki != 0:
            recommendations_newmovie[i, -int(ki):] = -1

    return recommendations_newmovie