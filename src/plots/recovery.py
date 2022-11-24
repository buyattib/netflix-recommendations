import numpy as np
import matplotlib.pyplot as plt

# Recovery no va en funcion de L, grafico para los distintos metodos y pesos.

r = []
std_r = []
rw = []
std_rw = []

path = './data/metrics/metricsVsL'
tops = [5, 10, 15, 20, 25, 30, 40, 50]
files_names = ["probs", "heats", "hybrid2"]

for method in files_names:
    metrics = np.load(path + f"/{method}.npy")
    metrics_w = np.load(path + f"/{method}_weighted.npy")

    means = np.mean(metrics, axis=2)
    means_w = np.mean(metrics_w, axis=2)

    std = np.std(metrics, axis=2)
    std_w = np.std(metrics_w, axis=2)

    r.append(means[0][0])
    std_r.append(std[0][0])

    rw.append(means_w[0][0])
    std_rw.append(std_w[0][0])
    

methods = ["probs", "heats", "hybrid_0.2"]
plt.errorbar(methods, r, yerr=std_r, fmt=".", label="Sin pesar", c="dodgerblue")
plt.errorbar(methods, rw, yerr=std_rw, fmt=".", label="Pesado", c="lime")

plt.xlabel("Métodos")
plt.ylabel("Recovery")
plt.legend()

plt.show()