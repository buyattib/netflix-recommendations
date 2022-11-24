import numpy as np
import matplotlib.pyplot as plt

path = './data/metrics/metricsVsL'
tops = [5, 10, 15, 20, 25, 30, 40, 50]
file_names = ["probs", "heats", "hybrid2"]

means = np.zeros((6, 8, 3)) #6 metricas, 8 valores de L y 3 metodos
stds = np.zeros((6, 8, 3))

means_w = np.zeros((6, 8, 3)) #6 metricas, 8 valores de L y 3 metodos
stds_w = np.zeros((6, 8, 3))

for i, method in enumerate(file_names):
    metrics = np.load(path + f"/{method}.npy")
    metrics_w = np.load(path + f"/{method}_weighted.npy")

    means[:, :, i] = np.mean(metrics[1:, :, :], axis=2)
    stds[:, :, i] = np.std(metrics[1:, :, :], axis=2)

    means_w[:, :, i] = np.mean(metrics_w[1:, :, :], axis=2)
    stds_w[:, :, i] = np.std(metrics_w[1:, :, :], axis=2)

metrics_names = ["Precision", "Recall", "EP", "ER", "Personalization", "Novelty"]
methods = ["probs", "heats", "hybrid_0.2"]
colors = ["dodgerblue", "lime", "deeppink"]

for j, metric in enumerate(metrics_names):
    for i, method in enumerate(methods):
        plt.errorbar(tops, means[j, :, i], yerr=stds[j, :, i], label=methods[i], c=colors[i], fmt=".", alpha=0.5)
        plt.errorbar(tops, means_w[j, :, i], yerr=stds_w[j, :, i], c=colors[i], fmt=".")

        plt.plot(tops, means[j, :, i], linestyle="--", c=colors[i], alpha=0.5)
        plt.plot(tops, means_w[j, :, i], linestyle="--", c=colors[i])

        plt.xlabel("L")
        plt.ylabel(metrics_names[j])
        
    plt.legend()
    plt.show()