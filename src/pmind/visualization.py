import numpy as np
import matplotlib.pyplot as plt
def plot_perf_vs_rb_composition(proportions, performances):
    # TODO: for now it just take the last evaluation, 
    # need to consider n-th evaluation instead
    means = np.array([perf[-1].mean().item() for perf in performances])
    stds = np.array([perf[-1].std().item() for perf in performances])
    plt.plot(proportions, means)
    plt.fill_between(proportions, means - stds, means + stds, alpha = 0.1)
    plt.title("Offline learning performance\nfor different replay buffer compositions")
    plt.xlabel("% of uniform exploration")
    plt.ylabel("evaluation performance")
    plt.show()