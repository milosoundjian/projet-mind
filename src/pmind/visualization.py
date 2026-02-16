import numpy as np
import matplotlib.pyplot as plt

def plot_perf_vs_rb_composition(proportions, performances, last_n=1, label=None):
    # proportions are timepoint x env x seed
    # average over seeds and then pool mean and std by environments and timepoints
    means = np.array([perf.mean(-1)[-last_n:].mean().item() for perf in performances])
     # TODO: do a better std agglomeration
    stds = np.array([perf.mean(-1)[-last_n:].std().item() for perf in performances])

    plt.plot(proportions, means, label=label)
    plt.fill_between(
        proportions,
        means - stds,
        means + stds,
        alpha=0.1,
    )
def plot_perf_vs_rb_composition_from_dict(proportions, perf_dict, last_n=1,legend_title="", fig_name=None):

    plt.figure()

    for label, performances in perf_dict.items():
        plot_perf_vs_rb_composition(proportions, performances,last_n ,label)

    plt.title("Offline learning performance\nfor different replay buffer compositions")
    plt.xlabel("% of uniform exploration")
    plt.ylabel("evaluation performance (mean $\\pm$ std)")
    if fig_name:
        plt.savefig(fig_name)
    plt.legend(title=legend_title)
    plt.show()
