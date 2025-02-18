import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import laplace

from metaCausal.decompose import separate_linear
from metaCausal.graph_plotting import plotdata
from metaCausal.mechanism_sampling import sample_mechanisms_linear

from multiprocessing import Pool

from pathlib import Path


save_path = Path("../sample_plots/")
#file_format = "pdf"
file_format = "png"


def plot_setup(params):
    id, num_mechanisms, max_class_imbalance = params

    rng = np.random.default_rng(100 + id)
    # sample a new mechanism setup
    labels_gt, data, parameters_gt = sample_mechanisms_linear(
        num_mechanisms, rng,
        num_samples_per_mechanism=500,
        min_slope_abs=0.2, max_slope_abs=5,
        max_bias_abs=5,
        min_variance=0.1, max_variance=4.0,
        min_sample_x=-5, max_sample_x=5,
        max_class_imbalance=max_class_imbalance
    )

    x = data[:, 0]
    y = data[:, 1]
    slopes = parameters_gt[:,0]
    interc = parameters_gt[:,1]
    keepxy = parameters_gt[:,3] == 0

    fig, ax = plt.subplots()
    plotdata(x, y, keepxy=keepxy, labels=labels_gt, title=None, slopes=slopes, interc=interc, newfigure=False, ax=ax)
    plt.savefig(save_path / f"ci0{int(max_class_imbalance*10)}_{num_mechanisms}_{id}.{file_format}")
    plt.close()



def main():
    rng = np.random.default_rng(100)

    save_path.mkdir(exist_ok=True, parents=True)

    start_at_mech = 1
    test_max_num_mechanisms = 4
    plot_num_setups = 10
    max_class_imbalance = 0.2

    for num_mechanisms in range(start_at_mech, test_max_num_mechanisms+1):
        print(f"Plotting Num Mechanisms {num_mechanisms}.")

        for i in range(plot_num_setups):
            plot_setup((i, num_mechanisms, max_class_imbalance))

    print("done.")


if __name__ == "__main__":
    main()
