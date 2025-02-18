import math

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap

from metaCausal.VX_sigm import sigm


def id_func(x, grad):
    return np.sign(grad)


def fixp(selfStress):
    selfStress = (1-0.05) * np.minimum(selfStress, 1)  # decay stress level
    sigmV, sigmGrad = sigm(selfStress)

    y = sigmV - selfStress
    yGrad = sigmGrad - 1

    return y, yGrad


def main():
    rng = np.random.default_rng(100)
    plt.rcParams['text.usetex'] = True

    # Apply the default theme
    sns.set_theme()
    sns.set_style("whitegrid")

    n = 5
    points = np.reshape(np.mgrid[0:n+0.5:0.5, 0:n+1], (2, -1)).T / n
    print(points.shape)

    fixpV, fixpGrad = fixp(points[:, 0] + 0.5*points[:,1])

    metaState = np.sign(fixpV)

    labels = ["MS1", "0", "MS2"]
    col_dict = {-2: "green",
                -0.01: "green",
                0: "black",
                0.01: "red",
                2: "red"}
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

    fig, ax = plt.subplots(figsize=(7,5), dpi=120)
    #plt.figure()

    rw=0.1
    eps=0.05
    r=[matplotlib.patches.Rectangle((-rw/2,-eps),rw, 1.0+2*eps)]
    pc = PatchCollection(r, facecolor="lightgrey", alpha=0.5)
    ax.add_collection(pc)

    print(np.max(points))

    cax=plt.quiver(
        points[:,1], points[:,0],
        np.zeros_like(fixpV), fixpV,
        metaState, scale=6,
        width=0.008,
        headwidth=4,
        headlength=5,
        headaxislength=5,
        cmap=cm)

    ax.grid(axis='y')

    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[np.sign(x)+1])

    #cb = fig.colorbar(cax, ticks=[-1,0,1], format=fmt, boundaries=[-2,2])

    #plt.scatter([], [], marker=r'$\longrightarrow$', c="yellow", s=120, label="Meta State 1")
    #plt.scatter([], [], marker=r'$\longrightarrow$', c="yellow", s=120, label="Meta State 2")
    #plt.legend()
    font_sizeA = 28
    font_sizeB = 24
    plt.xticks(fontsize=font_sizeB)
    plt.yticks(np.linspace(0,1,2*n+1), fontsize=font_sizeB)
    plt.xlabel("External Stressors", fontsize=font_sizeA)
    plt.ylabel("Stress Level", fontsize=font_sizeA)
    plt.tight_layout()
    plt.savefig("../figures/stressLvl.pdf")
    plt.show()


if __name__ == "__main__":
    main()
