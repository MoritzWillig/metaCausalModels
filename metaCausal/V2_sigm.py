import math

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap

from metaCausal.VX_sigm import sigm


def drawCoord(ax):

    # Select length of axes and the space between tick labels
    min = 0.0
    max = 1.0
    xmin, xmax, ymin, ymax = min, max, min, max
    ticks_frequency = 0.1
    minortick_freq = 0.1

    # Set identical scales for both axes
    #ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    #ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    #ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Create minor ticks placed at each integer to enable drawing of minor grid
    # lines: note that this has no effect in this example with ticks_frequency=1
    ax.set_xticks(np.arange(xmin, xmax + 1, minortick_freq), minor=True)
    ax.set_yticks(np.arange(xmin, xmax + 1, minortick_freq), minor=True)

    # Draw major and minor grid lines
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    # Draw arrows
    #arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    #ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    #ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)


def main():
    rng = np.random.default_rng(100)

    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Apply the default theme
    #sns.set_theme()
    #sns.set_style("whitegrid")

    x = np.linspace(0,1.01,100)
    y, _ = sigm(x)

    fig, ax = plt.subplots(figsize=(7,5), dpi=120)

    plt.plot(x,y, color=(0.93,0.6,0.1), linewidth=2)
    plt.plot(x,x, color=(0.05,0.4,0.93), linewidth=2)

    plt.scatter([0,0.5,1], [0,0.5,1], s=40, color=(0.9, 0.2, 0.05), zorder=100)

    #ax.grid()
    drawCoord(ax)

    eps=0.04
    plt.xlim(-0.1,1+eps)
    plt.ylim(-0.1,1+eps)

    font_sizeA = 28
    font_sizeB = 24
    n = 5
    #plt.xticks(fontsize=font_sizeB)
    #plt.yticks(np.linspace(0,1,n+1), fontsize=font_sizeB)
    ax.xaxis.set_tick_params(labelsize=font_sizeB)
    ax.yaxis.set_tick_params(labelsize=font_sizeB)

    plt.xlabel(r"Stress Level${}_{\text{t-1}}$", fontsize=font_sizeA)
    plt.ylabel(r"Stress Level${}_{\text{t}}$", fontsize=font_sizeA)
    plt.tight_layout()
    plt.savefig("../figures/sigm.pdf")
    plt.show()


if __name__ == "__main__":
    main()
