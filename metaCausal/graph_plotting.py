import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

def plot_data(data, labels, parameters=None, xlim=None, ylim=None):
    x_min = np.min(data[:, 0])
    x_max = np.max(data[:, 0])
    y_min = np.min(data[:, 1])
    y_max = np.max(data[:, 1])

    plt.figure(figsize=(10,10))
    if xlim is not None:
        plt.xlim(xlim[0], ylim[1])

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.scatter(data[:,0], data[:,1], c=labels, s=50)

    if parameters is not None:
        for i in range(len(parameters)):
            slope = parameters[i, 0]
            bias = parameters[i, 1]
            variance = parameters[i, 2]

            plt.plot([x_min,x_max], [bias+slope*x_min, bias+slope*x_max])
    plt.show()


def plotdata(x, y, keepxy=None, labels=None, title=None, slopes=None, interc=None, newfigure=True, plot_extend=1.1, ax=None):
    rng=np.random.default_rng(1)

    sns.set_theme()
    sns.set_style("white")
    #sns.color_palette("tab10", as_cmap=True)
    n=len(slopes)

    if newfigure:
        fig, ax = plt.subplots()

    if n == 1:
        plt.scatter(x, y, c=labels/n, vmin=0, vmax=1, cmap=sns.color_palette("hls", as_cmap=True), zorder=rng.integers(n))
    else:
        labels = labels / n
        for i in range(len(x)):
            plt.scatter(x[i], y[i], c=labels[i], vmin=0, vmax=1, cmap=sns.color_palette("hls", as_cmap=True), zorder=rng.integers(n))


    scalex = np.max(np.abs(x)) * plot_extend
    scaley = np.max(np.abs(y)) * plot_extend
    print(">", scalex, scaley)

    plt.xlim(-scalex, scalex)
    plt.ylim(-scaley, scaley)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)

    if title is not None:
        plt.title(title)

    if slopes is not None:
        for i in range(slopes.shape[0]):
            scale = scalex if keepxy[i] else scaley
            xx = [-scale, scale]
            yy = [xx[0]*slopes[i] + interc[i], xx[1]*slopes[i] + interc[i]]

            if keepxy[i]:
                plt.plot(xx, yy, zorder=n+1, linestyle="--", linewidth=2.5, c="#888")
            else:
                plt.plot(yy, xx, zorder=n+1, linestyle="--", linewidth=2.5, c="#888")
    if newfigure:
        plt.show()
