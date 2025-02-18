import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst
import json
import pandas as pd
from functools import lru_cache

# modified from: https://github.com/mike0295/Anderson-Darling-test-for-Laplace-distribution/tree/main

"""
This Anderson-Darling test for Laplacian distribution assumes that 
we do not know the location and scale parameters of the original distribution.

Null hypothesis: The sample data is drawn from a Laplacian distribution.
If result Anderson Statistic is larger than the critical value, 
the null hypothesis is rejected at the corresponding significance level.

Source for critical values:
https://www.tandfonline.com/doi/pdf/10.1081/SAC-9687287

Example code is at the bottom. Simply run the whole file to run example code
"""


anderson_sig_table_uneven = np.array([
    [0.5,  -0.928269, 0.2916],
    [0.75, -0.569497, 0.2637],
    [0.8,  -0.477586, 0.2636],
    [0.85, -0.367685, 0.2600],
    [0.9,  -0.227272, 0.2525],
    [0.95, -0.017972, 0.2053],
    [0.975, 0.163424, -0.0755],
    [0.990, 0.365898, -0.2964],
    [0.995, 0.503329, -0.6978],
    [0.9975, 0.623224, -0.8344],
    [0.999, 0.760512, -1.0847]])

anderson_sig_table_even = np.array([
    [0.5,  -0.932288, 3.2310],
    [0.75, -0.571021, 3.2644],
    [0.8,  -0.477644, 3.2895],
    [0.85, -0.368555, 3.3703],
    [0.9,  -0.228027, 3.4067],
    [0.95, -0.018564, 0.2053],
    [0.975, 0.162622, 3.2346],
    [0.990, 0.365076, 3.0296],
    [0.995, 0.500015, 2.7197],
    [0.9975, 0.616555, 2.4751],
    [0.999, 0.759613, 2.1387]])

#np.seterr(all='raise')

@lru_cache(maxsize=None)
def get_index_crit_value_params(even, crit_value):
    table = anderson_sig_table_even if even else anderson_sig_table_uneven
    for row in table:
        if row[0] == crit_value:
            return row[1], row[2]
    else:
        raise RuntimeError(f"Crit value entry for {crit_value} not found.")


def laplace_ad_test(
        data, loc=None, scale=None, crit_value=0.95,
        min_prob_clip=1e-15, # don't set it below 1e-16, or it won't have any effect... (1-min_prob_clip needs to be != 1)
        verbose=False,
        plot=False,
        supress_warnings=False,
        min_scale=0):
    n = data.shape[0]

    if loc is None:
        loc = np.median(data)
    if scale is None:
        scale = np.sum(np.abs(data - loc)) / n
    scale = np.maximum(scale, min_scale)

    if not supress_warnings and n < 25:
        logging.log(logging.WARN, "Anderson testing becomes instable with less than 25 samples.")

    z_i = np.zeros_like(data)

    for i, d in enumerate(data):
        if d < loc:
            z_i[i] = np.exp((d-loc)/scale)/2
        else:
            z_i[i] = 1 - np.exp((-d+loc)/scale)/2

    if min_prob_clip is not None:
        # clip to minimum probability to avoid numerical errors
        z_i = np.minimum(np.maximum(z_i, min_prob_clip), 1-min_prob_clip)

    rank = np.arange(1, n+1)
    z_i = np.sort(z_i)
    rev_z_i = np.flip(z_i)
    s = (2*rank-1)/n
    s *= (np.log(z_i)+np.log(1-rev_z_i))
    s = np.sum(s)
    A_sq = -n-s

    if verbose:
        print("Anderson statistic: {}\n".format(A_sq))
        print_crit_values(n)

    if plot:
        x = np.linspace(np.min(data), np.max(data), 1000)
        pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
        plt.plot(x, pdf, color='red', label='laplace')
        plt.hist(data, bins=85, density=True)
        plt.legend()
        plt.show()

    r1, r2 = get_index_crit_value_params(n % 2 == 0, crit_value)
    crit_value = np.exp(r1 + r2 / n)

    # lower A_sq is better (=distribution is more Laplacian)
    return A_sq <= crit_value, A_sq, crit_value


def print_crit_values(n):
    if n % 2 == 1:
        table = anderson_sig_table_uneven
    else:
        table = anderson_sig_table_even

    crit_val = {'Significance':[], 'Critical value':[]}
    for row in table:
        crit_val['Significance'].append(row[0])
        crit_val['Critical value'].append(np.exp(row[1]+row[2]/n))

    crit_val = pd.DataFrame.from_dict(crit_val)
    print("Critical value table\n", crit_val.to_string(index=False))


if __name__ == "__main__":
    # test some stuff
    data = sst.laplace.rvs(size=1000)
    print(laplace_ad_test(data, verbose=True, plot=True))

    data = sst.norm.rvs(size=1000)
    print(laplace_ad_test(data, verbose=True, plot=True))

    data = sst.laplace.rvs(size=1000)
    data = data[np.logical_or(data>0.5, data<-0.5)]
    print(laplace_ad_test(data, verbose=True, plot=True))


    cvs = [5, 10, 20, 50, 100, 1000, 10000]
    print(f"Crit values for {cvs} samples")
    re1, re2 = get_index_crit_value_params(True, 0.95)
    print("Crit Even  ", [np.exp(re1 + re2 / n) for n in cvs])

    ru1, ru2 = get_index_crit_value_params(False, 0.95)
    print("Crit Uneven", [np.exp(ru1 + ru2 / n) for n in cvs])
