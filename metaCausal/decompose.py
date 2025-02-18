import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import laplace
from sklearn.exceptions import ConvergenceWarning

from metaCausal.anderson import laplace_ad_test
from metaCausal.graph_plotting import plotdata
from metaCausal.mechanism_sampling import sample_mechanisms_linear
from sklearn.linear_model import QuantileRegressor


log1em8 = np.log(1e-8)


def calculate_theoretical_num_samples(num_classes, max_imbalance, return_probablity=False):
    avg_prob = 1 / num_classes
    max_prob = avg_prob * (1+max_imbalance)
    min_prob = avg_prob * (1-max_imbalance)

    # probability for selecting a new class
    # first sample always guaranteed
    full_prob = 1
    # next num_classes//2 samples after selecting the biggest class
    for i in range(1, num_classes//2 + 1):
        full_prob *= 1 - (i*max_prob)
    # next num_classes//2 - 1 samples after selecting the smallest class
    # (-1 as there will be no class selected after the last one)
    for i in range(1, num_classes//2):
        full_prob *= i*min_prob
    # adjust for odd number of classes where one class has average size
    if num_classes % 2 == 1:
        residual_prob = 1 - ((num_classes/2 - 0.5) * max_prob) - avg_prob
        full_prob *= residual_prob

    # probability for selecting the same class
    # biggest class is selected num_classes//2 times; smallest class as well
    full_prob *= min_prob**(num_classes//2) * max_prob**(num_classes//2)
    # if odd number of classes, the average class is selected once
    if num_classes % 2 == 1:
        full_prob *= avg_prob

    if return_probablity:
        return full_prob

    # failure after n trials: failing n times in a row = (1-p)^n
    # atleast one success = 1 - (1-p)^n > 0.95
    # subtract 1: (1-p)^n <= 0.05
    # take log: n*ln(1-p) <= ln(0.05)
    # solve for n: n = ln(0.05) / ln(1-p)

    return math.ceil(math.log(0.05) / math.log(1-full_prob))


def nSamples(p, threshold=0.05):
    return np.ceil(np.log(threshold) / np.log(1-p))


real_sample_counts = {
    0.0: [
        np.NaN,  # zero-mechanisms....
        nSamples(4219 / 5000),
        nSamples(1740 / 5000),
        nSamples(592 / 5000),
        nSamples(86 / 5000)
    ],
    0.1: [
        np.NaN,  # zero-mechanisms....
        nSamples(4219/5000),
        nSamples(1702/5000),
        nSamples(577/5000),
        nSamples(84/5000)
    ],
    0.2: [
        np.NaN,  # zero-mechanisms....
        nSamples(4219/5000),
        nSamples(1567/5000),
        nSamples(555/5000),
        nSamples(84/5000)
    ]
}


def get_num_mechanisms(num_mechanisms, max_imbalance=0.0, empirical=True):
    if empirical:
        if max_imbalance not in real_sample_counts:
            raise RuntimeError(f"empirical results for max_imbalance = {real_sample_counts} where not evaluated.")
        return int(real_sample_counts[max_imbalance][num_mechanisms])
    else:
        int(calculate_theoretical_num_samples(num_mechanisms, max_imbalance))


def em(
        data,
        initial_parameters,
        num_em_iters,
        tol=1e-2,
        debug=False,
        debug_func=None):
    parameters = np.copy(initial_parameters)  # [K,4] - (slope, intercept, variance, direction)
    num_mechanisms = parameters.shape[0]  # = K

    warnings.filterwarnings('error', category=ConvergenceWarning, module='sklearn')  # QuantileRegressor doesn't properly check results and crashes
    qr = QuantileRegressor(quantile=0.5, alpha=0, fit_intercept=True, solver='highs')

    x = data[:, 0]
    y = data[:, 1]

    active_mechanisms = np.ones(num_mechanisms)
    all_converged = False
    labels = None
    residuals = None
    for i in range(num_em_iters+1):
        # data = [N,2]
        slope = parameters[:, 0]  # [K]
        intercept = parameters[:, 1]  # [K]
        aab = parameters[:, 2]  # [K] ; mean absolute error

        # select inverse mechanism if needed
        keep_xy = parameters[:, 3] == 0
        xx = np.where(keep_xy[None, :], x[:, None], y[:, None])
        yy = np.where(keep_xy[None, :], y[:, None], x[:, None])

        y_pred = xx * slope[None,:] + intercept[None, :]  # [N, K]
        residuals = yy - y_pred  # [N, K]

        #probs = 0.5*aab*np.exp(-np.abs(x)/aab)
        probs = laplace.pdf(residuals, loc=0, scale=aab)
        weights = probs / np.maximum(np.sum(probs, axis=1), 1e-20)[:, None]  # normalize weights per data point

        labels = np.argmax(weights, axis=1)

        if debug:
            plotdata(x, y, keep_xy, labels=labels, title=f"iter {i}", slopes=slope, interc=intercept)
            print("aab", aab)

        if all_converged or i == num_em_iters:
            # we recomputed labels again, but do not update parameters
            break

        # maximization
        all_converged = True
        for k in range(num_mechanisms):
            sample_weight = weights[:, k]
            is_mechanism_weight_active = np.sum(sample_weight) > 1e-6
            is_mechanism_label_active = np.sum(labels == k) > 2
            is_mechanism_active = is_mechanism_weight_active and is_mechanism_label_active
            active_mechanisms[k] = is_mechanism_active
            if not is_mechanism_active:
                # not enough support to fit line...
                continue

            # Least Absolute Deviations corresponds to median regression (minimizes means absolute error; L1 loss)
            def check_fit(xxx, yyy):
                try:
                    qr.fit(xxx[:, None], yyy, sample_weight=sample_weight)
                except ConvergenceWarning:
                    # QuantileRegressor doesn't properly check results and crashes
                    return None, None, None, None, False
                except TypeError:
                    # Just in case ConvergenceWarning aren't raised as errors for some reason...
                    return None, None, None, None, False

                # compute residuals for in-class points
                yyy_pred = xxx * qr.coef_ + qr.intercept_  # [N]
                yyy_residuals = yyy[labels == k] - yyy_pred[labels == k]  # [N]
                _, A_sq, _ = laplace_ad_test(
                    yyy_residuals,
                    crit_value=0.95,
                    supress_warnings=True)

                yyy_mean_residual = np.mean(np.abs(yyy_residuals))

                return qr.coef_, qr.intercept_, yyy_mean_residual, A_sq, True

            # in case of parameters[k, 3] was set to prefer yx direction, xx and yy here is already inversed.
            xy_coef, xy_inter, xy_mean_residual, xy_anderson, conv = check_fit(xx[:, k], yy[:, k])
            if conv:
                yx_coef, yx_inter, yx_mean_residual, yx_anderson, conv = check_fit(yy[:, k], xx[:, k])  # check if reverse direction is more 'laplacian'
            if not conv:
                active_mechanisms[k] = False
                continue

            # see comment above, this basically indicates to us whether to keep or inverse current mechanism direction.
            prefer_xy = xy_anderson < yx_anderson
            if prefer_xy:
                coef = xy_coef
                inter = xy_inter
                aab = xy_mean_residual
            else:
                coef = yx_coef
                inter = yx_inter
                aab = yx_mean_residual

            if abs(parameters[k, 0] - coef) > tol or abs(parameters[k, 1] - inter) > tol:
                all_converged = False

            parameters[k, 0] = coef
            parameters[k, 1] = inter
            parameters[k, 2] = np.maximum(aab, 1e-8)
            # again prefer_xy tells us to keep or inverse existing mechanism direction. (boiling down to xor)
            parameters[k, 3] = prefer_xy == parameters[k, 3]

            if not prefer_xy:
                # if we swapped the mechanism direction, we also need to swap the data for the next iteration
                temp = xx[:, k].copy()
                xx[:, k] = yy[:, k]
                yy[:, k] = temp

    point_counts = np.empty(num_mechanisms, dtype=int)
    joint_log_prob = 0  # higher is better
    for k in range(num_mechanisms):
        k_assigned = (labels == k)
        point_counts[k] = int(np.sum(k_assigned))
        active_mechanisms[k] = point_counts[k] != 0
        if not active_mechanisms[k]:
            continue

        log_probs = np.maximum(laplace.logpdf(residuals[k_assigned, k], loc=0, scale=parameters[k, 2]), log1em8)
        joint_log_prob += np.sum(log_probs)

    if np.isnan(joint_log_prob):
        raise RuntimeError()

    return parameters, active_mechanisms, labels, point_counts, joint_log_prob


def separate_linear(
        data,
        num_mechanisms,
        num_em_iters,
        number_sample_repetitions,
        rng: np.random.Generator,
        debug_func=None):

    best_score = -np.inf
    best_sample_results = None
    #print("!!!! num sample rep",number_sample_repetitions)
    for r in range(number_sample_repetitions):
        ids = rng.choice(data.shape[0], size=(num_mechanisms,2), replace=False)

        parameters = np.empty((num_mechanisms, 4))  # [slope, intercept, variance, direction]
        parameters[:, 0] = (data[ids[:,1],1] - data[ids[:,0],1]) / (data[ids[:,1],0] - data[ids[:,0],0])  # slope = (y1-y0)/(x1-x0)
        parameters[:, 1] = data[ids[:,0],1] - parameters[:, 0] * data[ids[:,0],0]  # intercept = y0 - slope * x0
        parameters[:, 2] = 1.0
        parameters[:, 3] = 0  # try xy direction first

        if debug_func is not None:
            debug_func("parameters_initial", parameters=parameters)

        sample_results = em(data, parameters, num_em_iters, debug_func=debug_func)

        _, _, _, point_counts, joint_log_prob = sample_results
        if joint_log_prob > best_score:
            best_sample_results = sample_results
            best_score = joint_log_prob
        #print("???", joint_log_prob, point_counts)

    if best_sample_results is None:
        raise RuntimeError()

    return best_sample_results  # (parameters, active_mechanisms, labels, point_counts, joint_log_prob)


def test_calculate_num_samples():
    # example max_imbalance = 0.2
    max_imbalance = 0.2

    print(f"=== TEST THEO NUM SAMPLES (mi {max_imbalance}) ===")
    print(f"max imb {max_imbalance}")
    num_classes = 1
    target_prob = 1
    calc_prob = calculate_theoretical_num_samples(num_classes, max_imbalance, return_probablity=True)
    print(f"num_classes: {num_classes}, target_prob: {target_prob}, calc_prob: {calc_prob}")

    num_classes = 2
    target_prob = 1 * 0.4 * 0.6 * 0.4
    calc_prob = calculate_theoretical_num_samples(num_classes, max_imbalance, return_probablity=True)
    print(f"num_classes: {num_classes}, target_prob: {target_prob}, calc_prob: {calc_prob}")

    num_classes = 3
    max_prob = (1 / 3) * 1.2
    min_prob = (1 / 3) * 0.8
    target_prob = 1 * (1 - max_prob) * (1 - max_prob - (1 / 3)) * max_prob * (1 / 3) * min_prob
    calc_prob = calculate_theoretical_num_samples(num_classes, max_imbalance, return_probablity=True)
    print(f"num_classes: {num_classes}, target_prob: {target_prob}, calc_prob: {calc_prob}")

    num_classes = 4
    target_prob = 1 * 0.7 * 0.4 * 0.2 * 0.3 * 0.3 * 0.2 * 0.2
    calc_prob = calculate_theoretical_num_samples(num_classes, max_imbalance, return_probablity=True)
    print(f"num_classes: {num_classes}, target_prob: {target_prob}, calc_prob: {calc_prob}")

    num_classes = 5
    max_prob = (1 / 5) * 1.2
    min_prob = (1 / 5) * 0.8
    target_prob = 1 * (1 - max_prob) * (1 - 2 * max_prob) * (1 - 2 * max_prob - (1 / 5)) * (
                1 - 2 * max_prob - (1 / 5) - min_prob) * max_prob * max_prob * (1 / 5) * min_prob * min_prob
    target_prob2 = 1 * (1 - max_prob) * (1 - 2 * max_prob) * (1 - 2 * max_prob - (1 / 5)) * (
        min_prob) * max_prob * max_prob * (1 / 5) * min_prob * min_prob
    calc_prob = calculate_theoretical_num_samples(num_classes, max_imbalance, return_probablity=True)
    print(f"num_classes: {num_classes}, target_prob: {target_prob}, calc_prob: {calc_prob}")
    print(f"num_classes: {num_classes}, target_prob2: {target_prob2}, calc_prob: {calc_prob}")

    num_classes = 10
    max_prob = (1 / 10) * 1.2
    min_prob = (1 / 10) * 0.8
    target_prob = 1 * 0.88 * 0.76 * 0.64 * 0.52 * 0.4 * 0.32 * 0.24 * 0.16 * 0.08 * max_prob ** 5 * min_prob ** 5
    calc_prob = calculate_theoretical_num_samples(num_classes, max_imbalance, return_probablity=True)
    print(f"num_classes: {num_classes}, target_prob: {target_prob}, calc_prob: {calc_prob}")

    print("=== COMPUTE TABLE Probs ===")
    for md in [0.0, 0.1, 0.2]:
        for k in range(1,4+1):
            calc_prob = calculate_theoretical_num_samples(k, md, return_probablity=True)
            print(f"max imb {md}, k {k}: {calc_prob}")

    print("=== COMPUTE TABLE SAMPLES ===")
    for md in [0.0, 0.1, 0.2]:
        for k in range(2,4+1):
            calc_prob = calculate_theoretical_num_samples(k, md, return_probablity=False)
            print(f"max imb {md}, k {k}: {calc_prob}")

    print("=== empirical samples ===")
    for max_imbalance in real_sample_counts.keys():
        print(f"Imbalance: {max_imbalance}")
        for num_mechanisms in range(1,4+1):
            print(f"{num_mechanisms}: {int(get_num_mechanisms(num_mechanisms, max_imbalance=max_imbalance, empirical=True))}")


if __name__ == "__main__":
    test_calculate_num_samples()
