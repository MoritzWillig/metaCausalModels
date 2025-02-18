import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import laplace
import sys

sys.path.insert(0,'../')

from metaCausal.anderson import laplace_ad_test
from metaCausal.decompose import separate_linear, calculate_theoretical_num_samples, get_num_mechanisms
from metaCausal.graph_plotting import plotdata
from metaCausal.mechanism_sampling import sample_mechanisms_linear
import itertools
import time

from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('-p', type=int, help='number of threads', default=8)
args = parser.parse_args()
num_threads = args.p
print(f"Running with {num_threads} threads.")


def find_mechanisms(
        data,
        max_num_mechanisms,
        rng,
        max_imbalance=0.2,
        max_weight_diff=0.1,
        debug=False):
    results = [None]
    _deb_start_at_mechanism = 1
    #_deb_start_at_mechanism = 3
    for num_mechanisms in range(_deb_start_at_mechanism, max_num_mechanisms+1):
        # parameters_predicted, active_mechanisms, labels_predicted, point_counts, joint_log_prob
        mechanisms_result = separate_linear(
            data,
            num_mechanisms,
            num_em_iters=5 if num_mechanisms < 3 else 10,
            number_sample_repetitions=get_num_mechanisms(num_mechanisms, max_imbalance=max_imbalance, empirical=True),
            rng=rng,
            debug_func=None)
        results.append(mechanisms_result)

    x = data[:, 0]
    y = data[:, 1]

    best_mean_A_sq = np.inf
    best_num_mechanisms = 0
    best_parameters_predicted = None
    best_labels_predicted = None
    best_point_counts = None
    best_joint_log_prob = None

    for i in range(_deb_start_at_mechanism, max_num_mechanisms+1):
        parameters_predicted, active_mechanisms, labels_predicted, point_counts, joint_log_prob = results[i]
        num_mechanisms = int(np.sum(active_mechanisms))

        # compute residuals (same code as in em method)
        slope = parameters_predicted[:, 0]  # [K]
        intercept = parameters_predicted[:, 1]  # [K]
        aab = parameters_predicted[:, 2]  # [K] ; mean absolute error

        # select inverse mechanism if needed
        keep_xy = parameters_predicted[:, 3] == 0
        xx = np.where(keep_xy[None, :], x[:, None], y[:, None])
        yy = np.where(keep_xy[None, :], y[:, None], x[:, None])

        y_pred = xx * slope[None,:] + intercept[None, :]  # [N, K]
        residuals = yy - y_pred  # [N, K]

        #probs = 0.5*aab*np.exp(-np.abs(x)/aab)
        probs = laplace.pdf(residuals, loc=0, scale=aab)
        weights = probs / np.maximum(np.sum(probs, axis=1), 1e-20)[:, None]  # normalize weights per data point
        labels = np.argmax(weights, axis=1)

        # perform test with core samples only, to prevent skewing the distribution
        if num_mechanisms != 1:
            max_weights = np.max(weights, axis=1)
            weights_2 = np.copy(weights)
            weights_2[np.arange(len(weights_2)), labels] = -1  # 'remove' highest values
            max_weight_2 = np.max(weights_2, axis=1)

            # only select samples where the confidence of the top class is X% higher than the second one
            core_samples = max_weight_2 < max_weights*(1-max_weight_diff)
        else:
            core_samples = np.ones_like(y, dtype=bool)

        if debug:
            #if max_num_mechanisms
            #plotdata(x, y, labels=labels, title=f"all")
            plotdata(x[core_samples], y[core_samples], labels=labels[core_samples], title=f"core")


        residuals = residuals[core_samples, :]
        labels = labels[core_samples]

        all_accepted = []
        all_A_sq = []
        #print("XXXXXXXXX", active_mechanisms)
        for k, active in enumerate(active_mechanisms):
            if not active:
                continue

            #_, unique_counts = np.unique(labels, return_counts=True)
            #print("unique", unique_counts)
            #print("pk", point_counts)

            k_residuals = residuals[:,k][labels == k]
            is_H0_accepted, A_sq, crit_value = laplace_ad_test(
                k_residuals,
                loc=None,
                scale=None,
                crit_value=0.95,
                supress_warnings=True,
                plot=False,
                min_scale=1e-10)

            if debug:
                print(f"{k+1}/{i}", "!!", A_sq, crit_value, is_H0_accepted)

            all_accepted.append(is_H0_accepted)
            all_A_sq.append(A_sq)

        if np.alltrue(all_accepted):
            mean_A_sq = np.mean(all_A_sq)
            if mean_A_sq < best_mean_A_sq:
                best_num_mechanisms = num_mechanisms
                best_parameters_predicted = parameters_predicted
                best_labels_predicted = labels_predicted
                best_point_counts = point_counts
                best_joint_log_prob = joint_log_prob

    return best_num_mechanisms, best_parameters_predicted, best_labels_predicted, best_point_counts, best_joint_log_prob


def test_setup(params):
    id, actual_num_mechanisms, max_num_mechanisms, max_class_imbalance = params

    rng = np.random.default_rng(100 + id)
    # sample a new mechanism setup

    labels_gt, data, parameters_gt = sample_mechanisms_linear(
        actual_num_mechanisms, rng,
        num_samples_per_mechanism=500,
        min_slope_abs=0.2, max_slope_abs=5,
        max_bias_abs=5,
        min_variance=0.1, max_variance=4.0,
        min_sample_x=-5, max_sample_x=5,
        max_class_imbalance=max_class_imbalance
    )

    predicted_num_mechanisms, best_parameters_predicted, best_labels_predicted, best_point_counts, best_joint_log_prob = find_mechanisms(
        data,
        max_num_mechanisms,
        rng,
        max_imbalance=max_class_imbalance,
        max_weight_diff=0.4
    )

    success = predicted_num_mechanisms == actual_num_mechanisms
    success_diff = predicted_num_mechanisms - actual_num_mechanisms


    # FIXME calculate accuracy
    """
    # parameters = [K,4] (slope, intercept, variance, direction)
    pp = parameters_predicted[:, :2]
    dirp = parameters_predicted[:, 3]
    pgt = parameters_gt[:, :2]
    dirgt = parameters_gt[:, 3]

    # if gt and pred direction disagree we flip prediction parameters
    # compute inverse parameter predictions:
    clipped_scale = np.maximum(np.abs(parameters_predicted[:, 0]), 1e-10) * np.sign(parameters_predicted[:, 0])
    ppi = np.array([
        1 / clipped_scale,
        -parameters_predicted[:, 1]/ clipped_scale
    ]).T

    count number of correct classifications
    permutations = itertools.permutations(range(num_mechanisms))
    for permutation in permutations:
        permutation = list(permutation)  # a tuple in np would select a single point at [x0,x1,...] (crashes).
        params_pred_permuted = pp[permutation, :]
        params_predi_permuted = ppi[permutation, :]
        dir_pred_permuted = dirp[permutation]
    """

    #return success, success_diff, predicted_num_mechanisms, accuracy
    return success, success_diff, predicted_num_mechanisms


def main():
    rng = np.random.default_rng(100)

    start_at_mech = 1
    test_max_num_mechanisms = 4
    test_num_setups = 100
    max_class_imbalance = 0.2
    print(f"{test_num_setups} setups with resamples")

    success_all = np.zeros(test_max_num_mechanisms+1)
    success_diff_all = np.zeros(test_max_num_mechanisms+1)
    predicted_num_mechanisms_conf = np.zeros((test_max_num_mechanisms+1, test_max_num_mechanisms+1))

    for num_mechanisms in range(start_at_mech, test_max_num_mechanisms+1):
        print(f"======================================================")
        print(f"Collecting statistics for {num_mechanisms} mechanisms.")

        start_time = time.time()
        with Pool(num_threads) as p:
            results = p.map(test_setup, [(i, num_mechanisms, test_max_num_mechanisms, max_class_imbalance) for i in range(test_num_setups)])

        for success, success_diff, predicted_num_mechanisms in results:
            success_all[num_mechanisms] += 1
            success_diff_all[num_mechanisms] += success_diff
            predicted_num_mechanisms_conf[num_mechanisms, predicted_num_mechanisms] += 1

        end_time = time.time()
        print(f"Pred: {predicted_num_mechanisms_conf[num_mechanisms, :]}")
        print(f"{num_mechanisms} mechanisms eval took {end_time - start_time} sec.")

    accuracies = success_all / test_num_setups
    avg_accuracy = np.mean(accuracies)
    success_diff_all = success_diff_all / test_num_setups

    print(f"Avg Accuracy {avg_accuracy}")
    print("Accuracies:")
    print(accuracies)
    print("Avg Errors:")
    print(success_diff_all)
    print("Confusion Matrix:")
    print(predicted_num_mechanisms_conf)

    print("done.")


if __name__ == "__main__":
    main()
