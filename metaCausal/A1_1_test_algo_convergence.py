import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import laplace

from metaCausal.decompose import separate_linear
from metaCausal.mechanism_sampling import sample_mechanisms_linear
from sklearn.utils.fixes import parse_version, sp_version
from sklearn.linear_model import QuantileRegressor
import itertools
import time

from multiprocessing import Pool


def compute_statistics(
        active_mechanisms, labels_gt, label_predicted, parameters_gt, parameters_predicted,
        p0_parameter_tolerance, p1_parameter_tolerance
):
    # we check all permutations between prediction and real parameters
    # if at any we get a tolerance below 0.2 we assume to successfully having found the parameters.
    # only check slope and intercept (variance/scale might be distorted due to overlap effects between mechanisms)

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

    smallest_summed_avg_param_error = 10000
    best_permutation = None
    p0_avg_diff = None
    p1_avg_diff = None
    best_num_correct_directions = None

    success_line = False
    # check all mechanisms active
    if np.sum(active_mechanisms) == parameters_gt.shape[0]:
        num_mechanisms = parameters_gt.shape[0]
        permutations = itertools.permutations(range(num_mechanisms))
        for permutation in permutations:
            permutation = list(permutation)  # a tuple in np would select a single point at [x0,x1,...] (crashes).
            params_pred_permuted = pp[permutation, :]
            params_predi_permuted = ppi[permutation, :]
            dir_pred_permuted = dirp[permutation]

            # use inverted parameters if gt and prediction disagree on the direction
            correct_dir = dir_pred_permuted == dirgt
            params_pred_permuted = np.where(correct_dir[:, None], params_pred_permuted, params_predi_permuted)

            param_diff = params_pred_permuted - pgt  # [K, 2]
            param_diff_abs = np.abs(param_diff)
            if np.all(param_diff_abs[:, 0] < p0_parameter_tolerance) and np.all(param_diff_abs[:, 1] < p1_parameter_tolerance):
                success_line = True

                total_param_error = np.sum(param_diff_abs[:, 0]) + np.sum(param_diff_abs[:, 1])
                if total_param_error < smallest_summed_avg_param_error:
                    p0_avg_diff = np.mean(param_diff_abs[:, 0])
                    p1_avg_diff = np.mean(param_diff_abs[:, 1])
                    smallest_summed_avg_param_error = p0_avg_diff + p1_avg_diff
                    best_num_correct_directions = np.sum(correct_dir)

    return success_line, best_num_correct_directions, p0_avg_diff, p1_avg_diff, best_permutation


def test_setup(params):
    id, num_mechanisms, test_num_resamples, max_class_imbalance = params

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

    # we set number of algorithm samples to one to be able to record the results of every sample.
    # We manually simulate the sampling by repeatedly calling the algorithm.
    success_line_all = []
    num_correct_directions_all = []
    p0_avg_diff_all = []
    p1_avg_diff_all = []
    permutation_all = []

    for _ in range(test_num_resamples):
        parameters_predicted, active_mechanisms, labels_predicted, point_counts, joint_log_prob = separate_linear(
            data,
            num_mechanisms,
            num_em_iters=5 if num_mechanisms < 3 else 10,
            number_sample_repetitions=1,
            rng=rng,
            debug_func=None)

        success_line, num_correct_directions, p0_avg_diff, p1_avg_diff, permutation = compute_statistics(
            active_mechanisms,
            labels_gt, labels_predicted,
            parameters_gt, parameters_predicted,
            p0_parameter_tolerance=0.2,
            p1_parameter_tolerance=0.2
        )
        success_line_all.append(success_line)
        num_correct_directions_all.append(num_correct_directions)
        p0_avg_diff_all.append(p0_avg_diff)
        p1_avg_diff_all.append(p1_avg_diff)
        permutation_all.append(permutation)

    return success_line_all, num_correct_directions_all, p0_avg_diff_all, p1_avg_diff_all, permutation_all

def main():
    rng = np.random.default_rng(100)

    start_at_mech = 1
    test_max_num_mechanisms = 4
    test_num_setups = 500
    test_num_resamples = 10
    max_class_imbalance = 0.0
    print(f"{test_num_setups} setups with {test_num_resamples} resamples ({max_class_imbalance} max class imbalance)")

    for num_mechanisms in range(start_at_mech, test_max_num_mechanisms+1):
        print(f"======================================================")
        print(f"Collecting statistics for {num_mechanisms} mechanisms.")

        start_time = time.time()
        successful = 0
        num_correct_directions_c = []
        p0_avg_diff_c = []
        p1_avg_diff_c = []
        with Pool(8) as p:
            results = p.map(test_setup, [(i, num_mechanisms, test_num_resamples, max_class_imbalance) for i in range(test_num_setups)])

        for success_line_all, num_correct_directions_all, p0_avg_diff_all, p1_avg_diff_all, permutation_all in results:
            for i in range(len(success_line_all)):
                success_line = success_line_all[i]
                num_correct_directions = num_correct_directions_all[i]
                p0_avg_diff = p0_avg_diff_all[i]
                p1_avg_diff = p1_avg_diff_all[i]
                #permutation = permutation_all[i]
                if success_line:
                    successful += 1
                    num_correct_directions_c.append(num_correct_directions)
                    p0_avg_diff_c.append(p0_avg_diff)
                    p1_avg_diff_c.append(p1_avg_diff)

        print(f"Found {successful}/{test_num_setups*test_num_resamples} samples to converge.")
        if len(num_correct_directions_c) != 0:
            print(f"Directed {np.mean(num_correct_directions_c)}/{num_mechanisms} mechanisms on avg.")
            print(f"Avg P0 error: {np.mean(p0_avg_diff_c)}")
            print(f"Avg P1 error: {np.mean(p1_avg_diff_c)}")

        end_time = time.time()
        print(f"{num_mechanisms} mechanisms eval took {end_time - start_time} sec.")

    print("done.")


if __name__ == "__main__":
    main()
