import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math


def sample_mechanisms_linear(
        num_mechanisms, rng: np.random.Generator,
        num_samples_per_mechanism=100,
        min_slope_abs=0.2, max_slope_abs=5,
        max_bias_abs=5,
        min_variance=0.1, max_variance=2.0,
        min_sample_x=-10, max_sample_x=10,
        max_class_imbalance=0.0,
        noise_func=lambda var, size, rng: rng.laplace(0, var, size=size)
        ):
    """
    NOTE: slope and intercept parameters are always provided as if mechanism has x->y direction.
    To calcluate parameters in case of y-> (parameters[:, 3] == 1) use: slope' = 1/slope; intercept' = -intercept/slope

    :param num_mechanisms:
    :param rng:
    :param num_samples_per_mechanism:
    :param min_slope_abs:
    :param max_slope_abs:
    :param max_bias_abs:
    :param min_variance:
    :param max_variance:
    :param min_sample_x:
    :param max_sample_x:
    :param max_class_imbalance:
    :param noise_func:
    :return:
    """
    if num_mechanisms == 0:
        raise RuntimeError("Number of mechanisms parameter can not be zero.")

    num_total_samples = (
            math.floor(num_samples_per_mechanism * (1 + max_class_imbalance) * (num_mechanisms // 2)) +
            math.ceil(num_samples_per_mechanism * (1 - max_class_imbalance) * (num_mechanisms // 2)) +
            (0 if num_mechanisms % 2 == 0 else num_samples_per_mechanism)
    )

    ids = []
    data = np.empty([num_total_samples, 2])

    assert num_total_samples == num_mechanisms * num_samples_per_mechanism

    # sample linear equation parameters
    parameters = np.empty((num_mechanisms, 4))  # [slope, bias, variance, direction]
    parameters[:, 0] = rng.uniform(min_slope_abs, max_slope_abs, size=num_mechanisms) * (rng.binomial(1, p=0.5, size=num_mechanisms)-0.5)*2
    parameters[:, 1] = rng.uniform(-max_bias_abs, max_bias_abs, size=num_mechanisms)
    parameters[:, 2] = rng.uniform(min_variance, max_variance, size=num_mechanisms)
    parameters[:, 3] = rng.binomial(1, p=0.5, size=num_mechanisms)  # swap direction?: 0=(x->y); 1=(y->x)
    # note that parameters 0-2 are in the mechanism direction indicated by parameters[:, 3]!!!
    # in case one wants to check for inverted mechanism parameters
    # the parameters for the inverse line are given by: slope' = 1/slope; intercept' = -intercept/slope

    # sample data points
    save_idx = 0
    for mechanism_id in range(num_mechanisms):
        mechanism = parameters[mechanism_id]

        if mechanism_id < num_mechanisms // 2:
            num_specific_mechanism_samples = math.floor(num_samples_per_mechanism * (1 + max_class_imbalance))
        elif 2*mechanism_id + 1 == num_mechanisms:
            num_specific_mechanism_samples = num_samples_per_mechanism
        else:
            num_specific_mechanism_samples = math.ceil(num_samples_per_mechanism * (1 - max_class_imbalance))

        xs = rng.uniform(min_sample_x, max_sample_x, size=num_specific_mechanism_samples)
        ys = xs*mechanism[0] + mechanism[1] + noise_func(mechanism[2], size=num_specific_mechanism_samples, rng=rng)

        # swap mechanism direction
        tx = ys*mechanism[3] + (1-mechanism[3]) * xs
        ty = xs*mechanism[3] + (1-mechanism[3]) * ys
        xs = tx
        ys = ty

        data[save_idx:save_idx+num_specific_mechanism_samples,:] = np.vstack([xs,ys]).T
        save_idx += num_specific_mechanism_samples

        ids.extend([mechanism_id]*num_specific_mechanism_samples)

    ids = np.array(ids)

    # keep_normal = parameters[:, 3] == 0
    # inverse_parameters_p0 = 1 / parameters[:, 0]
    # inverse_parameters_p1 = -parameters[:, 1] / np.abs(parameters[:, 0], 1e-10)
    # parameters[:, 0] = np.where(keep_normal, parameters, inverse_parameters_p0)
    # parameters[:, 1] = np.where(keep_normal, parameters, inverse_parameters_p1)

    # check that all data was written
    assert save_idx == num_mechanisms * num_samples_per_mechanism

    return ids, data, parameters
