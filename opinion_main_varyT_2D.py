import pickle
import sys, os, copy
import pandas as pd
import numpy as np
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter, sleep
from itertools import product

from opinion_resources_opt1 import (
    return_random_starting_points as return_starting_points_opt1,
)
from opinion_resources_opt2 import (
    return_random_starting_points as return_starting_points_opt2,
)
from opinion_resources_twolevel import perform_two_level_analysis, fit_level_one, fit_level_two, perform_what_if_intervention

t_start = perf_counter()

how_many_samples = 1
num_starting_points = 3
max_workers = 48
eps = 1e-9
how_many_prediction_samples = 50

P = 1  # number of platforms
M = 3  # number of opinions on each platform
K = 3  # number of interventions

# load sample dataset
samples = pickle.load(
    open("data/nato_tweets.p", "rb")
)

# load exogenous signal
S = pickle.load(open("data/nato_googletrends.p", "rb"))
S = np.concatenate([[x] * 24 for x in S])

# load interventions
X = pickle.load(open("data/nato_news.p", "rb"))
X = X[list(range(K)), :].astype(float)
X = np.repeat(X, 24, axis=1)

regularization_list = list(
    product(
        [0],  # level 1 regularization parameter
        [0.1],  # level 2 regularization parameter
    )
)

# training interval, 75 days x 24 hours
T_list = [75 * 24]
# test interval, 91 days x 24 hours
T_test_list = [91 * 24]


def main():

    if len(sys.argv) != 3:
        sys.exit()
    try:
        T_index = int(sys.argv[1])
        mode = int(sys.argv[2])
        
        # 1 - fit level one
        # 2 - fit level two
        # 3 - what if interventions
        # 4 - all
        if mode not in [1, 2, 3]:
            sys.exit()
    except:
        print("One of your arguments was not an integer.")
        sys.exit()

    # set train and test intervals
    how_long = T_list[T_index]
    how_long_test = T_test_list[T_index]

    opt1_starting_points = return_starting_points_opt1(num_starting_points, samples, P)
    opt2_starting_points = return_starting_points_opt2(num_starting_points, P, M, K)
    hyparam_x0index_list = list(
        product(regularization_list, range(num_starting_points))
    )

    r_regularization = [x[0] for x in hyparam_x0index_list]
    r_opt_1_x0 = [opt1_starting_points[x[1]] for x in hyparam_x0index_list]
    r_opt_2_x0 = [opt2_starting_points[x[1]] for x in hyparam_x0index_list]
    r_timeaveraged = [True] * len(hyparam_x0index_list)
    (
        r_samples,
        r_X,
        r_P,
        r_M,
        r_K,
        r_howlong,
        r_howlongtest,
        r_howmanysamples,
        r_howmanypredictionsamples,
        r_S,
    ) = list(
        zip(
            *repeat(
                [
                    samples,
                    X,
                    P,
                    M,
                    K,
                    how_long,
                    how_long_test,
                    how_many_samples,
                    how_many_prediction_samples,
                    S,
                ],
                len(hyparam_x0index_list),
            )
        )
    )
    r_logfitlabel = [
        "_".join(["T" + str(how_long), "reg" + str(x[0]), "start" + str(x[1])])
        for x in hyparam_x0index_list
    ]

    if mode == 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            sim_outputs = executor.map(
                fit_level_one,
                r_opt_1_x0,
                r_samples,
                r_S,
                r_regularization,
                r_timeaveraged,
                r_P,
                r_howlong,
                r_howmanysamples,
                r_logfitlabel,
            )
        fit_duration = perf_counter() - t_start
        pickle.dump(
            [sim_outputs, fit_duration],
            open("hypertuning/res_opt1.p", "wb"),
        )
    elif mode == 2:
        with open("hypertuning/res_opt1.p", "rb") as file:
            res_opt1 = pickle.load(file)[0]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            sim_outputs = executor.map(
                fit_level_two,
                res_opt1,
                r_opt_2_x0,
                r_samples,
                r_X,
                r_S,
                r_regularization,
                r_timeaveraged,
                r_P,
                r_M,
                r_K,
                r_howlong,
                r_howlongtest,
                r_howmanysamples,
                r_howmanypredictionsamples,
                r_logfitlabel,
            )
        fit_duration = perf_counter() - t_start
        pickle.dump(
            [sim_outputs, fit_duration],
            open("hypertuning/res_opt2.p", "wb"),
        )
    elif mode == 3:
        with open("hypertuning/res_opt2.p", "rb") as file:
            res_opt2 = pickle.load(file)[0]

        r_xopt = []
        for i in range(num_starting_points):
            r_xopt.append(res_opt2[0][i][0])

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            sim_outputs = executor.map(
                perform_what_if_intervention,
                r_xopt,
                r_samples,
                r_X,
                r_S,
                r_P,
                r_M,
                r_K,
                r_howlong,
                r_howlongtest,
            )
        fit_duration = perf_counter() - t_start
        pickle.dump(
            [sim_outputs, fit_duration],
            open("hypertuning/what_if.p", "wb"),
        )
    elif mode == 4:
        sim_output_collector = []
        fit_duration_collector = []
    
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            sim_outputs = executor.map(
                perform_two_level_analysis,
                r_opt_1_x0,
                r_opt_2_x0,
                r_samples,
                r_X,
                r_S,
                r_regularization,
                r_timeaveraged,
                r_P,
                r_M,
                r_K,
                r_howlong,
                r_howlongtest,
                r_howmanysamples,
                r_howmanypredictionsamples,
                r_logfitlabel,
            )

        sim_outputs = list(sim_outputs)
        fit_duration = perf_counter() - t_start

        sim_output_collector.append(sim_outputs)
        fit_duration_collector.append(fit_duration)

        pickle.dump(
            [sim_output_collector, fit_duration_collector],
            open(f"hypertuning/all_nato.p", "wb"),
        )


if __name__ == "__main__":
    main()
