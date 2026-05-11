import colorsys

import os
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_colors(n, s=0.7, v=0.9):
    colors = []
    for i in range(n):
        h = i / n  # evenly spaced hue
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append(
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        )
    return colors


def smooth(x, window, mode="gaussian"):
    if window % 2 == 0:
        raise ValueError("Window size must be odd")

    radius = window // 2

    if mode == "gaussian":
        sigma = (window - 1) / 6  # key conversion

        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-(t**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        return np.convolve(x, kernel, mode="valid")

    elif mode == "max":
        return np.array([np.max(x[i : i + window]) for i in range(len(x) - window + 1)])

    else:
        raise ValueError("Unknown mode")


def aggregate_learning_curves(
    rb_composition_experiment,
    smooth_mode=None,
    smooth_window=3,
    step_to_take=None,
    last_steps=1,
):

    rb_composition = rb_composition_experiment["rb_composition"]
    eval_interval = rb_composition_experiment["eval_interval"]

    # List by proportion of arrays EVAL_NUM x ENV x SEED
    performances = rb_composition_experiment["performances"]
    nb_evals, nb_envs, nb_seeds = performances[0].shape

    last_performances = np.zeros(len(rb_composition))
    std_last_performances = np.zeros(len(rb_composition))

    learning_curves = []
    aggregated_learning_curves = []
    std_learning_curves = []

    for i_prop in range(len(rb_composition)):
        if smooth_mode is not None:
            learning_curves.append(
                np.empty((nb_evals - smooth_window + 1, nb_envs, nb_seeds))
            )
        else:
            learning_curves.append(np.empty((nb_evals, nb_envs, nb_seeds)))

        for i_env in range(nb_envs):
            for i_seed in range(nb_seeds):
                learning_curve = performances[i_prop][:, i_env, i_seed]
                if smooth_mode is not None:
                    learning_curve = smooth(
                        learning_curve, window=smooth_window, mode=smooth_mode
                    )
                learning_curves[i_prop][:, i_env, i_seed] = learning_curve

        # NOTE: Choose aggregation: first on ENV and then on SEED dimension
        aggregated_learning_curves.append(learning_curves[i_prop].mean(1).mean(1))
        std_learning_curves.append(learning_curves[i_prop].mean(1).std(1))

        if step_to_take is not None:
            step_slice = slice(
                (step_to_take - last_steps) // eval_interval,
                step_to_take // eval_interval,
            )
        else:
            step_slice = slice(-last_steps // eval_interval, None)

        last_performances[i_prop] = aggregated_learning_curves[i_prop][
            step_slice
        ].mean()
        std_last_performances[i_prop] = std_learning_curves[i_prop][
            step_slice
        ].mean()  # TODO: better std pooling

    return (
        learning_curves,
        aggregated_learning_curves,
        std_learning_curves,
        last_performances,
        std_last_performances,
        step_slice,
    )


def plot_learning_curves(
    rb_composition_experiment,
    smooth_mode=None,
    smooth_window=3,
    plot_all_curves=False,
    plot_std=True,
    plot_margin=0.05,
    ax=None,
):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    rb_composition = rb_composition_experiment["rb_composition"]
    eval_interval = rb_composition_experiment["eval_interval"]
    rb_composition_type = rb_composition_experiment["type"]
    env_name = rb_composition_experiment["cfg"].gym_env.env_name
    nb_evals, nb_envs, nb_seeds = rb_composition_experiment["performances"][0].shape

    learning_curves, aggregated_learning_curves, std_learning_curves, *_ = (
        aggregate_learning_curves(rb_composition_experiment, smooth_mode, smooth_window)
    )

    colors = generate_colors(len(rb_composition))

    min_perf, max_perf = np.inf, -np.inf

    for i_prop, proportion in enumerate(rb_composition):
        if plot_all_curves:
            for i_env in range(nb_envs):
                for i_seed in range(nb_seeds):
                    learning_curve = learning_curves[i_prop][:, i_env, i_seed]

                    ax.plot(
                        (np.arange(len(learning_curve)) + 1) * eval_interval,
                        learning_curve,
                        color=colors[i_prop],
                        alpha=0.03,
                    )  # label = proportion if i_seed == 0 and i_env == 0 else None
        agg_perf = aggregated_learning_curves[i_prop]
        std_perf = std_learning_curves[i_prop]
        ax.plot(
            (np.arange(len(learning_curves[i_prop])) + 1) * eval_interval,
            agg_perf,
            color=colors[i_prop],
            label=proportion,
            alpha=1,
        )

        if plot_std:
            ax.fill_between(
                (np.arange(len(learning_curves[i_prop])) + 1) * eval_interval,
                agg_perf - std_perf,
                agg_perf + std_perf,
                color=colors[i_prop],
                # label=proportion,
                alpha=0.1,
            )

        min_perf = np.minimum(learning_curves[i_prop].min(), min_perf)
        max_perf = np.maximum(learning_curves[i_prop].max(), max_perf)
        y_lim_range = np.abs(max_perf - min_perf)
        ax.set_ylim(
            [
                min_perf - plot_margin * y_lim_range,
                max_perf + plot_margin * y_lim_range,  # type: ignore
            ]
        )
        ax.set_ylabel("evaluation performance")
        ax.set_xlabel("nb steps")


def plot_rb_compositions(
    rb_composition_experiment,
    smooth_mode=None,
    smooth_window=3,
    step_to_take=None,
    last_steps=1,
    ax=None,
    label=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    rb_composition = rb_composition_experiment["rb_composition"]
    eval_interval = rb_composition_experiment["eval_interval"]
    rb_composition_type = rb_composition_experiment["type"]
    env_name = rb_composition_experiment["cfg"].gym_env.env_name

    _, _, _, means, stds, step_slice = aggregate_learning_curves(
        rb_composition_experiment,
        smooth_mode,
        smooth_window,
        step_to_take,
        last_steps,
    )

    ax.plot(rb_composition, means, label=label)
    ax.fill_between(
        rb_composition,
        means - stds,
        means + stds,
        alpha=0.1,
    )

    ax.set_xlabel(
        "% of uniform exploration in replay buffer"
        if rb_composition_type == "uniform_proportions"
        else "truncated gaussian noise on policy actions"
    )
    ax.set_ylabel("evaluation performance (mean $\\pm$ std)")
