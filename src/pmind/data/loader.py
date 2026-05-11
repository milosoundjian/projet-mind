from . import paths
import re
import os
import glob
import numpy as np
import torch

from . import paths

from collections import defaultdict


def nested_dict():
    return defaultdict(nested_dict)


def to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: to_dict(v) for k, v in d.items()}
    return d


def load_buffers(env_name, type_):
    buffers = nested_dict()
    env_dir = paths.MODELS_DIR / env_name
    if type_ == "unif":
        pattern = r"rb-unif\.pt"

    elif type_ == "exploit":
        pattern = r"rb-(?P<reward>-?\d+)\.pt"

    elif type_ == "action":
        pattern = r"rb-(?P<reward>-?\d+)-noise-(?P<noise>\d+(?:\.\d+)?)\.pt"

    elif type_ == "branch":
        pattern = r"rb-(?P<reward>-?\d+)-noise-(?P<noise>\d+(?:\.\d+)?)-d(?P<branch_depth>\d+)-k(?P<n_branches>\d+)\.pt"

    else:
        raise AttributeError("Type can be only: unif, exploit, action, branch")
    pattern = re.compile(pattern)
    for file in env_dir.glob("rb-*.pt"):
        match = pattern.fullmatch(file.name)
        if match:
            info = match.groupdict()
            reward = info.get("reward", None)
            noise = info.get("noise", None)
            branch_depth = info.get("branch_depth", None)
            n_branches = info.get("n_branches", None)

            buffer = torch.load(file, weights_only=False)
            if type_ == "unif":
                buffers = buffer
            elif type_ == "exploit":
                buffers[reward] = buffer
            elif type_ == "action":
                buffers[reward][noise] = buffer
            elif type_ == "branch":
                buffers[reward][noise][branch_depth][n_branches] = buffer
    return to_dict(buffers)


def load_all_buffers(env_name):
    buffers = {}
    for type_ in ("unif", "exploit", "action", "branch"):
        buffers[type_] = load_buffers(env_name, type_=type_)
    return buffers


def load_experiment_results(
    env_names,
    input_dir,
    from_single_experiments=True,
    seeds_to_exclude=[],
    verbose=False,
):

    if from_single_experiments:
        pattern = re.compile(
            r"result-(?P<env_name>.+-v\d+)-(?P<reward>-?\d+)-(?P<prop>\d+\.\d+)-(?P<seed>\d+)\.pt"
        )

        all_performances = {}
        for rb_composition_type in ("uniform_proportions", "noise_levels"):
            if verbose:
                print("Collecting", rb_composition_type)
            all_performances[rb_composition_type] = {}
            for env_name in env_names:
                all_performances[rb_composition_type][env_name] = {}
                results_found = False
                seeds = set()
                proportions = set()
                exploit_rewards = set()
                for dirname in os.listdir(
                    input_dir / f"results-{rb_composition_type}/"
                ):
                    dir_seeds = {}
                    if env_name in dirname:
                        if verbose:
                            print(dirname)
                        for fname in os.listdir(
                            input_dir / f"results-{rb_composition_type}/" / dirname
                        ):
                            match = pattern.search(fname)
                            if match:
                                results_found = True
                                info = match.groupdict()
                                reward = info["reward"]
                                prop = info["prop"]
                                seed = info["seed"]
                                exploit_rewards.add(reward)
                                proportions.add(prop)
                                if (reward, prop) not in dir_seeds:
                                    dir_seeds[(reward, prop)] = set()
                                dir_seeds[(reward, prop)].add(seed)

                        for single_experiment_seeds in dir_seeds.values():
                            if len(seeds) == 0:
                                seeds = single_experiment_seeds
                            else:
                                seeds &= single_experiment_seeds

                if results_found:
                    seeds = sorted(seeds)
                    if verbose:
                        print(f"{len(seeds)} seeds kept")
                    if seeds_to_exclude:
                        seeds = [seed for seed in seeds if seed not in seeds_to_exclude]
                    proportions = sorted(proportions)
                    exploit_rewards = sorted(exploit_rewards)

                    for exploit_reward in exploit_rewards:
                        dirname = f"{input_dir}/results-{rb_composition_type}/results-{env_name}-{exploit_reward}"
                        d_ref = torch.load(
                            f"{dirname}/result-{env_name}-{exploit_reward}-{proportions[0]}-{seeds[0]}.pt",
                            weights_only=False,
                        )
                        test_log = {
                            "performances": None,
                            "buffer_size": d_ref["buffer_size"],
                            "rb_composition": proportions,
                            "eval_interval": d_ref["eval_interval"],
                            "cfg": d_ref["cfg"],
                            "seeds": seeds,
                            "type": d_ref["type"] + "s",
                        }
                        # TODO: add asserts that all files have consistent metadata
                        test_log["performances"] = [
                            np.stack(
                                [
                                    torch.load(
                                        f"{dirname}/result-{env_name}-{exploit_reward}-{proportion}-{seed}.pt",
                                        weights_only=False,
                                    )["performances"]  # [0].reshape(-1,1)
                                    for seed in seeds
                                ],
                                -1,
                            )
                            for proportion in proportions
                        ]
                        all_performances[rb_composition_type][env_name][
                            exploit_reward
                        ] = test_log
    else:
        all_performances = {}
        for rb_composition_type in ("uniform_proportions", "noise_levels"):
            all_performances[rb_composition_type] = {}
            for env_name in env_names:
                all_performances[rb_composition_type][env_name] = {}
                if verbose:
                    print(f"{env_name}:")
                for intermediate_path in glob.glob(
                    f"{input_dir}/{rb_composition_type}-{env_name}-scoring-*"
                ):
                    exploit_performance = float(intermediate_path.split("-scoring-")[1])
                    if verbose:
                        print(f"    {exploit_performance}")
                    all_performances[rb_composition_type][env_name][
                        exploit_performance
                    ] = torch.load(intermediate_path, weights_only=False)
    return all_performances


def load_all_experiments(
    runs_dir=paths.RUNS_DIR,
    env_names=["CartPoleContinuous-v1", "Pendulum-v1", "MountainCarContinuous-v0"],
    experiments=None,
    verbose=False,
):
    if experiments is None:
        experiments = [
            p.name for p in runs_dir.iterdir() if p.is_dir() and p.name != "tmp"
        ]

    all_logs = nested_dict()
    for experiment in experiments:
        print(f"EXPERIMENT: {experiment}")
        library, algo, unif, noise, length = experiment.split("-")
        experiments_logs = load_experiment_results(
            env_names=env_names,
            input_dir=runs_dir / experiment,
            from_single_experiments=True,
            seeds_to_exclude=[],
            verbose=verbose,
        )

        for rb_type in ("unif", "action", "branch"):
            if rb_type not in all_logs[length][library][algo]:
                all_logs[length][library][algo][rb_type] = None

        if unif:
            all_logs[length][library][algo][unif] = experiments_logs[
                "uniform_proportions"
            ]

        if noise:
            all_logs[length][library][algo][noise] = experiments_logs["noise_levels"]

    return to_dict(all_logs)
