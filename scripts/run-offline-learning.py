#!/usr/bin/env python3

import argparse
import torch
from omegaconf import OmegaConf

from pmind.replay import test_rb_uniform_proportion
from pmind.algorithms import TD3
from pmind.config.loader import load_config
import bbrl_gymnasium
import bbrl_utils

bbrl_utils.setup()


def main():
    parser = argparse.ArgumentParser(description="Run offline TD3 experiment")

    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument(
        "--proportion", type=float, required=True, help="Mix proportion"
    )
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument(
        "--exploit_reward", type=int, required=True, help="Exploit reward identifier"
    )

    args = parser.parse_args()

    ENV_NAME = args.env
    PROPORTION = args.proportion
    SEED = args.seed
    EXPLOIT_REWARD = args.exploit_reward

    # ---- constants ----
    BUFFER_SIZE = 200_000
    MAX_STEPS = 100_000
    NB_EVAL_ENVS = 5
    EVAL_INTERVAL = 100
    BATCH_SIZE = 64
    NN_ARCHITECTURE = [32, 16]

    # ---- setup ----
    bbrl_utils.setup()

    # ---- load replay buffers ----
    rb_unif = torch.load(
        f"/Vrac/21500050/projet-mind/results/models/{ENV_NAME}/rb-unif.pt",
        weights_only=False,
    )
    rb_exploit = torch.load(
        f"/Vrac/21500050/projet-mind/results/models/{ENV_NAME}/rb-{EXPLOIT_REWARD}.pt",
        weights_only=False,
    )

    # ---- config ----
    cfg = load_config("td3")
    cfg = OmegaConf.create(cfg.environments[ENV_NAME])
    cfg_offline = OmegaConf.create(cfg)

    cfg_offline.algorithm.n_steps = MAX_STEPS
    cfg_offline.algorithm.max_epochs = None

    cfg_offline.algorithm.batch_size = BATCH_SIZE
    cfg_offline.algorithm.architecture.actor_hidden_size = NN_ARCHITECTURE
    cfg_offline.algorithm.architecture.critic_hidden_size = NN_ARCHITECTURE

    cfg_offline.algorithm.eval_interval = EVAL_INTERVAL
    cfg_offline.algorithm.nb_evals = NB_EVAL_ENVS

    # offline-specific
    cfg_offline.algorithm.learning_starts = None
    cfg_offline.algorithm.action_noise = None
    cfg_offline.algorithm.target_policy_noise = None

    # ---- run ----
    result = test_rb_uniform_proportion(
        rb_unif,
        rb_exploit,
        BUFFER_SIZE,
        PROPORTION,
        TD3,
        cfg_offline,
        EXPLOIT_REWARD,
        SEED,
    )

    # ---- save ----
    output_path = f"/users/Etu0/21500050/results/result-{ENV_NAME}-{EXPLOIT_REWARD}-{PROPORTION}-{SEED}.pt"
    torch.save(result, output_path)

    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
