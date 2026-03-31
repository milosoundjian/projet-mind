import torch
from omegaconf import OmegaConf


import time
import datetime
import os
import sys

import bbrl_utils

import bbrl_gymnasium

from pmind.algorithms import TD3

from pmind.replay import (
    test_rb_uniform_proportions,
    test_rb_noise_levels
)

from pmind.config.loader import load_config

bbrl_utils.setup()


OUTPUT_DIR = "/tempory/rl_env/results"
BUFFER_DIR = "/tempory/rl_env/models"

# OUTPUT_DIR = "/Users/vlad/Documents/University/Master-MIND/projet-mind/experiments/results/test"
# BUFFER_DIR = "/Users/vlad/Documents/University/Master-MIND/projet-mind/experiments/models"

device_type = sys.argv[1] if len(sys.argv) > 1 else None
assert device_type == "cpu" or device_type == "cuda" and torch.cuda.is_available(), "Invalid device"

device = torch.device(device_type)

BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else None
assert isinstance(BATCH_SIZE, int), "Invalid batch size"

print(f"Running on: {device.type}")
print(f"Batch size: {BATCH_SIZE}")

ENV_NAMES = (
    "CartPoleContinuous-v1",
    # "Pendulum-v1",
    # "MountainCarContinuous-v0",
    # "LunarLanderContinuous-v3",
)
BUFFER_SIZE = 1000#200_000
BEST_ONLY = True
TEST_RB_COMPOSITIONS = True
TEST_NOISE_LEVELS = False
PROPORTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [0, 1, 2, 3, 4]
# NOTE: set small values for test:
MAX_STEPS = 10_000#100_000#15_000
NB_EVAL_ENVS = 5#10
EVAL_INTERVAL = 100
NN_ARCHITECTURE = [400,300]

def main():

    for ENV_NAME in ENV_NAMES:
        print(f"Testing {ENV_NAME}:")
        cfg = load_config("td3")
        cfg = OmegaConf.create(cfg.environments[ENV_NAME])

        best_reward, _ = torch.load(
            f"{BUFFER_DIR}/{ENV_NAME}/best-policy.pt", weights_only=False
        )

        # Load replay buffers from file
        rb_exploit = torch.load(f"{BUFFER_DIR}/{ENV_NAME}/rb-exploit.pt", weights_only=False)
        rb_unif = torch.load(f"{BUFFER_DIR}/{ENV_NAME}/rb-unif.pt", weights_only=False)

        cfg_offline = OmegaConf.create(cfg)

        cfg_offline.algorithm.n_steps = MAX_STEPS
        cfg_offline.algorithm.max_epochs = None

        cfg_offline.algorithm.batch_size = BATCH_SIZE
        cfg_offline.algorithm.architecture.actor_hidden_size = NN_ARCHITECTURE
        cfg_offline.algorithm.architecture.critic_hidden_size = NN_ARCHITECTURE
        

        cfg_offline.algorithm.eval_interval = EVAL_INTERVAL
        cfg_offline.algorithm.nb_evals = NB_EVAL_ENVS  # nb of evaluation envs in parallel

        # learning starts immediately for offline:
        cfg_offline.algorithm.learning_starts = None

        # there is no exploration in offline learning
        cfg_offline.algorithm.action_noise = None
        cfg_offline.algorithm.target_policy_noise = None
        
        if TEST_RB_COMPOSITIONS:
            print("TESTING UNIFORM PROPORTIONS")
            for reward, rb_by_noise in sorted(rb_exploit.items(), reverse=True):
                print(f"Policy with reward {best_reward} :")
                performances = test_rb_uniform_proportions(
                    rb_unif,
                    rb_by_noise[0],
                    BUFFER_SIZE,
                    PROPORTIONS,
                    TD3,
                    cfg_offline,
                    SEEDS,
                    device
                )
                torch.save(performances, f"{OUTPUT_DIR}/uniform_proportions-{ENV_NAME}-scoring-{reward:.0f}")
                
                if BEST_ONLY:
                    print("Skipped intermediate policies")
                    break
        
        if TEST_NOISE_LEVELS:
            print("TESTING NOISE LEVELS")
            for reward, rb_by_noise in sorted(rb_exploit.items(), reverse=True):
                print(f"Policy with reward {best_reward} :")
                performances = test_rb_noise_levels(
                    rb_by_noise,
                    BUFFER_SIZE,
                    TD3,
                    cfg_offline,
                    SEEDS,
                    device
                ) 
                torch.save(performances, f"{OUTPUT_DIR}/noise_levels-{ENV_NAME}-scoring-{reward:.0f}")
                
                if BEST_ONLY:
                    print("Skipped intermediate policies")
                    break

if __name__ == "__main__":
    main()

