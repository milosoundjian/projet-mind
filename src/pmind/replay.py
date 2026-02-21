from functools import partial
from typing import Callable, Type
from enum import Enum

import numpy as np

from gymnasium import Wrapper
import gymnasium as gym

import torch

from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from bbrl.workspace import Workspace
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl_utils.algorithms import EpochBasedAlgo

from tqdm import trange

from pmind.visualization import plot_perf_vs_rb_composition


def collect_policy_transitions(policy_agent: Agent, env_name: str, buffer_size: int):

    # TODO: is it okay to use only one parallel env?
    gym_agent = ParallelGymAgent(
        partial(make_env, env_name=env_name, autoreset=True), num_envs=1
    )  # TODO: .seed(seed)

    t_agents = TemporalAgent(Agents(gym_agent, policy_agent))
    workspace = Workspace()
    t_agents(workspace, t=0, n_steps=buffer_size)

    while workspace.time_size() // 2 < buffer_size:
        t_agents(workspace, t=workspace.time_size(), n_steps=1000)

    transitions = workspace.get_transitions()
    rb = ReplayBuffer(buffer_size)
    rb.put(transitions)
    return rb

class SupportedEnv(Enum):
    CARTPOLE = "ContinuousCartPoleEnv"
    PENDULUM = "PendulumEnv"
    MOUNTAINCAR = "Continuous_MountainCarEnv"
    LUNARLANDER = "LunarLander"


class UniformExplorationWrapper(Wrapper):
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env_type = SupportedEnv(env.unwrapped.__class__.__name__)
        self.rejections = 0
    
    def is_valid_state(self, state):

        # TODO:  verify validity conditions

        if self.env_type == SupportedEnv.CARTPOLE:
            x, x_dot, theta, theta_dot = state
            x_threshold = 2.4
            theta_threshold = 12 * np.pi / 180  # radians

            if abs(x) > x_threshold:
                return False
            if abs(theta) > theta_threshold:
                return False
            return True
        elif self.env_type == SupportedEnv.PENDULUM:
            # never terminates
            return True
        elif self.env_type == SupportedEnv.MOUNTAINCAR:
            position, velocity = state
            goal_position = 0.45
            if position >= goal_position:
                return False
            return True
        elif self.env_type == SupportedEnv.LUNARLANDER:
            x, y, x_dot, y_dot, angle, angle_dot, left_leg, right_leg = state

            # Reject clearly aberrant states:
            # out of bounds
            if abs(x) > 1.0:
                return False
            # underground
            if y <= 0:
                return False
            return True
    

    def uniform_reset(self):
        # TODO: adapt this method for "LunarLander-v3" - observation != state there
        self.reset()
        while True:
            state = self.observation_space.sample()
            if self.is_valid_state(state):
                break
            self.rejections += 1
        #TODO: is self.unwrapped.state = state necessary?
        self.state = self.unwrapped.state = state
        return state


    def uniform_action(self):
        return self.action_space.sample()

    # TODO: set seed
    


# Vlad's version: 

# def collect_uniform_transitions_2(env_name: str, buffer_size: int):
#     def format_tensor(item, dtype=torch.float32):
#         return torch.tensor(item, dtype=dtype).unsqueeze(0)
#     # NOTE: expected buffer composition and types:
#     # env/env_obs : torch.float32
#     # env/terminated : torch.bool
#     # env/truncated : torch.bool
#     # env/done : torch.bool
#     # env/reward : torch.float32
#     # env/cumulated_reward : torch.float32
#     # env/timestep : torch.int64
#     # action : torch.float32

#     env = UniformExplorationWrapper(gym.make(env_name))
#     rb = ReplayBuffer(buffer_size)
#     for _ in range(buffer_size):
#         workspace = Workspace()

#         # TODO: what to do in case when we teleport into a terminal state directly?
#         # it can be bad since uniform exploration becomes filled with garbage transitions
#         state = env.uniform_reset()
#         action = env.uniform_action()

#         workspace.set("env/env_obs", 0, format_tensor(state))
#         workspace.set("action", 0, format_tensor(action))
#         workspace.set("env/timestep", 0, format_tensor(0, torch.int64))

#         observation, reward, terminated, truncated, _ = env.step(action)

#         workspace.set("env/env_obs", 1, format_tensor(observation))
#         workspace.set("env/terminated", 1, format_tensor(terminated, torch.bool))
#         workspace.set("env/truncated", 1, format_tensor(truncated, torch.bool))
#         workspace.set("env/done", 1, format_tensor(terminated or truncated, torch.bool))
#         workspace.set("env/reward", 1, format_tensor(reward))
#         workspace.set("env/cumulated_reward", 1, format_tensor(reward))
#         next_action = env.action_space.sample()
#         workspace.set("action", 1, format_tensor(next_action))
#         workspace.set("env/timestep", 1, format_tensor(1, torch.int64))

#         rb.put(workspace.get_transitions())
#     return rb


def collect_uniform_transitions(env_name: str, buffer_size: int = 100_000, print_rejections = True):
    env = UniformExplorationWrapper(gym.make(env_name))
    # Set up the replay buffer
    rb = ReplayBuffer(buffer_size)
    rb.variables = {}
    rb.variables["env/env_obs"] = torch.empty(
        [buffer_size, 2, env.observation_space.shape[0]], dtype=torch.float32
    )
    rb.variables["env/terminated"] = torch.empty([buffer_size, 2], dtype=torch.bool)
    rb.variables["env/truncated"] = torch.empty([buffer_size, 2], dtype=torch.bool)
    rb.variables["env/done"] = torch.empty([buffer_size, 2], dtype=torch.bool)
    rb.variables["env/reward"] = torch.empty([buffer_size, 2], dtype=torch.float32)
    rb.variables["env/cumulated_reward"] = torch.empty([buffer_size, 2], dtype=torch.float32)
    rb.variables["env/timestep"] = torch.empty([buffer_size, 2], dtype=torch.int64)
    # TODO: consider discrete case for action space
    rb.variables["action"] = torch.empty([buffer_size, 2, env.action_space.shape[0]])

    for i in trange(buffer_size):

        state_0 = env.uniform_reset()
        action = env.uniform_action()
        state_1, reward, terminated, truncated, _ = env.step(action)

        rb.variables["env/env_obs"][i, 0, :] = torch.tensor(state_0)
        rb.variables["env/env_obs"][i, 1, :] = torch.tensor(state_1)

        rb.variables["env/truncated"][i, 0] = torch.tensor(False)
        rb.variables["env/truncated"][i, 1] = torch.tensor(truncated)

        rb.variables["env/terminated"][i, 0] = torch.tensor(False)
        rb.variables["env/terminated"][i, 1] = torch.tensor(terminated)

        rb.variables["env/done"][i, 0] = torch.tensor(False)
        rb.variables["env/done"][i, 1] = torch.tensor(truncated or terminated)

        rb.variables["env/reward"][i, 0] = torch.tensor(0.0)
        rb.variables["env/reward"][i, 1] = torch.tensor(reward)

        rb.variables["env/cumulated_reward"][i, 0] = torch.tensor(0.0)
        rb.variables["env/cumulated_reward"][i, 1] = torch.tensor(reward)

        rb.variables["env/timestep"][i, 0] = torch.tensor(0)
        rb.variables["env/timestep"][i, 1] = torch.tensor(1)

        rb.variables["action"][i, 0, :] = torch.tensor(action)
        rb.variables["action"][i, 1, :] = torch.zeros_like(torch.tensor(action)) # dummy

    rb.is_full = True

    if print_rejections:
        print(f"{env.rejections} of proposed states were rejected")

    return rb


def mix_transitions(
    rb1: ReplayBuffer, 
    rb2: ReplayBuffer, 
    buffer_size: int, 
    proportion: float,
    seed = int
):
    # TODO: set seed actually!
    size1 = int(buffer_size * proportion)
    size2 = buffer_size - size1
    transitions1 = rb1.get_shuffled(size1)
    transitions2 = rb2.get_shuffled(size2)
    rb_mixed = ReplayBuffer(buffer_size)
    rb_mixed.put(transitions1)
    rb_mixed.put(transitions2)
    return rb_mixed


def test_rb_compositions(
    rb_unif: ReplayBuffer,
    rb_exploit: ReplayBuffer,
    buffer_size: int,
    proportions: list,
    agent_constructor: Type[EpochBasedAlgo],
    cfg,
    offline_run: Callable[[EpochBasedAlgo, ReplayBuffer], None],
    seeds = [1],
):
    performances = []
    for prop in proportions:
        algo = cfg.algorithm
        # TODO: actually less time-points, why???
        max_nb_timepoints = int(algo.n_steps * algo.max_epochs / algo.eval_interval)
        nb_envs = algo.nb_evals

        all_evals = np.empty((max_nb_timepoints,nb_envs, len(seeds)))

        for i, seed in enumerate(seeds):
            rb_mixed = mix_transitions(
                rb_unif, rb_exploit, buffer_size=buffer_size, proportion=prop, seed = seed
            )
            cfg.algorithm.seed = seed
            offline_agent = agent_constructor(cfg)
            offline_run(offline_agent, rb_mixed)
            current_evals = np.array(offline_agent.eval_rewards)
            nb_timepoints = current_evals.shape[0]
            all_evals[:nb_timepoints,:,i] = current_evals
            max_nb_timepoints = np.minimum(nb_timepoints, max_nb_timepoints)

        performances.append(all_evals[:max_nb_timepoints,:,:])
        
    return performances # proportions x time-point x env x seed
