from functools import partial
from typing import Callable, Type
from collections.abc import Iterable
from enum import Enum
import re
from pathlib import Path

import numpy as np

from gymnasium import Wrapper
import gymnasium as gym

import torch

from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from bbrl.workspace import Workspace
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl_utils.algorithms import EpochBasedAlgo

import d3rlpy

from tqdm.notebook import tqdm

from pmind.visualization import plot_perf_vs_rb_composition

from pmind.algorithms import TD3
from pmind.agents import AddTruncatedGaussianNoise


def collect_policy_transitions(
    policy_agent: Agent, env_name: str, buffer_size: int, action_noise=0.0
):

    # TODO: is it okay to use only one parallel env?
    gym_agent = ParallelGymAgent(
        partial(make_env, env_name=env_name, autoreset=True), num_envs=1
    )  # TODO: .seed(seed)

    if action_noise != 0:
        noise_agent = AddTruncatedGaussianNoise(
            action_noise=action_noise, action_space=gym_agent.action_space
        )
        t_agents = TemporalAgent(Agents(gym_agent, policy_agent, noise_agent))
    else:
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

        # TODO:  verify validity conditions with Gymnasium pages on each env

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
            goal_position = 0.5
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
        initial_state, _ = self.reset()

        if self.env_type == SupportedEnv.PENDULUM:
            state = self.observation_space.sample()
            cos_th, sin_th, th_dot = state
            th = np.arctan2(sin_th, cos_th)
            self.state = self.unwrapped.state = np.array([th, th_dot])

        elif self.env_type == SupportedEnv.LUNARLANDER:
            rng = self.unwrapped.np_random
            body = self.unwrapped.lander
            legs = self.unwrapped.legs

            # Do a uniform translation in space
            # TODO: set better bounds
            dx = rng.uniform(-5, 5)  # tuned maybe
            dy = rng.uniform(-10, 0)

            # Move body
            body.position = (body.position[0] + dx, body.position[1] + dy)

            # Move legs with same translation
            for leg in legs:
                leg.position = (leg.position[0] + dx, leg.position[1] + dy)

            vx = rng.uniform(-1, 1)
            vy = rng.uniform(-1, 0)
            vth = rng.uniform(-1, 1)

            body.linearVelocity = (vx, vy)
            for leg in legs:
                leg.linearVelocity = (vx, vy)

            # Do a few random steps to get a bit of randomness
            for _ in range(
                20
            ):  # Tune this maybe? This costs performance for collecting the uniform replay buffer though
                self.step([0, 0])

            # TODO: body and both legs rotate independently, trickier...
            # th = rng.uniform(-0.2, 0.2)
            # body.angle = th
            # for leg in legs:
            #     leg.angle = th

            # body.angularVelocity = vth
            # for leg in legs:
            #     leg.angularVelocity = vth
            state, *_ = self.step(
                [0, 0]
            )  # do nothing, just to detect whether ground was touched

        else:
            while True:
                state = self.observation_space.sample()
                if self.is_valid_state(state):
                    break
                self.rejections += 1

            self.unwrapped.state = state

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


def collect_uniform_transitions(
    env_name: str, buffer_size: int = 100_000, print_rejections=True
):
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
    rb.variables["env/cumulated_reward"] = torch.empty(
        [buffer_size, 2], dtype=torch.float32
    )
    rb.variables["env/timestep"] = torch.empty([buffer_size, 2], dtype=torch.int64)
    # TODO: consider discrete case for action space
    rb.variables["action"] = torch.empty([buffer_size, 2, env.action_space.shape[0]])

    for i in tqdm(range(buffer_size), desc="Uniform transitions"):
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
        rb.variables["action"][i, 1, :] = torch.zeros_like(
            torch.tensor(action)
        )  # dummy

    rb.is_full = True

    if print_rejections:
        print(f"{env.rejections} of proposed states were rejected")

    return rb


def mix_transitions(
    rb1: ReplayBuffer, rb2: ReplayBuffer, buffer_size: int, proportion: float, seed=int
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


def test_rb_uniform_proportions(
    rb_unif: ReplayBuffer,
    rb_exploit: ReplayBuffer,
    buffer_size: int,
    proportions: Iterable,
    agent_constructor: Type[TD3],  # TODO: fow now only TD3 supports offline
    cfg,
    seeds=[1],
    device=torch.device("cpu"),
):
    performances = []
    policies = []
    replay_buffers = []
    for prop in proportions:
        algo = cfg.algorithm
        max_nb_timepoints = int(algo.n_steps / algo.eval_interval)
        nb_envs = algo.nb_evals

        all_evals = np.empty((max_nb_timepoints, nb_envs, len(seeds)))

        policies.append([])
        replay_buffers.append([])

        for i, seed in enumerate(seeds):
            # Set the seeds
            np.random.seed(seed)
            torch.manual_seed(seed)

            rb_mixed = mix_transitions(
                rb_unif, rb_exploit, buffer_size=buffer_size, proportion=prop, seed=seed
            )
            cfg.algorithm.seed = seed
            offline_agent = agent_constructor(cfg, offline=True).to(device)
            offline_agent.train(rb_mixed)
            policies[-1].append(offline_agent.policies)
            replay_buffers[-1].append(rb_mixed)
            current_evals = np.array(offline_agent.eval_rewards)
            nb_timepoints = current_evals.shape[0]
            all_evals[:nb_timepoints, :, i] = current_evals
            # correct the size of the resulting array
            max_nb_timepoints = np.minimum(nb_timepoints, max_nb_timepoints)

        performances.append(all_evals[:max_nb_timepoints, :, :])

    return {
        "performances": performances,  # proportions x time-point x env x seed
        "buffer_size": buffer_size,
        "rb_composition": proportions,
        "eval_interval": cfg.algorithm.eval_interval,
        "cfg": cfg,
        "seeds": seeds,
        "type": "uniform_proportions",
        "n_steps": cfg.algorithm.n_steps,
        "save_rb_policy_interval": cfg.algorithm.get("save_rb_policy_interval"),
        "policies": policies,  # proportions x seed x time-point
        "replay_buffers": replay_buffers,  # proportions x seed
    }


def test_rb_uniform_proportion(
    rb_unif: ReplayBuffer,
    rb_exploit: ReplayBuffer,
    buffer_size: int,
    proportion: float,
    agent_constructor: Type[TD3],
    cfg,
    exploit_reward: int,
    seed: int = 1,
    device=torch.device("cpu"),
):
    algo = cfg.algorithm
    max_nb_timepoints = int(algo.n_steps / algo.eval_interval)
    nb_envs = algo.nb_evals

    # Set the seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    rb_mixed = mix_transitions(
        rb_unif,
        rb_exploit,
        buffer_size=buffer_size,
        proportion=proportion,
        seed=seed,
    )

    cfg.algorithm.seed = seed
    offline_agent = agent_constructor(cfg, offline=True).to(device)
    offline_agent.train(rb_mixed)

    current_evals = np.array(offline_agent.eval_rewards)
    nb_timepoints = min(current_evals.shape[0], max_nb_timepoints)

    performances = current_evals[:nb_timepoints, :]  # time-point x env

    return {
        "performances": performances,  # time-point x env
        "buffer_size": buffer_size,
        "rb_composition": proportion,
        "eval_interval": cfg.algorithm.eval_interval,
        "cfg": cfg,
        "exploit_reward": exploit_reward,
        "seed": seed,
        "type": "uniform_proportion",
    }


def test_rb_noise_levels(
    rb_by_noise: dict[float, ReplayBuffer],
    buffer_size: int,
    agent_constructor: Type[TD3],
    cfg,
    seeds=[1],
    device=torch.device("cpu"),
):
    action_noises = []
    performances = []
    policies = []
    replay_buffers = []
    for action_noise, rb_noise in sorted(rb_by_noise.items()):
        algo = cfg.algorithm
        max_nb_timepoints = int(algo.n_steps / algo.eval_interval)
        nb_envs = algo.nb_evals

        all_evals = np.empty((max_nb_timepoints, nb_envs, len(seeds)))

        policies.append([])
        replay_buffers.append([])

        for i, seed in enumerate(seeds):
            # Set the seeds
            np.random.seed(seed)
            torch.manual_seed(seed)

            cfg.algorithm.seed = seed
            offline_agent = agent_constructor(cfg, offline=True).to(device)
            offline_agent.train(rb_noise)

            policies[-1].append(offline_agent.policies)
            replay_buffers[-1].append(rb_noise)

            current_evals = np.array(offline_agent.eval_rewards)
            nb_timepoints = current_evals.shape[0]
            all_evals[:nb_timepoints, :, i] = current_evals
            # correct the size of the resulting array
            max_nb_timepoints = np.minimum(nb_timepoints, max_nb_timepoints)

        performances.append(all_evals[:max_nb_timepoints, :, :])
        action_noises.append(action_noise)

    return {
        "performances": performances,  # proportions x time-point x env x seed
        "buffer_size": buffer_size,
        "rb_composition": action_noises,
        "eval_interval": cfg.algorithm.eval_interval,
        "cfg": cfg,
        "seeds": seeds,
        "type": "noise_levels",
        "n_steps": cfg.algorithm.n_steps,
        "save_rb_policy_interval": cfg.algorithm.get("save_rb_policy_interval"),
        "policies": policies,  # proportions x seed x time-point
        "replay_buffers": replay_buffers,  # proportions x seed
    }


def test_rb_noise_level(
    rb: ReplayBuffer,
    action_noise,
    buffer_size: int,
    agent_constructor: Type[TD3],
    cfg,
    seed: int,
    device=torch.device("cpu"),
):
    algo = cfg.algorithm
    max_nb_timepoints = int(algo.n_steps / algo.eval_interval)
    nb_envs = algo.nb_evals

    # Set the seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg.algorithm.seed = seed
    offline_agent = agent_constructor(cfg, offline=True).to(device)
    offline_agent.train(rb)

    current_evals = np.array(offline_agent.eval_rewards)
    nb_timepoints = min(current_evals.shape[0], max_nb_timepoints)

    performances = current_evals[:nb_timepoints, :]  # time-point x env

    return {
        "performances": performances,
        "buffer_size": buffer_size,
        "rb_composition": action_noise,
        "eval_interval": cfg.algorithm.eval_interval,
        "cfg": cfg,
        "seed": seed,
        "type": "noise_levels",
    }


def load_rb_files(directory, action_noises=[0.0]):
    directory = Path(directory)
    pattern = re.compile(r"^rb-(-?\d+)-noise-(\d*\.\d+|\d+)\.pt$")

    result = {}

    for file in directory.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                reward = int(match.group(1))
                action_noise = float(match.group(2))
                if action_noise in action_noises:
                    if reward not in result:
                        result[reward] = {}
                    result[reward][action_noise] = torch.load(file, weights_only=False)

    return result


def convert_rb_to_dataset(rb, contains_teleportation):
    """Convert ReplayBuffer from BBRL to dataset (replay buffer) from d3rlpy"""

    observations = rb.variables["env/env_obs"].numpy()
    actions = rb.variables["action"].numpy()
    rewards = rb.variables["env/reward"].numpy()
    terminals = rb.variables["env/terminated"].numpy()
    timeouts = rb.variables["env/truncated"].numpy()
    nb_transitions = rb.size()
    
    # NOTE: for their dataset, r1 = r(s1, a1) and t1 = t(s1, a1),
    # so rewards and terminals are shifted by 1 compared to BBRL

    if not contains_teleportation:
        observations = np.array([transition[0] for transition in observations])
        actions = np.array([transition[0] for transition in actions])

        rewards = np.array([transition[1] for transition in rewards])
        terminals = np.array([transition[1] for transition in terminals])
        timeouts = np.array([transition[1] for transition in timeouts])
    else:
        observations = observations.reshape(nb_transitions*2, -1)
        actions = actions.reshape(nb_transitions*2, -1)
        
        # rewards = rewards.reshape(nb_transitions*2)
        # terminals = terminals.reshape(nb_transitions*2)
        
        rewards = np.array([transition[::-1] for transition in rewards]).reshape(nb_transitions*2)
        terminals = np.array([transition[::-1] for transition in terminals]).reshape(nb_transitions*2)
        # timeouts = np.array([transition[::-1] for transition in timeouts]).reshape(nb_transitions*2)
        timeouts = (np.arange(nb_transitions*2) %2) 

    timeouts = timeouts & (~terminals)  # if terminated, then it's not timeout
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )
    return dataset
