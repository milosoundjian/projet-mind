from functools import partial
from typing import Callable, Type

import numpy as np

from gymnasium import Wrapper
import gymnasium as gym

import torch

from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from bbrl.workspace import Workspace
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl_utils.algorithms import EpochBasedAlgo

from pmind.visualization import plot_perf_vs_rb_composition

def collect_policy_transitions(policy_agent:Agent, env_name: str, buffer_size: int):
    
    # TODO: is it okay to use only one parallel env?
    gym_agent = ParallelGymAgent(partial(make_env, env_name=env_name, autoreset=True), 
                            num_envs=1) # TODO: .seed(seed)
    
    t_agents = TemporalAgent(Agents(gym_agent, policy_agent))
    workspace = Workspace() 
    t_agents(workspace,t=0, n_steps=buffer_size)

    while workspace.time_size() // 2 < buffer_size:
        t_agents(workspace,t=workspace.time_size(), n_steps=1000)

    transitions = workspace.get_transitions()
    rb = ReplayBuffer(buffer_size)
    rb.put(transitions)
    return rb


class UniformWrapper(Wrapper):
    
    def uniform_reset(self):
        self.reset()
        self.state = self.unwrapped.state = self.observation_space.sample()
        return self.state
    def uniform_action(self):
        # TODO: doesn't work with "Pendulum-v1"
        return self.action_space.sample()
    # TODO: set seed
    # TODO: adapt these methods for environments with trickier states 
    # e.g. "LunarLander-v3"
    
def format_tensor(item, dtype=torch.float32):
    return torch.tensor(item, dtype=dtype).unsqueeze(0)

def collect_uniform_transitions(env_name: str, buffer_size: int):

    # NOTE: expected buffer composition and types:
    # env/env_obs : torch.float32
    # env/terminated : torch.bool
    # env/truncated : torch.bool
    # env/done : torch.bool
    # env/reward : torch.float32
    # env/cumulated_reward : torch.float32
    # env/timestep : torch.int64
    # action : torch.float32

    env = UniformWrapper(gym.make(env_name))
    rb = ReplayBuffer(buffer_size)
    for _ in range(buffer_size):

        workspace = Workspace()

        # TODO: what to do in case when we teleport into a terminal state directly?
        # it can be bad since uniform exploration becomes filled with garbage transitions
        state = env.uniform_reset()
        action = env.uniform_action()

        workspace.set("env/env_obs", 0, format_tensor(state))
        workspace.set("action", 0, format_tensor(action))
        workspace.set("env/timestep", 0, format_tensor(0, torch.int64))

        observation, reward, terminated, truncated, _ = env.step(action)


        workspace.set("env/env_obs", 1,  format_tensor(observation))
        workspace.set("env/terminated", 1, format_tensor(terminated, torch.bool))
        workspace.set("env/truncated", 1, format_tensor(truncated, torch.bool))
        workspace.set("env/done", 1,  format_tensor(terminated or truncated, torch.bool))
        workspace.set("env/reward", 1,format_tensor(reward))
        workspace.set("env/cumulated_reward", 1, format_tensor(reward))
        next_action = env.action_space.sample()
        workspace.set("action", 1, format_tensor(next_action))
        workspace.set("env/timestep", 1, format_tensor(1, torch.int64))

        rb.put(workspace.get_transitions())
    return rb


def mix_transitions(rb1: ReplayBuffer, rb2: ReplayBuffer,buffer_size: int, proportion: float):
    # TODO: add possibility to set seed
    size1 = int(buffer_size*proportion)
    size2 = buffer_size - size1
    transitions1 = rb1.get_shuffled(size1)
    transitions2 = rb2.get_shuffled(size2)
    rb_mixed = ReplayBuffer(buffer_size)
    rb_mixed.put(transitions1)
    rb_mixed.put(transitions2)
    return rb_mixed

def test_rb_compositions(rb_unif: ReplayBuffer,
                         rb_best: ReplayBuffer,
                        buffer_size: int,
                        proportions: list,
                        agent_constructor: Type[EpochBasedAlgo], 
                        cfg,
                        offline_run: Callable[[EpochBasedAlgo, ReplayBuffer], None],
                        plot = True
                        ):
    # TODO: do multiple seeds and then average!!
    performances = []
    for prop in proportions:
        rb_mixed = mix_transitions(rb_unif, 
                           rb_best,
                           buffer_size=buffer_size, 
                           proportion=prop)
        offline_agent = agent_constructor(cfg)
        offline_run(offline_agent, rb_mixed)
        performances.append(np.array(offline_agent.eval_rewards))
    if plot:
        plot_perf_vs_rb_composition(proportions, performances)
    return performances