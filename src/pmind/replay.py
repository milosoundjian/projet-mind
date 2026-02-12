from functools import partial
from typing import Callable, Type

import numpy as np

from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from bbrl.workspace import Workspace
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl_utils.algorithms import EpochBasedAlgo

def get_gym_agent(env_name:str,num_envs:int, seed:int=42):
    '''
    Get multiple gymnasium environments as ParallelGymAgent with autoreset=True
    
    :param env_name: Name of the environment
    :type env_name: str
    :param num_envs: Number of parallel environments
    :type num_envs: int
    :param seed: Seed
    :type seed: int
    '''
    return ParallelGymAgent(partial(make_env, env_name=env_name, autoreset=True), 
                            num_envs=num_envs).seed(seed)

def get_workspace(policy_agent:Agent, gym_agent: ParallelGymAgent, epoch_size:int):
    '''
    Runs a policy on parallel gymnasium environments and returns the workspace.
    
    :param policy_agent: Policy agent
    :type policy_agent: Agent
    :param gym_agent: Parallel gymnasium environment agent
    :type gym_agent: ParallelGymAgent
    :param epoch_size: Epoch size (number of steps)
    :type epoch_size: int
    '''
    t_agents = TemporalAgent(Agents(gym_agent, policy_agent))
    workspace = Workspace() 
    t_agents(workspace, n_steps=epoch_size)
    return workspace

def get_random_transitions(workspace: Workspace, batch_size:int):
    '''
    Sample random transitions from a workspace.
    
    :param workspace: Workspace
    :type workspace: Workspace
    :param batch_size: Number of transitions to sample
    :type batch_size: int
    '''
    transitions = workspace.get_transitions()
    assert batch_size <= transitions.batch_size() # n _nv * nb_steps
    # TODO: check if does what we want: sample a single subworkspace,
    #  2 timesteps because we are sampling from transitions t, t+1
    # TODO: check if it's a sampling without replacement
    # TODO: set seed
    return transitions.sample_subworkspace(n_times=1,
                                           n_batch_elements=batch_size,
                                           n_timesteps=2)

def mix_transitions(workspace1: Workspace, workspace2: Workspace,buffer_size: int, proportion: float):
    '''
    Create a replay buffer based on a random mix of transitions 
    from 2 workspaces in a given proportion. 
    
    :param workspace1: First workspace
    :type workspace1: Workspace
    :param workspace2: Second workspace
    :type workspace2: Workspace
    :param batch_size: Number of transitions in final replay buffer
    :type batch_size: int
    :param proportion: Proportion of transitions coming from first workspace
    :type proportion: float
    '''
    # TODO: add possibility to set seed
    size1 = int(buffer_size*proportion)
    size2 = buffer_size - size1
    transitions1 = get_random_transitions(workspace1, size1)
    transitions2 = get_random_transitions(workspace2, size2)
    rb_mixed = ReplayBuffer(buffer_size)
    rb_mixed.put(transitions1)
    rb_mixed.put(transitions2)
    return rb_mixed

def test_rb_compositions(workspace_unif: Workspace,
                          workspace_best: Workspace,
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
        rb_mixed = mix_transitions(workspace_unif, 
                           workspace_best,
                           buffer_size=buffer_size, 
                           proportion=prop)
        offline_agent = agent_constructor(cfg)
        offline_run(offline_agent, rb_mixed)
        performances.append(np.array(offline_agent.eval_rewards))
    if plot:
        plot_perf_vs_rb_composition(proportions, performances)
    return performances