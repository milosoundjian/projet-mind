import copy
import math
from functools import partial

from typing import Callable, Type
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.distributions import Normal

import numpy as np


from bbrl_utils.notebook import tqdm

import bbrl_gymnasium
from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl.visu.plot_policies import plot_policy
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from bbrl.workspace import Workspace
from bbrl_utils.nn import (
    build_mlp,
    setup_optimizer,
    copy_parameters,
    soft_update_params,
)

# ============== DQN & DDQN ===============

class DiscreteQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        q_values = self.model(obs)
        self.set((f"{self.prefix}q_values", t), q_values)

class ArgmaxActionSelector(Agent):
    """BBRL agent that selects the best action based on Q(s,a)"""

    def forward(self, t: int, **kwargs):
        q_values = self.get(("q_values", t))
        action = q_values.argmax(-1)
        self.set(("action", t), action)


class EGreedyActionSelector(Agent):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, t: int, **kwargs):
        # Retrieves the q values
        # (matrix nb. of episodes x nb. of actions)
        q_values: torch.Tensor = self.get(("q_values", t))
        size, nb_actions = q_values.shape

        # Flag
        is_random = torch.rand(size) < self.epsilon

        # Actions (random / argmax)
        random_action = torch.randint(nb_actions, size=(size,))
        max_action = q_values.argmax(-1)

        # Choose the action based on the is_random flag
        action = torch.where(is_random, random_action, max_action)

        # Sets the action at time t
        self.set(("action", t), action)

class DQN(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        # Get the two agents (critic and target critic)
        critic = DiscreteQAgent(
            obs_size, cfg.algorithm.architecture.hidden_size, act_size
        )
        target_critic = copy.deepcopy(critic).with_prefix("target/")

        # Builds the train agent that will produce transitions
        explorer = EGreedyActionSelector(cfg.algorithm.epsilon)
        self.train_policy = Agents(critic, explorer)

        self.eval_policy = Agents(critic, ArgmaxActionSelector())

        # Creates two temporal agents just for "replaying" some parts
        # of the transition buffer
        self.t_q_agent = TemporalAgent(critic)
        self.t_target_q_agent = TemporalAgent(target_critic)

        # Get an agent that is executed on a complete workspace
        self.optimizer = setup_optimizer(cfg.optimizer, self.t_q_agent)

        self.last_critic_update_step = 0

def dqn_compute_critic_loss(
    cfg, reward, must_bootstrap, q_values, target_q_values, action
):
    """Compute the critic loss

    :param reward: The reward $r_t$ (shape 2 x B)
    :param must_bootstrap: The must bootstrap flag at $t+1$ (shape 2 x B)
    :param q_values: The Q-values (shape 2 x B x A)
    :param target_q_values: The target Q-values (shape 2 x B x A)
    :param action: The chosen actions (shape 2 x B)
    :return: _description_
    """
    
    # Implement the DQN loss

    # Adapt from the previous notebook and adapt to our case (target Q network)
    # Don't forget that we deal with transitions (and not episodes)
    
    gamma = cfg.algorithm.discount_factor
    
    qvals = q_values[0].gather(dim=1, index=action[0].unsqueeze(-1)).squeeze(-1)

    # NOTE: use target Q fn (updated occasionally) to 
    # compute maximal (next) Q, i.e. it is used for
    # BOTH next action selection and getting corresponding Q
    max_q = target_q_values[1].amax(-1).detach() 
    target = reward[0] + gamma * max_q * must_bootstrap[1]

    # Compute critic loss (no need to use must_bootstrap here since we are dealing with "full" transitions)
    mse = nn.MSELoss()
    critic_loss = mse(target, qvals)

    return critic_loss

def ddqn_compute_critic_loss(
    cfg, reward, must_bootstrap, q_values, target_q_values, action
):
    """Compute the critic loss

    :param reward: The reward $r_t$ (shape 2 x B)
    :param must_bootstrap: The must bootstrap flag at $t+1$ (shape 2 x B)
    :param q_values: The Q-values (shape 2 x B x A)
    :param target_q_values: The target Q-values (shape 2 x B x A)
    :param action: The chosen actions (shape 2 x B)
    :return: the loss (a scalar)
    """

    # Implement the double DQN loss

    gamma = cfg.algorithm.discount_factor
    
    qvals = q_values[0].gather(dim=1, index=action[0].unsqueeze(-1)).squeeze(-1)

    # NOTE: double DQN uses target Q fn (updated occasionally) ONLY
    # to get the Q of maximum action
    # BUT maximum action is chosen based on online Q function
    max_action = q_values.argmax(-1).detach() 
    max_q = target_q_values[1].gather(dim=-1, index=max_action[0].unsqueeze(-1)).squeeze(-1)
    # NOTE: equivalent to gather:
    # max_q = target_q_values[1, torch.arange(target_q_values.shape[1]), max_action].detach()

    target = reward[0] + gamma * max_q * must_bootstrap[1]

    # Compute critic loss (no need to use must_bootstrap here since we are dealing with "full" transitions)
    mse = nn.MSELoss()
    critic_loss = mse(target, qvals)

    return critic_loss

def run_dqn(dqn: DQN, compute_critic_loss):
    for rb in dqn.iter_replay_buffers():
        for _ in range(dqn.cfg.algorithm.n_updates):
            rb_workspace = rb.get_shuffled(dqn.cfg.algorithm.batch_size)

            # The q agent needs to be executed on the rb_workspace workspace
            dqn.t_q_agent(rb_workspace, t=0, n_steps=2)
            with torch.no_grad():
                dqn.t_target_q_agent(rb_workspace, t=0, n_steps=2)

            q_values, terminated, reward, action, target_q_values = rb_workspace[
                "q_values", "env/terminated", "env/reward", "action", "target/q_values"
            ]

            # Determines whether values of the critic should be propagated
            must_bootstrap = ~terminated

            # Compute critic loss
            critic_loss = compute_critic_loss(
                dqn.cfg, reward, must_bootstrap, q_values, target_q_values, action
            )
            # Store the loss for tensorboard display
            dqn.logger.add_log("critic_loss", critic_loss, dqn.nb_steps)
            dqn.logger.add_log(
                "q_values/min", q_values.max(-1).values.min(), dqn.nb_steps
            )
            dqn.logger.add_log(
                "q_values/max", q_values.max(-1).values.max(), dqn.nb_steps
            )
            dqn.logger.add_log(
                "q_values/mean", q_values.max(-1).values.mean(), dqn.nb_steps
            )

            dqn.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dqn.t_q_agent.parameters(), dqn.cfg.algorithm.max_grad_norm
            )
            dqn.optimizer.step()

            # Update target
            if (
                dqn.nb_steps - dqn.last_critic_update_step
                > dqn.cfg.algorithm.target_critic_update
            ):
                dqn.last_critic_update_step = dqn.nb_steps
                copy_parameters(dqn.t_q_agent, dqn.t_target_q_agent)

            dqn.evaluate()

# ============== DDPG & TD3 ===============

class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        # Get the current state $s_t$ and the chosen action $a_t$
        obs = self.get(("env/env_obs", t))  # shape B x D_{obs}
        action = self.get(("action", t))  # shape B x D_{action}

        # Compute the Q-value(s_t, a_t)
        obs_act = torch.cat((obs, action), dim=1)  # shape B x (D_{obs} + D_{action})
        # Get the q-value (and remove the last dimension since it is a scalar)
        q_value = self.model(obs_act).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)

class ContinuousDeterministicActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(
            layers, activation=nn.ReLU(), output_activation=nn.Tanh()
        )

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        action = self.model(obs)
        self.set(("action", t), action)


class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        self.set(("action", t), action)

class AddOUNoise(Agent):
    """
    Ornstein-Uhlenbeck process noise for actions as suggested by DDPG paper
    """

    def __init__(self, std_dev, theta=0.15, dt=1e-2):
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = 0

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
        )
        self.x_prev = x
        self.set(("action", t), x)

def compute_critic_loss(
    cfg,
    reward: torch.Tensor,
    must_bootstrap: torch.Tensor,
    q_values: torch.Tensor,
    target_q_values: torch.Tensor,
):
    """Compute the DDPG critic loss from a sample of transitions

    :param cfg: The configuration
    :param reward: The reward (shape 2xB)
    :param must_bootstrap: Must bootstrap flag (shape 2xB)
    :param q_values: The computed Q-values (shape 2xB)
    :param target_q_values: The Q-values computed by the target critic (shape 2xB)
    :return: the loss (a scalar)
    """
    # Compute temporal difference
    # [[STUDENT]]...

    gamma = cfg.algorithm.discount_factor

    # NOTE: target_q_values are computed based on critic of the next action
    # proposed by actor
    target = reward[0] + gamma * target_q_values[1] * must_bootstrap[1]

    mse = nn.MSELoss()
    critic_loss = mse(target, q_values[0])

    return critic_loss

def compute_actor_loss(q_values):
    """Returns the actor loss

    :param q_values: The q-values (shape 2xB)
    :return: A scalar (the loss)
    """
    return - q_values[0].mean() # start or end of transition, or both?


class DDPG(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        # we create the critic and the actor, but also an exploration agent to
        # add noise and a target critic. The version below does not use a target
        # actor as it proved hard to tune, but such a target actor is used in
        # the original paper.

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        self.critic = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic/")
        self.target_critic = copy.deepcopy(self.critic).with_prefix("target-critic/")

        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )

        # As an alternative, you can use `AddOUNoise`
        noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)

        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor # NOTE: pure exploitation for evaluation

        # [[STUDENT]]...
        self.t_q_agent = TemporalAgent(self.critic)
        self.t_target_q_agent = TemporalAgent(self.target_critic)
        self.t_actor_agent = TemporalAgent(self.actor)


        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic)


def run_ddpg(ddpg: DDPG):
    for rb in ddpg.iter_replay_buffers():
        # Get a sample of transitions (shapes 2x...) from the replay buffer
        rb_workspace = rb.get_shuffled(ddpg.cfg.algorithm.batch_size)

        # Compute the critic loss

        # Critic update
        # Compute critic loss
        # [[STUDENT]]...
        ddpg.t_q_agent(rb_workspace, t=0, n_steps=1) # evaluate actions from rb with online Q
        # better performance without this line: why????
        # ddpg.t_actor_agent(rb_workspace, t=1, n_steps=1) # do actor's actions
        with torch.no_grad():
            ddpg.t_target_q_agent(rb_workspace, t=1, n_steps=1) # evaluate actor's action with target Q

        reward, terminated, q_values, target_q_values = rb_workspace[
            "env/reward", 
            "env/terminated",
            "critic/q_value", 
            "target-critic/q_value"]
        
        critic_loss = compute_critic_loss(
            cfg=ddpg.cfg, 
            reward=reward,
            must_bootstrap=~terminated,
            q_values=q_values,
            target_q_values=target_q_values
        )


        # Gradient step (critic)
        ddpg.logger.add_log("critic_loss", critic_loss, ddpg.nb_steps)
        ddpg.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ddpg.critic.parameters(), ddpg.cfg.algorithm.max_grad_norm
        )
        ddpg.critic_optimizer.step()

        # Compute the actor loss
        # [[STUDENT]]...
        
        ddpg.t_actor_agent(rb_workspace, t=0, n_steps=1) # do an action proposed by actor
        ddpg.t_q_agent(rb_workspace, t=0, n_steps=1) # evaluate it with critic

        actor_loss = compute_actor_loss(rb_workspace["critic/q_value"])

        # print(f"critic / actor loss: {critic_loss} ::: {actor_loss}")


        # Gradient step (actor)
        ddpg.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ddpg.actor.parameters(), ddpg.cfg.algorithm.max_grad_norm
        )
        ddpg.actor_optimizer.step()

        # Soft update of target q function
        soft_update_params(
            ddpg.critic, ddpg.target_critic, ddpg.cfg.algorithm.tau_target
        )

        # Evaluate the actor if needed
        if ddpg.evaluate():
            if ddpg.cfg.plot_agents:
                plot_policy(
                    ddpg.actor,
                    ddpg.eval_env,
                    ddpg.best_reward,
                    str(ddpg.base_dir / "plots"),
                    ddpg.cfg.gym_env.env_name,
                    stochastic=False,
                )


class AddNoiseClip(Agent):
    def __init__(self, sigma, clip):
        super().__init__()
        self.sigma = sigma
        self.clip = clip

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = torch.clip(dist.sample(),-self.clip, +self.clip)
        self.set(("action", t), action)

class TD3(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Define the agents and optimizers for TD3

        # ADAPTED FROM DDPG:

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        self.critic_1 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic_1/")
        self.critic_2 = copy.deepcopy(self.critic_1).with_prefix("critic_2/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix("target-critic_1/")
        self.target_critic_2 = copy.deepcopy(self.critic_1).with_prefix("target-critic_2/")

        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
        self.target_actor = copy.deepcopy(self.actor)

        # As an alternative, you can use `AddOUNoise`
        noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)

        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor # NOTE: pure exploitation for evaluation

        # TD3 SPECIFIC
        noise_clip_agent = AddNoiseClip(sigma=cfg.algorithm.target_policy_noise,
                                        clip=cfg.algorithm.target_policy_noise_clip)
        self.t_q_agent_1 = TemporalAgent(self.critic_1)
        self.t_target_q_agent_1 = TemporalAgent(self.target_critic_1)
        self.t_q_agent_2 = TemporalAgent(self.critic_2)
        self.t_target_q_agent_2 = TemporalAgent(self.target_critic_2)
        self.t_actor_agent = TemporalAgent(self.actor)
        self.t_target_actor_agent = TemporalAgent(Agents(self.target_actor, noise_clip_agent))


        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer_1 = setup_optimizer(cfg.critic_optimizer, self.critic_1)
        self.critic_optimizer_2 = setup_optimizer(cfg.critic_optimizer, self.critic_2)

        self.last_policy_update_step = 0


def run_td3(td3: TD3):
    for rb in td3.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(td3.cfg.algorithm.batch_size)

        # Implement the learning loop

        # ADAPTED FROM DDPG:

        # Critic update
        # Compute critic loss
        # [[STUDENT]]...
        td3.t_q_agent_1(rb_workspace, t=0, n_steps=1) # evaluate actions from rb with online Q
        td3.t_q_agent_2(rb_workspace, t=0, n_steps=1)
        # better performance without this line: is it?? why????
        td3.t_target_actor_agent(rb_workspace, t=1, n_steps=1) # do actor's actions
        with torch.no_grad():
            td3.t_target_q_agent_1(rb_workspace, t=1, n_steps=1) # evaluate actor's action with target Q
            td3.t_target_q_agent_2(rb_workspace, t=1, n_steps=1)

        reward, terminated,\
        q_values_1, target_q_values_1,\
        q_values_2, target_q_values_2,\
        = rb_workspace[
            "env/reward", 
            "env/terminated",
            "critic_1/q_value", 
            "target-critic_1/q_value",
            "critic_2/q_value", 
            "target-critic_2/q_value",
            ]
        
        critic_loss_1 = compute_critic_loss(
            cfg=td3.cfg, 
            reward=reward,
            must_bootstrap=~terminated,
            q_values=q_values_1,
            target_q_values=torch.min(target_q_values_1,target_q_values_2)
        )

        critic_loss_2 = compute_critic_loss(
            cfg=td3.cfg, 
            reward=reward,
            must_bootstrap=~terminated,
            q_values=q_values_2,
            target_q_values=torch.min(target_q_values_1,target_q_values_2)
        )


        # Gradient step (critic)
        td3.logger.add_log("critic_loss_1", critic_loss_1, td3.nb_steps)
        td3.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.critic_1.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_optimizer_1.step()

        td3.logger.add_log("critic_loss_2", critic_loss_2, td3.nb_steps)
        td3.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.critic_2.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_optimizer_2.step()

        # Compute the actor loss
        # [[STUDENT]]...
        
        td3.t_actor_agent(rb_workspace, t=0, n_steps=1) # do an action proposed by actor
        td3.t_q_agent_1(rb_workspace, t=0, n_steps=1) # evaluate it with critic
        td3.t_q_agent_2(rb_workspace, t=0, n_steps=1)

        critic_choice = int(torch.randint(low=1, high=3, size=(1,)))
        actor_loss = compute_actor_loss(rb_workspace[f"critic_{critic_choice}/q_value"])

        # print(f"critic / actor loss: {critic_loss} ::: {actor_loss}")


        # Gradient step (actor)
        td3.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.actor.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.actor_optimizer.step()

        # Soft update of target q function
        soft_update_params(
            td3.critic_1, td3.target_critic_1, td3.cfg.algorithm.tau_target
        )

        soft_update_params(
            td3.critic_2, td3.target_critic_2, td3.cfg.algorithm.tau_target
        )

        # Update target policy
        if (
            td3.nb_steps - td3.last_policy_update_step
            > td3.cfg.algorithm.policy_delay
        ):
            td3.last_policy_update_step = td3.nb_steps
            copy_parameters(td3.t_actor_agent, td3.t_target_actor_agent)

        # Evaluate the actor if needed
        if td3.evaluate():
            if td3.cfg.plot_agents:
                plot_policy(
                    td3.actor,
                    td3.eval_env,
                    td3.best_reward,
                    str(td3.base_dir / "plots"),
                    td3.cfg.gym_env.env_name,
                    stochastic=False,
                )

# ================= OFFLINE ===================

def run_td3_offline(td3: TD3, fixed_rb: ReplayBuffer):

    # set replay buffer once and don't change it afterwards
    td3.replay_buffer = fixed_rb # TODO: any reason to but it as argument? is it used somewhere else?

    # NOTE: only a part of `iter_replay_buffers` code is taken,
    # so that replay buffer stays the same
    # TODO: in `iter_replay_buffers` train_workspace is defined once,
    # + copy of last step, but here we don't care about it because we have replay buffer?
    epochs_pb = tqdm(range(td3.cfg.algorithm.max_epochs))
    for epoch in epochs_pb:

        train_workspace = Workspace()
        td3.train_agent(
            train_workspace,
            t=0,
            n_steps=td3.cfg.algorithm.n_steps+1,
            stochastic=True,
        )
        td3.nb_steps += train_workspace.get_transitions().batch_size()


        epochs_pb.set_description(
                f"nb_steps: {td3.nb_steps}, "
                f"best reward: {td3.best_reward: .2f}, "
                f"running reward: {td3.running_reward: .2f}"
            )
        
        # Sample transitions from a fixed `replay_buffer` 
        # NOTE: modifications to this workspace don't affect rb
        rb_workspace = td3.replay_buffer.get_shuffled(td3.cfg.algorithm.batch_size)

        # The remainder is identical to `run_td3()`:
        # Implement the learning loop

        # ADAPTED FROM DDPG:

        # Critic update
        # Compute critic loss
        # [[STUDENT]]...
        td3.t_q_agent_1(rb_workspace, t=0, n_steps=1) # evaluate actions from rb with online Q
        td3.t_q_agent_2(rb_workspace, t=0, n_steps=1)
        # better performance without this line: is it?? why????
        td3.t_target_actor_agent(rb_workspace, t=1, n_steps=1) # do actor's actions
        with torch.no_grad():
            td3.t_target_q_agent_1(rb_workspace, t=1, n_steps=1) # evaluate actor's action with target Q
            td3.t_target_q_agent_2(rb_workspace, t=1, n_steps=1)

        reward, terminated,\
        q_values_1, target_q_values_1,\
        q_values_2, target_q_values_2,\
        = rb_workspace[
            "env/reward", 
            "env/terminated",
            "critic_1/q_value", 
            "target-critic_1/q_value",
            "critic_2/q_value", 
            "target-critic_2/q_value",
            ]
        
        critic_loss_1 = compute_critic_loss(
            cfg=td3.cfg, 
            reward=reward,
            must_bootstrap=~terminated,
            q_values=q_values_1,
            target_q_values=torch.min(target_q_values_1,target_q_values_2)
        )

        critic_loss_2 = compute_critic_loss(
            cfg=td3.cfg, 
            reward=reward,
            must_bootstrap=~terminated,
            q_values=q_values_2,
            target_q_values=torch.min(target_q_values_1,target_q_values_2)
        )


        # Gradient step (critic)
        td3.logger.add_log("critic_loss_1", critic_loss_1, td3.nb_steps)
        td3.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.critic_1.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_optimizer_1.step()

        td3.logger.add_log("critic_loss_2", critic_loss_2, td3.nb_steps)
        td3.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.critic_2.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_optimizer_2.step()

        # Compute the actor loss
        # [[STUDENT]]...
        
        td3.t_actor_agent(rb_workspace, t=0, n_steps=1) # do an action proposed by actor
        td3.t_q_agent_1(rb_workspace, t=0, n_steps=1) # evaluate it with critic
        td3.t_q_agent_2(rb_workspace, t=0, n_steps=1)

        critic_choice = int(torch.randint(low=1, high=3, size=(1,)))
        actor_loss = compute_actor_loss(rb_workspace[f"critic_{critic_choice}/q_value"])

        # print(f"critic / actor loss: {critic_loss} ::: {actor_loss}")


        # Gradient step (actor)
        td3.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.actor.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.actor_optimizer.step()

        # Soft update of target q function
        soft_update_params(
            td3.critic_1, td3.target_critic_1, td3.cfg.algorithm.tau_target
        )

        soft_update_params(
            td3.critic_2, td3.target_critic_2, td3.cfg.algorithm.tau_target
        )

        # Update target policy
        if (
            td3.nb_steps - td3.last_policy_update_step
            > td3.cfg.algorithm.policy_delay
        ):
            td3.last_policy_update_step = td3.nb_steps
            copy_parameters(td3.t_actor_agent, td3.t_target_actor_agent)

        # Evaluate the actor if needed
        if td3.evaluate():
            if td3.cfg.plot_agents:
                plot_policy(
                    td3.actor,
                    td3.eval_env,
                    td3.best_reward,
                    str(td3.base_dir / "plots"),
                    td3.cfg.gym_env.env_name,
                    stochastic=False,
                )

# ============== REPLAY BUFFER COMPOSITION ==============

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
    size1 = int(buffer_size*proportion)
    size2 = buffer_size - size1
    transitions1 = get_random_transitions(workspace1, size1)
    transitions2 = get_random_transitions(workspace2, size2)
    rb_mixed = ReplayBuffer(buffer_size)
    rb_mixed.put(transitions1)
    rb_mixed.put(transitions2)
    return rb_mixed


# =================== TEST RB COMPOSITIONS ===================

def plot_perf_vs_rb_composition(proportions, performances):
    # TODO: for now it just take the last evaluation, 
    # need to consider n-th evaluation instead
    means = np.array([perf[-1].mean().item() for perf in performances])
    stds = np.array([perf[-1].std().item() for perf in performances])
    plt.plot(proportions, means)
    plt.fill_between(proportions, means - stds, means + stds, alpha = 0.1)
    plt.title("Offline learning performance\nfor different replay buffer compositions")
    plt.xlabel("% of uniform exploration")
    plt.ylabel("evaluation performance")
    plt.show()

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
