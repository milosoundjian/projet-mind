import copy

from bbrl.agents import Agents, TemporalAgent
from bbrl_utils.nn import setup_optimizer
from bbrl_utils.algorithms import EpochBasedAlgo

from pmind.agents import (
    DiscreteQAgent,
    ContinuousQAgent,
    ContinuousDeterministicActor,
    EGreedyActionSelector,
    ArgmaxActionSelector,
    AddGaussianNoise,
    AddOUNoise,
    AddNoiseClip,
)

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
        # noise_agent = AddGaussianNoise(cfg.algorithm.action_noise) 
        noise_agent = AddOUNoise(cfg.algorithm.action_noise)

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


class OfflineTD3(EpochBasedAlgo):
    # Just as TD3 but without noise
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

        self.train_policy = Agents(self.actor)
        self.eval_policy = self.actor # NOTE: pure exploitation for evaluation

        # TD3 SPECIFIC

        self.t_q_agent_1 = TemporalAgent(self.critic_1)
        self.t_target_q_agent_1 = TemporalAgent(self.target_critic_1)
        self.t_q_agent_2 = TemporalAgent(self.critic_2)
        self.t_target_q_agent_2 = TemporalAgent(self.target_critic_2)
        self.t_actor_agent = TemporalAgent(self.actor)
        self.t_target_actor_agent = TemporalAgent(Agents(self.target_actor))


        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer_1 = setup_optimizer(cfg.critic_optimizer, self.critic_1)
        self.critic_optimizer_2 = setup_optimizer(cfg.critic_optimizer, self.critic_2)

        self.last_policy_update_step = 0