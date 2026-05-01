import copy
import re
import sys
from contextlib import contextmanager
import torch

from bbrl_utils.notebook import tqdm
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from bbrl_utils.nn import setup_optimizer, copy_parameters, soft_update_params
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl.visu.plot_policies import plot_policy
from bbrl.utils.replay_buffer import ReplayBuffer

import d3rlpy
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.algos import DQNConfig, TD3Config

import numpy as np

import gymnasium as gym
import bbrl_gymnasium

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
from pmind.losses import compute_critic_loss, compute_actor_loss

from pmind.replay import convert_rb_to_dataset


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
        self.eval_policy = self.actor  # NOTE: pure exploitation for evaluation

        # [[STUDENT]]...
        self.t_q_agent = TemporalAgent(self.critic)
        self.t_target_q_agent = TemporalAgent(self.target_critic)
        self.t_actor_agent = TemporalAgent(self.actor)

        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic)


class TD3(EpochBasedAlgo):
    def __init__(self, cfg, offline=False):
        super().__init__(cfg)

        self.device = torch.device("cpu")

        self.offline = offline

        # Save the intermediate policy scores
        self.intermediate_rewards = cfg.algorithm.intermediate_rewards
        self.intermediate_policies = [None] * len(self.intermediate_rewards)
        self.intermediate_index = 0

        self.save_rb_policy_interval = cfg.algorithm.get("save_rb_policy_interval")
        self.policies = []

        # Define the intermediate_n_steps for offline
        self.intermediate_n_steps = cfg.algorithm.intermediate_n_steps

        # Define the agents and optimizers for TD3

        # ADAPTED FROM DDPG:

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        self.critic_1 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic_1/")
        self.critic_2 = copy.deepcopy(self.critic_1).with_prefix("critic_2/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix(
            "target-critic_1/"
        )
        self.target_critic_2 = copy.deepcopy(self.critic_1).with_prefix(
            "target-critic_2/"
        )

        self.actor = ContinuousDeterministicActor(
            obs_size,
            cfg.algorithm.architecture.actor_hidden_size,
            act_size,
            cfg.action_scaling,
        )
        self.target_actor = copy.deepcopy(self.actor)

        # As an alternative, you can use `AddOUNoise`
        if not self.offline:  # Only have a noise agent if online learning
            # noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)
            noise_agent = AddOUNoise(cfg.algorithm.action_noise)
            self.train_policy = Agents(self.actor, noise_agent)
        else:
            self.train_policy = Agents(self.actor)
        self.eval_policy = self.actor  # NOTE: pure exploitation for evaluation

        # TD3 SPECIFIC
        if not self.offline:
            noise_clip_agent = AddNoiseClip(
                sigma=cfg.algorithm.target_policy_noise,
                clip=cfg.algorithm.target_policy_noise_clip,
            )
        self.t_q_agent_1 = TemporalAgent(self.critic_1)
        self.t_target_q_agent_1 = TemporalAgent(self.target_critic_1)
        self.t_q_agent_2 = TemporalAgent(self.critic_2)
        self.t_target_q_agent_2 = TemporalAgent(self.target_critic_2)
        self.t_actor_agent = TemporalAgent(self.actor)
        if not self.offline:
            self.t_target_actor_agent = TemporalAgent(
                Agents(self.target_actor, noise_clip_agent)
            )
        else:
            self.t_target_actor_agent = TemporalAgent(Agents(self.target_actor))

        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer_1 = setup_optimizer(cfg.critic_optimizer, self.critic_1)
        self.critic_optimizer_2 = setup_optimizer(cfg.critic_optimizer, self.critic_2)

        self.last_policy_update_step = 0

    def to(self, device: torch.types.Device):
        self.device = device

        self.critic_1.to(device)
        self.critic_2.to(device)
        self.target_critic_1.to(device)
        self.target_critic_2.to(device)

        self.actor.to(device)
        self.target_actor.to(device)

        return self

    def get_device(self):
        return self.device

    def train(self, rb: ReplayBuffer | None = None):
        """
        :param rb: The replay buffer if doing online training
        """
        if self.offline:
            assert isinstance(rb, ReplayBuffer), (
                "Replay buffer is necessary for offline learning"
            )
            rb.to(self.get_device())
            self.train_offline(rb)
        else:
            self.train_online()

    def save_intermediate_policy(self):
        """
        Used to save intermediate policies, not my proudest code
        """
        if (
            self.intermediate_index < len(self.intermediate_rewards)
            and self.best_reward >= self.intermediate_rewards[self.intermediate_index]
            and self.intermediate_policies[self.intermediate_index] is None
        ):
            # print(f"Saving with reward: {self.best_reward}")
            self.intermediate_policies[self.intermediate_index] = (
                float(self.best_reward),
                copy.deepcopy(self.best_policy),
            )
            self.intermediate_index += 1

    def get_learned_policies(self):
        learned_policies = {}
        for reward_policy in self.intermediate_policies:
            if reward_policy is not None:
                learned_policies[reward_policy[0]] = reward_policy[1]
        learned_policies[float(self.best_reward)] = self.best_policy
        return learned_policies

    def learn_loop(self, rb_workspace: Workspace):
        """
        Implementation of the learning part of the TD3 algorithm
        """
        # TODO: Saving of the intermediate policies. I want to do it in a more clean way.
        # Ideally have a TD3 variable saved_policies dictionary from score to policies.

        # Get actions and Q values
        self.t_q_agent_1(rb_workspace, t=0, n_steps=1)
        self.t_q_agent_2(rb_workspace, t=0, n_steps=1)
        self.t_target_actor_agent(rb_workspace, t=1, n_steps=1)  # Do actor's action

        with torch.no_grad():
            self.t_target_q_agent_1(rb_workspace, t=1, n_steps=1)
            self.t_target_q_agent_2(rb_workspace, t=1, n_steps=1)

        (
            reward,
            terminated,
            q_values_1,
            target_q_values_1,
            q_values_2,
            target_q_values_2,
        ) = rb_workspace[
            "env/reward",
            "env/terminated",
            "critic_1/q_value",
            "target-critic_1/q_value",
            "critic_2/q_value",
            "target-critic_2/q_value",
        ]

        # Compute the critic losses using the target q values
        critic_loss_1 = compute_critic_loss(
            cfg=self.cfg,
            reward=reward,
            must_bootstrap=~terminated,
            q_values=q_values_1,
            target_q_values=torch.min(target_q_values_1, target_q_values_2),
        )

        critic_loss_2 = compute_critic_loss(
            cfg=self.cfg,
            reward=reward,
            must_bootstrap=~terminated,
            q_values=q_values_2,
            target_q_values=torch.min(target_q_values_1, target_q_values_2),
        )

        # Gradient step (critic)
        self.logger.add_log("critic_loss_1", critic_loss_1, self.nb_steps)
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_1.parameters(), self.cfg.algorithm.max_grad_norm
        )
        self.critic_optimizer_1.step()

        self.logger.add_log("critic_loss_2", critic_loss_2, self.nb_steps)
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_2.parameters(), self.cfg.algorithm.max_grad_norm
        )
        self.critic_optimizer_2.step()

        # Compute actor loss
        self.t_actor_agent(
            rb_workspace, t=0, n_steps=1
        )  # do an action proposed by actor
        self.t_q_agent_1(rb_workspace, t=0, n_steps=1)  # evaluate it with critic
        self.t_q_agent_2(rb_workspace, t=0, n_steps=1)

        # Note: In the official implementation (https://github.com/sfujim/TD3/blob/master/TD3.py)
        # they always use critic 1
        critic_choice = int(torch.randint(low=1, high=3, size=(1,)))
        actor_loss = compute_actor_loss(rb_workspace[f"critic_{critic_choice}/q_value"])

        # Gradient step (actor)
        # Gradient step (actor)
        self.logger.add_log("actor_loss", actor_loss, self.nb_steps)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.algorithm.max_grad_norm
        )
        self.actor_optimizer.step()

        # Soft update of target Q
        soft_update_params(
            self.critic_1, self.target_critic_1, self.cfg.algorithm.tau_target
        )

        soft_update_params(
            self.critic_2, self.target_critic_2, self.cfg.algorithm.tau_target
        )

        # Hard update of target policy
        # Note: In the official implementation they use a soft update for this
        if (
            self.nb_steps - self.last_policy_update_step
            > self.cfg.algorithm.policy_delay
        ):
            self.last_policy_update_step = self.nb_steps
            copy_parameters(self.t_actor_agent, self.t_target_actor_agent)

        if (
            self.save_rb_policy_interval is not None
            and self.nb_steps > 0
            and self.nb_steps % self.save_rb_policy_interval == 0
        ):
            self.policies.append(self.best_policy)

        # Evaluation
        if self.evaluate():
            # If we have a new best policy, save it
            self.save_intermediate_policy()
            if self.cfg.plot_agents:
                plot_policy(
                    self.actor,
                    self.eval_env,
                    self.best_reward,
                    str(self.base_dir / "plots"),
                    self.cfg.gym_env.env_name,
                    stochastic=False,
                )

    def train_online(self):
        for rb in self.iter_replay_buffers():
            rb.to(self.device)
            rb_workspace = rb.get_shuffled(self.cfg.algorithm.batch_size)

            self.learn_loop(rb_workspace)

    def train_offline(self, rb: ReplayBuffer):
        # We are training on a fixed replay buffer
        self.replay_buffer = (
            rb  # TODO: any reason to put it as attribute? is it used somewhere else?
        )
        # TODO: do epochs even make sense for offline learning?
        # NOTE: no self.train_agent(...) performed
        steps_pb = tqdm(range(self.cfg.algorithm.n_steps))
        for step in steps_pb:
            self.nb_steps = step + 1

            # Set description
            steps_pb.set_description(
                f"nb_steps: {self.nb_steps}, "
                f"best reward: {self.best_reward: .2f}, "
                f"running reward: {self.running_reward: .2f}"
            )

            # Sample transitions from a fixed `replay_buffer`
            # NOTE: modifications to this workspace don't affect rb
            rb_workspace = self.replay_buffer.get_shuffled(
                self.cfg.algorithm.batch_size
            )

            self.learn_loop(rb_workspace)


# Should be deprecated now
class OfflineTD3(EpochBasedAlgo):
    # Just as TD3 but without noise
    def __init__(self, cfg):
        print("`OfflineTD3` deprecated use `TD3` with parameter `offline=True`")
        super().__init__(cfg)

        # Define the agents and optimizers for TD3

        # ADAPTED FROM DDPG:

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        self.critic_1 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic_1/")
        self.critic_2 = copy.deepcopy(self.critic_1).with_prefix("critic_2/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix(
            "target-critic_1/"
        )
        self.target_critic_2 = copy.deepcopy(self.critic_1).with_prefix(
            "target-critic_2/"
        )

        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
        self.target_actor = copy.deepcopy(self.actor)

        self.train_policy = Agents(self.actor)
        self.eval_policy = self.actor  # NOTE: pure exploitation for evaluation

        # TD3 SPECIFIC

        self.t_q_agent_1 = TemporalAgent(self.critic_1)
        self.t_target_q_agent_1 = TemporalAgent(self.target_critic_1)
        self.t_q_agent_2 = TemporalAgent(self.critic_2)
        self.t_target_q_agent_2 = TemporalAgent(self.target_critic_2)

        self.t_target_actor_agent = TemporalAgent(Agents(self.target_actor))

        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer_1 = setup_optimizer(cfg.critic_optimizer, self.critic_1)
        self.critic_optimizer_2 = setup_optimizer(cfg.critic_optimizer, self.critic_2)

        self.last_policy_update_step = 0

@contextmanager
def redirect_stdout_reward_to_tqdm(pbar):
    class TqdmRewardOutput:
        def __init__(self, pbar):
            self.buffer = ""
            self.pbar = pbar

        def write(self, msg):
            match_reward = re.search(r"'environment':\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", msg)
            match_step = re.search(r"step=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", msg)
            reward = float(match_reward.group(1)) if match_reward else None
            step = int(match_step.group(1)) if match_step else None
            if step is not None and reward is not None:
                self.pbar.set_description(f"step: {step} reward: {reward:.2f}")

        def flush(self):
            pass  
    old_stdout = sys.stdout
    sys.stdout = TqdmRewardOutput(pbar)
    try:
        yield
    finally:
        sys.stdout = old_stdout
            
class BBRLStyleAlgo:
    def __init__(self, cfg, offline=True):
        if not offline:
            raise NotImplementedError

        env = gym.make(cfg.gym_env.env_name)
        self.action_scaler = d3rlpy.preprocessing.MinMaxActionScaler(
            minimum=np.array(-1.0 * cfg.action_scaling),
            maximum=np.array(1.0 * cfg.action_scaling),
        )
        self.critic_encoder_factory = d3rlpy.models.VectorEncoderFactory(
            hidden_units=cfg.algorithm.architecture.critic_hidden_size,
            activation="tanh",
        )

        self.actor_encoder_factory = d3rlpy.models.VectorEncoderFactory(
            hidden_units=cfg.algorithm.architecture.actor_hidden_size,
            activation="tanh",
        )

        # Setup algorithm
        algo = self.setup_algo(cfg)

        # Initialize NN with right obs and action dims
        algo.build_with_env(env)

        # Setup metrics

        # This metric suggests how Q functions overfit to training sets.
        # If the TD error is large, the Q functions are overfitting.
        # td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)

        env_evaluator = EnvironmentEvaluator(env, n_trials=cfg.algorithm.nb_evals)

        self.initial_reward = env_evaluator(algo, dataset=None)

        self.cfg = cfg
        self.algo = algo
        self.env_evaluator = env_evaluator

    def setup_algo(self, cfg):
        raise NotImplementedError

    def to(self, device):
        # TODO: implement
        return self

    def train(self, rb):
        dataset = convert_rb_to_dataset(rb)

        with tqdm(
            total=int(self.cfg.algorithm.n_steps / self.cfg.algorithm.eval_interval), desc=f"step: 0 reward={self.initial_reward:.2f}"
        ) as pbar:
            self.pbar = pbar
            
            with redirect_stdout_reward_to_tqdm(pbar):

                offline_log = self.algo.fit(
                    dataset,
                    n_steps=self.cfg.algorithm.n_steps,
                    n_steps_per_epoch=self.cfg.algorithm.eval_interval,
                    evaluators={
                        # 'td_error': td_error_evaluator,
                        "environment": self.env_evaluator
                    },
                    show_progress=False,
                    # Disable saving
                    logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
                    save_interval=np.inf,  # don't save anything
                    epoch_callback=lambda algo, epoch, total_step: pbar.update(1),
                )

        self.offline_log = offline_log

    @property
    def eval_rewards(self):
        return np.array([log["environment"] for epoch, log in self.offline_log])

    @property
    def policies(self):
        # TODO: implement
        return []

    @property
    def best_policy(self):
        # TODO: understand how to get the best policy...
        return None


class BBRLStyleTD3(BBRLStyleAlgo):
    def __init__(self, cfg, offline=True):
        super().__init__(cfg, offline)

    def setup_algo(self, cfg):
        return d3rlpy.algos.TD3Config(
            gamma=cfg.algorithm.discount_factor,
            actor_learning_rate=cfg.actor_optimizer.lr,
            critic_learning_rate=cfg.critic_optimizer.lr,
            batch_size=cfg.algorithm.batch_size,
            tau=cfg.algorithm.tau_target,
            action_scaler=self.action_scaler,
            actor_encoder_factory=self.critic_encoder_factory,
            critic_encoder_factory=self.actor_encoder_factory,
            target_smoothing_clip=cfg.algorithm.target_policy_noise_clip,
            target_smoothing_sigma=cfg.algorithm.target_policy_noise,  # TODO: correct?
        ).create(device=None)


class BBRLStyleIQL(BBRLStyleAlgo):
    def __init__(self, cfg, offline=True):
        super().__init__(cfg, offline)

    def setup_algo(self, cfg):
        return d3rlpy.algos.IQLConfig(
            gamma=cfg.algorithm.discount_factor,
            actor_learning_rate=cfg.actor_optimizer.lr,
            critic_learning_rate=cfg.critic_optimizer.lr,
            batch_size=cfg.algorithm.batch_size,
            tau=cfg.algorithm.tau_target,
            action_scaler=self.action_scaler,
            actor_encoder_factory=self.critic_encoder_factory,
            critic_encoder_factory=self.actor_encoder_factory,
        ).create(device=None)
