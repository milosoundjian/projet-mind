import torch
from bbrl_utils.notebook import tqdm
from bbrl.workspace import Workspace
from bbrl_utils.nn import copy_parameters, soft_update_params
from bbrl.visu.plot_policies import plot_policy
from bbrl.utils.replay_buffer import ReplayBuffer

from pmind.algorithms import DQN, DDPG, TD3
from pmind.losses import compute_critic_loss, compute_actor_loss

from copy import deepcopy

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

def run_td3(td3: TD3, save_model_at_rewards=None):
    policies = {}

    save_model_at_rewards = deepcopy(save_model_at_rewards)
    
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
            # Here we are hitting a new score
            if save_model_at_rewards != [] and save_model_at_rewards[0] <= td3.best_reward:
                save_model_at_rewards.pop(0)
                print(f"Saving w/ reward: {td3.best_reward}")
                td3.best_policy.save_model(f"../models/{td3.cfg.gym_env['env_name']}-model-{int(td3.best_reward)}.pt")
                policies[int(td3.best_reward)] = td3.best_policy
            if td3.cfg.plot_agents:
                plot_policy(
                    td3.actor,
                    td3.eval_env,
                    td3.best_reward,
                    str(td3.base_dir / "plots"),
                    td3.cfg.gym_env.env_name,
                    stochastic=False,
                )

        if save_model_at_rewards == []:
            break
        
    return policies


def run_td3_offline(td3: TD3, fixed_rb: ReplayBuffer):

    # set replay buffer once and don't change it afterwards
    td3.replay_buffer = fixed_rb # TODO: any reason to put it as attribute? is it used somewhere else?

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
        
        # TODO: OS: quand vous faites target_q_value = torch.min (...),
        #  moi j'ajoute à la fin ".squeeze(-1)", mais c'est peut-être que 
        # je n'extraie pas les mêmes données que vous du workspace
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

def load_trained_agents(env_name, rewards):
    trained_agents = {}
    for k in rewards:
        trained_agents[k] = torch.load(f"../models/{env_name}-model-{k}.pt", weights_only=False)
    return trained_agents