import torch
import torch.nn as nn

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
    target = reward[1] + gamma * max_q * must_bootstrap[1]

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

    target = reward[1] + gamma * max_q * must_bootstrap[1]

    # Compute critic loss (no need to use must_bootstrap here since we are dealing with "full" transitions)
    mse = nn.MSELoss()
    critic_loss = mse(target, qvals)

    return critic_loss

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
    target = reward[1] + gamma * target_q_values[1] * must_bootstrap[1]

    mse = nn.MSELoss()
    critic_loss = mse(target, q_values[0])

    return critic_loss

def compute_actor_loss(q_values):
    """Returns the actor loss

    :param q_values: The q-values (shape 2xB)
    :return: A scalar (the loss)
    """
    return - q_values[0].mean() # start or end of transition, or both?