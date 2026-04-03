import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_rb_space_coverage(rb, state_x: tuple[int, str], state_y: tuple[int, str], state_space = None, ax=None, colorbar=True, title=None):
    rb_states = rb.variables["env/env_obs"].detach().numpy()
    rb_actions = rb.variables["action"].detach().numpy()
    nb_samples, _, state_dim = rb_states.shape
    _, _, action_dim = rb_actions.shape
    assert action_dim == 1, "Only one-dimensional action space is supported"
    
    state_x_idx, state_x_name = state_x
    state_y_idx, state_y_name = state_y
    
    states = rb_states[:,0,:].reshape(nb_samples, state_dim)
    actions = rb_actions[:,0,:].reshape(nb_samples, action_dim)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    sc = ax.scatter(states[:,state_x_idx], states[:,state_y_idx], alpha=0.4, s=0.01, c=actions, cmap="coolwarm")
    if colorbar:
        cbar = fig.colorbar(sc,ax=ax)
        cbar.set_label("action")
    
    if title is None:
        title = "Replay buffer state-action space coverage"
    ax.set_title(title)
    ax.set_xlabel(state_x_name)
    ax.set_ylabel(state_y_name)
    if state_space is not None:
        ax.set_xlim(state_space[state_x_idx])
        ax.set_ylim(state_space[state_y_idx])
    
    return ax
    
def plot_policy(policy, state_x, state_y, state_space, ax=None, grid_density = 50, colorbar=True):
    
    state_x_idx, state_x_name = state_x
    state_y_idx, state_y_name = state_y
    x_lim,  y_lim = state_space[state_x_idx], state_space[state_y_idx]
    
    x = np.linspace(*x_lim, num=grid_density)
    y = np.linspace(*y_lim, num=grid_density)
    X, Y = np.meshgrid(x, y)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel(state_x_name)
    ax.set_ylabel(state_y_name)

    Z = np.empty((grid_density, grid_density))
    for i_x in range(len(x)):
        for i_y in range(len(y)):
            Z[i_x, i_y] = policy.model(torch.tensor([[x[i_x], y[i_y]]]).float()).item() 
            
    cntr = ax.pcolormesh(X,Y,Z,cmap="coolwarm", vmin=-1, vmax=1)
    if colorbar:
        cbar = fig.colorbar(cntr,ax=ax)
        cbar.set_label("action")
    
    return ax

def plot_trajectories(env,policy,init_space, nb_traj, traj_length, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
        
    start_states = np.vstack(
        [np.linspace(*init_state, nb_traj) for init_state in init_space]
        ).T
    for t in range(nb_traj):
        trajectory = np.empty((env.observation_space.shape[0], traj_length))
        _, _ = env.reset()
        state = start_states[t]
        env.state = env.unwrapped.state = state
        trajectory[:, 0] = state
        for j in range(traj_length - 1):
            action = np.array([policy.model(torch.tensor(state).float()).item()])
            state, *_ = env.step(action)
            trajectory[:, j + 1] = state

        ax.plot(trajectory[0], trajectory[1], c="black", alpha=0.3)
        
    return ax