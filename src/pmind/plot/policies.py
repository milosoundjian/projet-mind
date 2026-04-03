import torch
import numpy as np
import matplotlib.pyplot as plt

ENV_NAMES = (
    "CartPoleContinuous-v1",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    # "LunarLanderContinuous-v3",
)

ALL_STATE_SPACES = {
    "CartPoleContinuous-v1": np.array(
        [[-4.8, 4.8], [-np.inf, np.inf], [-0.41887903, 0.41887903], [-np.inf, np.inf]],
        dtype=np.float32,
    ),
    "Pendulum-v1": np.array([[-1.0, 1.0], [-1.0, 1.0], [-8.0, 8.0]], dtype=np.float32),
    "MountainCarContinuous-v0": np.array(
        [[-1.2, 0.6], [-0.07, 0.07]], dtype=np.float32
    ),
}

ALL_STATE_NAMES = {
    "CartPoleContinuous-v1": ["position", "velocity", "angle", "angular velocity"],
    "Pendulum-v1": ["sine", "cosine", "angular velocity"],
    "MountainCarContinuous-v0": ["position", "velocity"],
    # "LunarLanderContinuous-v3" : [],
}

ALL_INIT_SPACES = {
    "CartPoleContinuous-v1": np.array([[-0.05, 0.05]] * 4),
    "Pendulum-v1": np.array([[-1, 1]] * 3),
    "MountainCarContinuous-v0": np.array([[-0.6, -0.4], [0, 0]]),
    # "LunarLanderContinuous-v3" : [],
}


def plot_rb_space_coverage(
    rb,
    state_x,
    state_y,
    state_names=None,
    state_space=None,
    ax=None,
    colorbar=True,
    title=None,
):
    rb_states = rb.variables["env/env_obs"].detach().numpy()
    rb_actions = rb.variables["action"].detach().numpy()
    nb_samples, _, state_dim = rb_states.shape
    _, _, action_dim = rb_actions.shape
    assert action_dim == 1, "Only one-dimensional action space is supported"

    states = rb_states[:, 0, :].reshape(nb_samples, state_dim)
    actions = rb_actions[:, 0, :].reshape(nb_samples, action_dim)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    sc = ax.scatter(
        states[:, state_x],
        states[:, state_y],
        alpha=0.4,
        s=0.01,
        c=actions,
        cmap="coolwarm",
    )
    if colorbar:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("action")

    if title is None:
        title = "Replay buffer state-action space coverage"
    ax.set_title(title)

    if state_names is not None:
        ax.set_xlabel(state_names[state_x])
        ax.set_ylabel(state_names[state_y])

    if state_space is not None:
        if np.all(~np.isinf(state_space[state_x])):
            ax.set_xlim(state_space[state_x])
        if np.all(~np.isinf(state_space[state_y])):
            ax.set_ylim(state_space[state_y])

    return ax


def plot_policy(
    policy,
    state_x,
    state_y,
    state_space,
    fixed_state=None,
    state_names=None,
    ax=None,
    grid_density=50,
    colorbar=True,
):

    state_space = state_space.copy()

    # NOTE: impose boundaries
    state_space[state_space == np.inf] = 1
    state_space[state_space == -np.inf] = -1
    x_lim, y_lim = state_space[state_x], state_space[state_y]

    # NOTE: fix other variables to 0 if it wasn't specified
    if fixed_state is None:
        fixed_state = np.zeros(len(state_space))

    x = np.linspace(*x_lim, num=grid_density)
    y = np.linspace(*y_lim, num=grid_density)
    X, Y = np.meshgrid(x, y)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if state_names is not None:
        ax.set_xlabel(state_names[state_x])
        ax.set_ylabel(state_names[state_y])

    Z = np.empty((grid_density, grid_density))
    for i_x in range(len(x)):
        for i_y in range(len(y)):
            state = fixed_state
            state[state_x] = x[i_x]
            state[state_y] = y[i_y]
            Z[i_x, i_y] = policy.model(torch.tensor([state]).float()).item()

    cntr = ax.pcolormesh(X, Y, Z, cmap="coolwarm", vmin=-1, vmax=1)
    if colorbar:
        cbar = fig.colorbar(cntr, ax=ax)
        cbar.set_label("action")

    return ax


def plot_trajectories(
    env,
    policy,
    init_space,
    nb_traj,
    traj_length,
    state_x,
    state_y,
    state_names=None,
    ax=None,
):
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
        
        if "pendulum" in str(env.unwrapped.__class__).lower():
            sin_th, cos_th, th_dot = state
            th = np.arctan2(sin_th, cos_th)
            env.state = env.unwrapped.state = np.array([th,th_dot])
        else:    
            env.state = env.unwrapped.state = state
        trajectory[:, 0] = state
        
        for j in range(traj_length - 1):
            action = np.array([policy.model(torch.tensor(state).float()).item()])
            state, *_ = env.step(action)
            trajectory[:, j + 1] = state
        

        ax.plot(trajectory[state_x], trajectory[state_y], c="black", alpha=0.3)
        if state_names is not None:
            ax.set_xlabel(state_names[state_x])
            ax.set_ylabel(state_names[state_y])
    return ax
