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
        [[-4.8, 4.8], 
         [-np.inf, np.inf], 
         [-0.41887903, 0.41887903], 
         [-0.5, 3]#[-np.inf, np.inf]
         ],
        dtype=np.float32,
    ),
    "Pendulum-v1": np.array([[-1.0, 1.0], [-1.0, 1.0], [-8.0, 8.0]], dtype=np.float32),
    "MountainCarContinuous-v0": np.array(
        [[-1.2, 0.6], [-0.07, 0.07]], dtype=np.float32
    ),
}

ALL_STATE_NAMES = {
    "CartPoleContinuous-v1": ["position", "velocity", "angle", "angular velocity"],
    "Pendulum-v1": ["cosine", "sine", "angular velocity"],
    "MountainCarContinuous-v0": ["position", "velocity"],
    # "LunarLanderContinuous-v3" : [],
}

ALL_INIT_SPACES = {
    "CartPoleContinuous-v1": np.array([[-0.05, 0.05]] * 4),
    "Pendulum-v1": np.array([[-1, 1]] * 3),
    "MountainCarContinuous-v0": np.array([[-0.6, -0.4], [0, 0]]),
    # "LunarLanderContinuous-v3" : [],
}

EPISODE_LENGTHS = {
    "CartPoleContinuous-v1": 500,
    "Pendulum-v1": 200,
    "MountainCarContinuous-v0": 900,
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
    pendulum_angle=False,
):
    rb_states = rb.variables["env/env_obs"].detach().numpy()
    rb_actions = rb.variables["action"].detach().numpy()
    nb_samples, _, state_dim = rb_states.shape
    _, _, action_dim = rb_actions.shape
    assert action_dim == 1, "Only one-dimensional action space is supported"

    states = rb_states[:, 0, :].reshape(nb_samples, state_dim)
    actions = rb_actions[:, 0, :].reshape(nb_samples, action_dim)

    if pendulum_angle:
        cos_th, sin_th = states[:, 0], states[:, 1]
        x = np.arctan2(sin_th, cos_th)
        y = states[:, 2]
        x_name = "angle"
        y_name = "angular velocity"
        x_lim = np.array([-np.pi, np.pi])
        y_lim = state_space[2] if state_space is not None else None
    else:
        x = states[:, state_x]
        y = states[:, state_y]
        x_name = state_names[state_x] if state_names is not None else None
        y_name = state_names[state_y] if state_names is not None else None
        x_lim = state_space[state_x] if state_space is not None else None
        y_lim = state_space[state_y] if state_space is not None else None

    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "polar"} if pendulum_angle else None
        )

    else:
        fig = ax.figure

    sc = ax.scatter(
        x,
        y,
        alpha=0.4,
        s=0.01,
        c=actions,
        cmap="coolwarm",
    )
    if colorbar:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("action")


    ax.set_xlabel(x_name) if x_name is not None else None
    ax.set_ylabel(y_name) if y_name is not None else None

    ax.set_xlim(x_lim) if (x_lim is not None and np.all(~np.isinf(x_lim))) else None
    ax.set_ylim(y_lim) if (y_lim is not None and np.all(~np.isinf(y_lim))) else None

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
    pendulum_angle=False,
):

    state_space = state_space.copy()

    # NOTE: impose boundaries
    state_space[state_space == np.inf] = 1
    state_space[state_space == -np.inf] = -1

    if pendulum_angle:
        x_name = "angle"
        y_name = "angular velocity"
        x_lim = np.array([-np.pi, np.pi])
        y_lim = state_space[2]
    else:
        x_name = state_names[state_x] if state_names is not None else None
        y_name = state_names[state_y] if state_names is not None else None
        x_lim = state_space[state_x]
        y_lim = state_space[state_y]

    # NOTE: fix other variables to 0 if it wasn't specified
    if fixed_state is None:
        fixed_state = np.zeros(len(state_space))

    x = np.linspace(*x_lim, num=grid_density)
    y = np.linspace(*y_lim, num=grid_density)
    X, Y = np.meshgrid(x, y)

    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "polar"} if pendulum_angle else None
        )

    else:
        fig = ax.figure

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel(x_name) if x_name else None
    ax.set_ylabel(y_name) if y_name else None

    Z = np.empty((grid_density, grid_density))
    for i_x in range(len(x)):
        for i_y in range(len(y)):
            if pendulum_angle:

                state = np.array([np.cos(x[i_x]), np.sin(x[i_x]), y[i_y]])
            else:
                state = fixed_state.copy()
                state[state_x] = x[i_x]
                state[state_y] = y[i_y]
                
            Z[i_x, i_y] = policy.model(torch.tensor([state]).float()).item()

    cntr = ax.pcolormesh(X, Y, Z, cmap="coolwarm", vmin=-1, vmax=1)
    if colorbar:
        cbar = fig.colorbar(cntr, ax=ax, shrink=0.5)
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
    pendulum_angle=False,
):
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "polar"} if pendulum_angle else None
        )

    else:
        fig = ax.figure

    init_linspaced = [np.linspace(*init_lim, nb_traj) for init_lim in init_space]
    grids = np.meshgrid(*init_linspaced, indexing="ij")
    start_states = np.stack(grids, axis=-1).reshape(-1, len(init_space))

    if pendulum_angle:
        x_name = "angle"
        y_name = "angular velocity"
    else:
        x_name = state_names[state_x] if state_names is not None else None
        y_name = state_names[state_y] if state_names is not None else None

    for t in range(len(start_states)):
        if pendulum_angle:
            trajectory = np.empty((2, traj_length))
        else:
            trajectory = np.empty((env.observation_space.shape[0], traj_length))
        _, _ = env.reset()
        state = start_states[t]

        if "pendulum" in str(env.unwrapped.__class__).lower():
            cos_th, sin_th, th_dot = state
            th = np.arctan2(sin_th, cos_th)
            env.state = env.unwrapped.state = np.array([th, th_dot])
        else:
            env.state = env.unwrapped.state = state
            
        if pendulum_angle:
            trajectory[:, 0] = np.array([th, th_dot])
        else:
            trajectory[:, 0] = state

        for j in range(traj_length - 1):
            action = np.array([policy.model(torch.tensor(state).float()).item()])
            state, _, terminated, truncated, _  = env.step(action)

            if pendulum_angle:
                cos_th, sin_th, th_dot = state
                th = np.arctan2(sin_th, cos_th)
                trajectory[:, j + 1] = np.array([th, th_dot])
            else:
                trajectory[:, j + 1] = state
            
            if terminated or truncated:
                trajectory = trajectory[:, :j + 2]
                break

        if pendulum_angle:
            x = trajectory[0]
            y = trajectory[1]
        else:
            x = trajectory[state_x]
            y = trajectory[state_y]

        ax.plot(x, y, c="black", alpha=0.3)

        ax.set_xlabel(x_name) if x_name else None
        ax.set_ylabel(y_name) if y_name else None
    return ax
