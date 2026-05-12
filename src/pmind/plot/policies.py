import torch
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym

from ..config.environments import (
    ALL_STATE_SPACES,
    ALL_ACTION_SPACES,
    ALL_STATE_NAMES,
    ALL_INIT_SPACES,
    DEFAULT_STATES,
    EPISODE_LENGTHS,
    get_plot_defaults,
)


def plot_rb_space_coverage(
    env_name,
    rb,
    state_x,
    state_y,
    ax=None,
    colorbar=True,
    pendulum_angle=False,
):
    state_names = ALL_STATE_NAMES[env_name]
    state_space = ALL_STATE_SPACES[env_name]
    action_min, action_max = ALL_ACTION_SPACES[env_name]
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
        vmin=action_min,
        vmax=action_max,
    )
    if colorbar:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("action")

    ax.set_xlabel(x_name) if x_name is not None else None
    ax.set_ylabel(y_name) if y_name is not None else None

    ax.set_xlim(x_lim) if (x_lim is not None and np.all(~np.isinf(x_lim))) else None
    ax.set_ylim(y_lim) if (y_lim is not None and np.all(~np.isinf(y_lim))) else None

    return ax


def plot_replay_buffers(
    replay_buffers,
    env_name,
    state_x=None,
    state_y=None,
    pendulum_angle=None,
    polar_coord=None,
):
    defaults = get_plot_defaults(env_name)
    if state_x is None:
        state_x = defaults["state_x"]
    if state_y is None:
        state_y = defaults["state_y"]
    if pendulum_angle is None:
        pendulum_angle = defaults["pendulum_angle"]
    if polar_coord is None:
        polar_coord = defaults["polar_coord"]

    n_buffers = len(replay_buffers)
    ncols = 2 if n_buffers > 1 else 1
    nrows = int(np.ceil(n_buffers / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, 10),
        subplot_kw={"projection": "polar"} if polar_coord else None,
    )

    axes = np.atleast_1d(axes).flatten()

    for i, (rb_type, rb) in enumerate(replay_buffers.items()):
        plot_rb_space_coverage(
            env_name=env_name,
            rb=rb,
            state_x=state_x,
            state_y=state_y,
            pendulum_angle=pendulum_angle,
            ax=axes[i],
        )

        axes[i].set_title(rb_type)

    for ax in axes[i + 1 :]:
        fig.delaxes(ax)

    plt.suptitle(f"Replay buffer state-action space coverage for {env_name}")
    plt.tight_layout()


def plot_policy(
    env_name,
    policy,
    state_x,
    state_y,
    fixed_state=None,
    ax=None,
    grid_density=50,
    colorbar=True,
    pendulum_angle=False,
):

    state_space = ALL_STATE_SPACES[env_name].copy()
    action_min, action_max = ALL_ACTION_SPACES[env_name]
    state_names = ALL_STATE_NAMES[env_name]

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

    cntr = ax.pcolormesh(X, Y, Z, cmap="coolwarm", vmin=action_min, vmax=action_max)
    if colorbar:
        cbar = fig.colorbar(cntr, ax=ax, shrink=0.5)
        cbar.set_label("action")

    return ax


def plot_trajectories(
    env_name,
    policy,
    nb_traj,
    state_x,
    state_y,
    traj_length=None,
    ax=None,
    pendulum_angle=False,
):
    env = gym.make(env_name)
    state_names = ALL_STATE_NAMES[env_name]
    init_space = ALL_INIT_SPACES[env_name]
    state_names = ALL_STATE_NAMES[env_name]
    if traj_length is None:
        traj_length = EPISODE_LENGTHS[env_name]

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
            state, _, terminated, truncated, _ = env.step(action)

            if pendulum_angle:
                cos_th, sin_th, th_dot = state
                th = np.arctan2(sin_th, cos_th)
                trajectory[:, j + 1] = np.array([th, th_dot])
            else:
                trajectory[:, j + 1] = state

            if terminated or truncated:
                trajectory = trajectory[:, : j + 2]
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


def plot_policies(
    policies: dict | list,  # dict[reward, policy] or list[policy] by timestep
    env_name,
    n_trajectories,
    state_x=None,
    state_y=None,
    pendulum_angle=None,
    polar_coord=None,
    save_rb_policy_interval=None,
    over_time=False,
    title=None,
):
    defaults = get_plot_defaults(env_name)
    if state_x is None:
        state_x = defaults["state_x"]
    if state_y is None:
        state_y = defaults["state_y"]
    if pendulum_angle is None:
        pendulum_angle = defaults["pendulum_angle"]
    if polar_coord is None:
        polar_coord = defaults["polar_coord"]

    n_subplots = len(policies)
    ncols = (
        4
        if n_subplots % 4 == 0
        else 3
        if n_subplots > 1 and n_subplots % 2 != 0
        else 2
        if n_subplots % 2 == 0
        else 1
    )
    nrows = int(np.ceil(n_subplots / ncols))

    subplot_width = 4
    subplot_height = 4
    figsize = (
        subplot_width * ncols,
        subplot_height * nrows,
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": "polar"} if polar_coord else None,
    )

    axes = np.atleast_1d(axes).flatten()

    if not over_time:
        policies = sorted(policies.items())

    for i, item in enumerate(policies):
        if over_time:
            policy = item
            if save_rb_policy_interval:
                subtitle = f"{(i + 1) * save_rb_policy_interval} steps"
            else:
                subtitle = f"checkpoint {i}"
        else:
            reward, policy = item
            subtitle = f"reward {reward}"

        plot_policy(
            env_name=env_name,
            policy=policy,
            state_x=state_x,
            state_y=state_y,
            fixed_state=None,
            ax=axes[i],
            pendulum_angle=pendulum_angle,
        )
        if n_trajectories > 0:
            plot_trajectories(
                env_name=env_name,
                policy=policy,
                nb_traj=n_trajectories,
                state_x=state_x,
                state_y=state_y,
                traj_length=None,
                ax=axes[i],
                pendulum_angle=pendulum_angle,
            )
        axes[i].set_title(subtitle)

    for ax in axes[i + 1 :]:
        fig.delaxes(ax)

    if over_time:
        fig.suptitle(
            f"Policy evolution during offline learning of {env_name}"
            if title is None
            else title
        )
    else:
        plt.suptitle(f"Policy in state space of {env_name}" if title is None else title)
    plt.tight_layout()


def plot_checkpoint_policies(checkpoint, n_trajectories=0):
    plot_policies(
        checkpoint["policies_over_time"],
        checkpoint["env_name"],
        n_trajectories=n_trajectories,
        save_rb_policy_interval=checkpoint["save_rb_policy_interval"],
        over_time=True,
        title=f"Policy evolution during offline learning of {checkpoint['env_name']} ({'exploit_reward'})\n{checkpoint['type']} {checkpoint['rb_composition']}",
    )
