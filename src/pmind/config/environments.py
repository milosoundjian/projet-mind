import numpy as np

ENV_NAMES = (
    "CartPoleContinuous-v1",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    # "LunarLanderContinuous-v3",
)

REWARDS_TO_PLOT = {
    "unif": {
        "CartPoleContinuous-v1": ["500"],
        "Pendulum-v1": ["-209", "-233"],
        "MountainCarContinuous-v0": ["61"],
    },
    "action": {
        "CartPoleContinuous-v1": ["500"],
        "Pendulum-v1": ["-209"],
        "MountainCarContinuous-v0": ["61"],
    },
    "branch": {
        "CartPoleContinuous-v1": ["500"],
        "Pendulum-v1": ["-209"],
        "MountainCarContinuous-v0": ["61"],
    },
}

REWARD_LIMITS = {
    "CartPoleContinuous-v1": (0, 500),
    "Pendulum-v1": (-1700, 0),
    "MountainCarContinuous-v0": (-100, 100),
}

ALL_ACTION_SPACES = {
    "CartPoleContinuous-v1": [-1, 1],
    "Pendulum-v1": [-2, 2],
    "MountainCarContinuous-v0": [-1, 1],
    # "LunarLanderContinuous-v3": = []
}

ALL_STATE_SPACES = {
    "CartPoleContinuous-v1": np.array(
        [
            [-4.8, 4.8],
            [-np.inf, np.inf],
            [-0.41887903, 0.41887903],
            [-0.5, 3],  # [-np.inf, np.inf]
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

DEFAULT_STATES = {
    "CartPoleContinuous-v1": (0, 3),  # or (2, 3)
    "Pendulum-v1": (0, 1),
    "MountainCarContinuous-v0": (0, 1),
}


def get_plot_defaults(env_name, verbose=False):
    pendulum_angle = env_name == "Pendulum-v1"
    state_names = ALL_STATE_NAMES[env_name]
    if verbose:
        print("==================================")
        print(env_name)
        print("state variables", state_names)
        print("bounded by:\n", ALL_STATE_SPACES[env_name])
        print("initialized in:\n", ALL_INIT_SPACES[env_name])

    # since we are limited to 2D, choose two state variables
    state_x, state_y = DEFAULT_STATES[env_name]
    if verbose:
        if pendulum_angle:
            print("Chosen for plot: angle and angular velocity")
        else:
            print(f"Chosen for plot: {state_names[state_x]} and {state_names[state_y]}")

    polar_coord = pendulum_angle or (
        env_name == "CartPoleContinuous-v1" and (2 in (state_x, state_y))
    )
    if verbose:
        print(f"using {'polar' if polar_coord else 'cartesian'} coordinates")
        print("==================================")

    return {
        "state_x": state_x,
        "state_y": state_y,
        "pendulum_angle": pendulum_angle,
        "polar_coord": polar_coord,
    }

def show_plot_defaults(env_names):
    for env_name in env_names:
        _ = get_plot_defaults(env_name, verbose=True)
    