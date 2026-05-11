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