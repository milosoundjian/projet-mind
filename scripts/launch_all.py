#!/usr/bin/env python3
import subprocess
import shlex

project_directory = "/Vrac/21500050/projet-mind"
scripts_directory = "/Vrac/21500050/projet-mind/scripts"


def run_command(cmd: list[str]):
    print(f"\n>>> Running: {' '.join(shlex.quote(c) for c in cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.stdout.strip():
        print("STDOUT:\n", result.stdout)
    if result.stderr.strip():
        print("STDERR:\n", result.stderr)

    return result


def build_remote_script(
    env: str, seed: int, exploit_reward, proportions: list[float]
) -> str:
    remote_log = f"{scripts_directory}/tmux-logs/seed-{seed}.log"

    lines = [
        "set -euo pipefail",
        'echo "==== Starting job ===="',
        'echo "Host: $(hostname)"',
        'echo "Date: $(date)"',
        f'echo "Seed: {seed}"',
        "",
        'echo "Step 1: cd project project"',
        f"cd {project_directory}",
        "",
        'echo "Step 2: activate venv"',
        "source .venv/bin/activate",
        'echo "Python: $(which python)"',
        "",
        'echo "Step 3: cd scripts"',
        f"cd {scripts_directory}",
        'echo "PWD: $(pwd)"',
        "",
    ]

    for p in proportions:
        lines += [
            f'echo "Running seed={seed}, proportion={p}"',
            f"python ./run-offline-learning.py --env={env} --proportion={p} --seed={seed} --exploit_reward={exploit_reward} > /dev/null 2>&1",
            'echo "Exit code: $?"',
            "",
        ]

    lines += [
        'echo "==== Finished ===="',
        'echo "Date: $(date)"',
    ]

    script = "\n".join(lines)

    return f"bash -lc {shlex.quote(f'{{ {script}; }} 2>&1 | tee -a {remote_log}')}"


def launch(
    machine: str, env: str, seed: int, exploit_reward: int, proportions: list[float]
):
    session = f"seed-{seed}"

    print(f"\n=== Launching on {machine} (seed={seed}) ===")

    remote_script = build_remote_script(env, seed, exploit_reward, proportions)

    remote_cmd = (
        f"tmux has-session -t {session} 2>/dev/null "
        f"&& tmux kill-session -t {session} ; "
        f"tmux new-session -d -s {session} {shlex.quote(remote_script)}"
    )

    cmd = ["ssh", machine, remote_cmd]
    result = run_command(cmd)

    if result.returncode != 0:
        print(f"Failed on {machine}")
    else:
        print(f"Started tmux session '{session}' on {machine}")
        print("Check logs:")
        print(f"ssh {machine} 'tmux attach -t {session}'")
        print(f"ssh {machine} 'tail -f {scripts_directory}/tmux-logs/seed-{seed}.log'")


def main():
    env = "MountainCarContinuous-v0"
    exploit_reward = 61
    machines = [
        "ppti-14-302-01",
        "ppti-14-302-02",
        "ppti-14-302-03",
        "ppti-14-302-05",
        "ppti-14-302-06",
        "ppti-14-302-08",
        "ppti-14-302-09",
        "ppti-14-302-12",
        "ppti-14-302-13",
        "ppti-14-302-15",
    ]
    seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    proportions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if len(machines) != len(seeds):
        raise ValueError("machines and seeds must match")

    for m, s in zip(machines, seeds):
        launch(m, env, s, exploit_reward, proportions)

    print("\nAll jobs launched.")


if __name__ == "__main__":
    main()
