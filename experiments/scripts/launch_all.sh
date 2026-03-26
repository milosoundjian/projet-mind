#!/usr/bin/env bash
set -euo pipefail

# =========================
# Hard-coded configuration
# =========================

ENV_NAME="MountainCarContinuous-v0"
PYTHON_SCRIPT="run-offline-learning.py" # <-- change this to your actual script name

REMOTE_VENV="/Vrac/pmind-rl-venv"
REMOTE_WORKDIR='$HOME/Documents/projet-mind/experiments/scripts'

MACHINES=(
	"ppti-14-502-03"
	"ppti-14-502-07"
	"ppti-14-502-08"
	"ppti-14-502-10"
	"ppti-14-502-12"
	"ppti-14-502-14"
	"ppti-14-502-15"
)

SEEDS=(
	0
	1
	2
	3
	4
	5
	6
)

EXPLOIT_REWARDS=(
	35
	61
	82
)

PROPORTIONS=(
	0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
)

# =========================
# Safety check
# =========================

if [ "${#MACHINES[@]}" -ne "${#SEEDS[@]}" ]; then
	echo "Error: MACHINES and SEEDS must have the same length."
	exit 1
fi

# =========================
# Launch jobs
# =========================

for i in "${!MACHINES[@]}"; do
	MACHINE="${MACHINES[$i]}"
	SEED="${SEEDS[$i]}"
	SESSION_NAME="pmind_${ENV_NAME}_seed${SEED}"

	echo "Launching seed ${SEED} on ${MACHINE} (tmux session: ${SESSION_NAME})"

	# Send a clean bash script to the remote machine
	ssh "${MACHINE}" 'bash -s' <<EOF
set -euo pipefail

# --- Activate environment and go to working directory ---
source "${REMOTE_VENV}/bin/activate"
cd "${REMOTE_WORKDIR}"

# Ensure logs directory exists
mkdir -p logs

echo "Running on \$(hostname)"
echo "Seed: ${SEED}"
echo "Starting tmux session: ${SESSION_NAME}"

# --- Launch tmux session (detached) ---
tmux new-session -d -s "${SESSION_NAME}" bash -lc '
  set -euo pipefail

  # Activate environment again inside tmux
  source "'"${REMOTE_VENV}"'/bin/activate"
  cd "'"${REMOTE_WORKDIR}"'"
  mkdir -p logs

  echo "Inside tmux on \$(hostname)"

  # --- Main experiment loop ---
  for EXPLOIT_REWARD in '"${EXPLOIT_REWARDS[*]}"'; do
    for PROPORTION in '"${PROPORTIONS[*]}"'; do

      echo "Running reward=\${EXPLOIT_REWARD}, proportion=\${PROPORTION}"

      # Run experiment
      python "'"${PYTHON_SCRIPT}"'" \
        --env "'"${ENV_NAME}"'" \
        --proportion "\${PROPORTION}" \
        --seed "'"${SEED}"'" \
        --exploit_reward "\${EXPLOIT_REWARD}" \
        >> "logs/'"${ENV_NAME}"'-seed'"${SEED}"'-reward\${EXPLOIT_REWARD}.log" 2>&1

    done
  done

  echo "All jobs completed for seed '"${SEED}"'"
'
EOF

done

echo
echo "All jobs launched."
echo "Check sessions with:"
echo "  ssh <machine> 'tmux ls'"
echo
echo "Attach with:"
echo "  ssh <machine> 'tmux attach -t <session_name>'"
