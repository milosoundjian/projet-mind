#!/usr/bin/env python3
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- CONFIG ----
HOSTS = [
    "ppti-14-302-03",
    "ppti-14-302-04",
    "ppti-14-302-05",
    "ppti-14-302-06",
    "ppti-14-302-07",
    "ppti-14-302-10",
    "ppti-14-302-12",
    "ppti-14-302-14",
    "ppti-14-302-15",
    "ppti-14-302-16",
]

USER = None  # e.g. "milos" or None to use your current user
MAX_WORKERS = 10
SSH_TIMEOUT = 10
# ----------------


def shutdown_tmux(host: str):
    target = f"{USER}@{host}" if USER else host

    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={SSH_TIMEOUT}",
        target,
        # Only kill if tmux exists; don't error if no sessions
        "command -v tmux >/dev/null 2>&1 && tmux kill-server 2>/dev/null || true",
    ]

    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=SSH_TIMEOUT + 5,
        )

        if result.returncode == 0:
            return host, True, "tmux shut down"
        else:
            return host, False, result.stderr.strip() or result.stdout.strip()

    except subprocess.TimeoutExpired:
        return host, False, "SSH timeout"


def main():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(shutdown_tmux, host) for host in HOSTS]

        for future in as_completed(futures):
            host, ok, msg = future.result()
            print(f"[{'OK' if ok else 'FAIL'}] {host}: {msg}")


if __name__ == "__main__":
    main()
