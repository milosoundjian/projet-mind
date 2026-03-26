#!/usr/bin/env bash

set -uo pipefail

# -----------------------
# CONFIG: list of machines
# -----------------------
hosts=(
    ppti-14-502-01
    ppti-14-502-02
    ppti-14-502-03
    ppti-14-502-04
    ppti-14-502-05
    ppti-14-502-06
    ppti-14-502-07
    ppti-14-502-08
    ppti-14-502-09
    ppti-14-502-10
    ppti-14-502-11
    ppti-14-502-12
    ppti-14-502-13
    ppti-14-502-14
    ppti-14-502-15
    ppti-14-502-16
)

output_file="cpu_usage.csv"
tmp_dir="$(mktemp -d)"

cleanup() {
    rm -rf "$tmp_dir"
}
trap cleanup EXIT

# Header
if [ ! -f "$output_file" ]; then
    echo "timestamp,host,status,cpu_usage_percent" > "$output_file"
fi

collect_cpu() {
    local host="$1"
    local out_file="$2"
    local timestamp cpu_usage ssh_status

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    cpu_usage=$(
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" '
            read cpu user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat
            total1=$((user + nice + system + idle + iowait + irq + softirq + steal))
            idle1=$((idle + iowait))

            sleep 1

            read cpu user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat
            total2=$((user + nice + system + idle + iowait + irq + softirq + steal))
            idle2=$((idle + iowait))

            total_diff=$((total2 - total1))
            idle_diff=$((idle2 - idle1))

            awk -v total="$total_diff" -v idle="$idle_diff" \
                '\''BEGIN {
                    if (total > 0) printf "%.2f", 100 * (total - idle) / total;
                    else print "0.00"
                }'\''
        ' 2>/dev/null
    )
    ssh_status=$?

    if [ "$ssh_status" -eq 0 ] && [ -n "$cpu_usage" ]; then
        printf "%s,%s,OK,%s\n" "$timestamp" "$host" "$cpu_usage" > "$out_file"
    else
        printf "%s,%s,UNREACHABLE,\n" "$timestamp" "$host" > "$out_file"
    fi
}

pids=()

for host in "${hosts[@]}"; do
    out_file="$tmp_dir/$host.csv"
    collect_cpu "$host" "$out_file" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

sort "$tmp_dir"/*.csv | tee -a "$output_file"