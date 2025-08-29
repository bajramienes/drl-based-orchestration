import pandas as pd
import numpy as np
import time
import psutil
import os
from datetime import datetime

# -----------------------------
# Config (aligned with DRL)
# -----------------------------
ARRIVAL_SCALE = 0.1          # match DRL
SLA_MULTIPLIER = 1.2         # deadline = 1.2 * duration
CPU_SAMPLE_INTERVAL = 0.05   # psutil sampling
TEMP_EWMA_ALPHA = 0.2        # same smoothing as DRL
COMPUTE_SIZE_BASELINE = 200  # baseline compute (single matmul)

# -----------------------------
# Setup
# -----------------------------
os.makedirs("logs", exist_ok=True)
df = pd.read_csv("workload.csv")

print("Starting BASELINE (FIFO) test...\n")

# init logging
out_path = "logs/baseline_log.csv"
with open(out_path, "w") as log_file:
    log_file.write(
        "job_id,start_time,phase,duration_sec,arrival_time_scaled,"
        "cpu_percent,estimated_temp_raw,estimated_temp_smoothed,"
        "memory_requested_MB,action_id,action_name,deadline_sec,elapsed_sec,"
        "sla_violation,migration_cost,reward\n"
    )

start_sim_time = time.time()
prev_temp = None

def sample_cpu_temp(prev_temp):
    cpu_pct = psutil.cpu_percent(interval=CPU_SAMPLE_INTERVAL)
    temp_raw = 40.0 + 0.5 * cpu_pct
    if prev_temp is None:
        temp_s = temp_raw
    else:
        a = TEMP_EWMA_ALPHA
        temp_s = a * temp_raw + (1 - a) * prev_temp
    return cpu_pct, temp_raw, temp_s

total_jobs = len(df)
for i, job in df.iterrows():
    job_id = job["job_id"]
    arrival_time = float(job["arrival_time"]) * ARRIVAL_SCALE
    duration = float(job["duration"])
    mem_mb = float(job["memory_mb"])
    deadline_sec = SLA_MULTIPLIER * duration

    # honor arrival
    while time.time() - start_sim_time < arrival_time:
        time.sleep(0.005)

    start_time_str = datetime.now().strftime("%H:%M:%S")

    # compute: single 200x200 matmul
    t0 = time.time()
    _ = np.dot(
        np.random.rand(COMPUTE_SIZE_BASELINE, COMPUTE_SIZE_BASELINE),
        np.random.rand(COMPUTE_SIZE_BASELINE, COMPUTE_SIZE_BASELINE)
    )
    elapsed = time.time() - t0

    # sample after compute
    cpu_pct, temp_raw, temp_s = sample_cpu_temp(prev_temp)
    prev_temp = temp_s

    # FIFO has no migration; action fixed
    action_id = -1
    action_name = "fifo"
    migration_cost = 0.0
    sla_violation = 1 if elapsed > deadline_sec else 0

    # reward composed the same way for comparability (λ are conceptual match to DRL)
    LAMBDA1, LAMBDA2, LAMBDA3 = 0.1, 2.0, 0.001
    reward = - (LAMBDA1 * temp_s + LAMBDA2 * sla_violation + LAMBDA3 * migration_cost)

    # write log row
    with open(out_path, "a") as log_file:
        log_file.write(
            f"{job_id},{start_time_str},baseline,{duration:.6f},{arrival_time:.6f},"
            f"{cpu_pct:.6f},{temp_raw:.6f},{temp_s:.6f},{mem_mb:.6f},"
            f"{action_id},{action_name},{deadline_sec:.6f},{elapsed:.6f},"
            f"{sla_violation},{migration_cost:.6f},{reward:.6f}\n"
        )

    if (i + 1) % 25 == 0 or (i + 1) == total_jobs:
        print(f"[BASELINE] Completed job {i + 1}/{total_jobs} "
              f"(elapsed {elapsed:.2f}s, temp_s {temp_s:.1f}°C, SLA {sla_violation})")

print(f"\nBaseline complete. Logs saved to {out_path}.")
