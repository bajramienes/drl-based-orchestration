import pandas as pd
import numpy as np
import time
import psutil
import os

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

# Load workload
df = pd.read_csv("workload.csv")

# Initialize log file
log_file = open("logs/baseline_log.csv", "w")
log_file.write("job_id,start_time,duration_sec,cpu_percent,estimated_temp,memory_requested_MB\n")

def stress_cpu(duration_sec):
    start = time.time()
    while time.time() - start < duration_sec:
        np.dot(np.random.rand(200, 200), np.random.rand(200, 200))

def estimate_temp(cpu_percent):
    return 40 + 0.5 * cpu_percent

print("Starting BASELINE test...\n")
start_sim_time = time.time()
total_jobs = len(df)

for i, job in df.iterrows():
    job_id = job["job_id"]
    arrival_time = job["arrival_time"] * 0.02  # scaled to be faster
    duration = job["duration"]
    memory_requested = job["memory_mb"]

    while time.time() - start_sim_time < arrival_time:
        time.sleep(0.01)

    job_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    psutil.cpu_percent(interval=None)
    stress_cpu(duration)
    cpu_percent = psutil.cpu_percent(interval=0.1)
    temp_est = estimate_temp(cpu_percent)

    log_file.write(f"{job_id},{job_start},{duration},{cpu_percent:.2f},{temp_est:.2f},{memory_requested}\n")
    print(f"[BASELINE] Completed job {i + 1}/{total_jobs}")

log_file.close()
print("\nBaseline execution complete. Logs saved to logs/baseline_log.csv.")
