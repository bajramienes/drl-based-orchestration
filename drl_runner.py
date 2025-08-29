import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import time
import os
import psutil
from datetime import datetime

# -----------------------------
# Config (tweak as needed)
# -----------------------------
ARRIVAL_SCALE = 0.1          # use same for baseline & DRL
GAMMA = 0.99
LR = 0.001
EPISODES = 3                 # keep small for quick runs; increase later if needed
LAMBDA1 = 0.1                # temp penalty
LAMBDA2 = 2.0                # SLA penalty
LAMBDA3 = 0.001              # migration cost penalty
SLA_MULTIPLIER = 1.2         # deadline = SLA_MULTIPLIER * duration (sec)
CPU_SAMPLE_INTERVAL = 0.05   # seconds for psutil sampling
TEMP_EWMA_ALPHA = 0.2        # EWMA smoothing for temp proxy
COMPUTE_SIZE = 1300          # 1300x1300 compute in DRL phase (as in your paper)

ACTION_NAMES = {0: "allocate", 1: "migrate", 2: "scale", 3: "idle"}

# -----------------------------
# Setup
# -----------------------------
os.makedirs("logs", exist_ok=True)

# Load dataset (expects columns: job_id, arrival_time, duration, memory_mb)
df = pd.read_csv("workload.csv")
total_jobs = len(df)
assert total_jobs % 5 == 0, "workload.csv should be divisible by 5 for even folds"
fold_size = total_jobs // 5

# -----------------------------
# Model
# -----------------------------
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Environment
# -----------------------------
class OrchestrationEnv:
    def __init__(self, workload):
        self.jobs = workload.reset_index(drop=True)
        self.index = 0
        self.done = False
        self.start_wall_clock = time.time()
        self.prev_temp = None  # for EWMA

    def reset(self):
        self.index = 0
        self.done = False
        self.start_wall_clock = time.time()
        self.prev_temp = None
        return self._get_state()

    def _sample_cpu_and_temp(self):
        # single sample to avoid double psutil calls
        cpu_pct = psutil.cpu_percent(interval=CPU_SAMPLE_INTERVAL)
        temp_raw = 40.0 + 0.5 * cpu_pct  # software proxy
        if self.prev_temp is None:
            temp_smoothed = temp_raw
        else:
            a = TEMP_EWMA_ALPHA
            temp_smoothed = a * temp_raw + (1 - a) * self.prev_temp
        self.prev_temp = temp_smoothed
        return cpu_pct, temp_raw, temp_smoothed

    def _get_state(self):
        if self.index >= len(self.jobs):
            self.done = True
            return np.zeros(4, dtype=np.float32)
        job = self.jobs.iloc[self.index]
        cpu_pct, _, temp_s = self._sample_cpu_and_temp()
        mem_pct = psutil.virtual_memory().percent
        return np.array([
            cpu_pct / 100.0,          # [0,1]
            mem_pct / 100.0,          # [0,1]
            temp_s / 100.0,           # scale °C to ~[0,1]
            float(job["memory_mb"]) / 4096.0
        ], dtype=np.float32)

    def step(self, action: int):
        job = self.jobs.iloc[self.index]

        # respect arrival time (scaled uniformly)
        arrival_time = float(job["arrival_time"]) * ARRIVAL_SCALE
        while time.time() - self.start_wall_clock < arrival_time:
            time.sleep(0.005)

        duration = float(job["duration"])
        deadline_sec = SLA_MULTIPLIER * duration

        # Start job execution
        start_time_str = datetime.now().strftime("%H:%M:%S")
        t0 = time.time()

        # Simulated compute
        # (Keep heavy compute; adjust COMPUTE_SIZE if you need faster runs)
        _ = np.dot(
            np.random.rand(COMPUTE_SIZE, COMPUTE_SIZE),
            np.random.rand(COMPUTE_SIZE, COMPUTE_SIZE)
        )

        elapsed = time.time() - t0
        sla_violation = 1 if elapsed > deadline_sec else 0

        # One more measurement after compute for logging & reward
        cpu_pct, temp_raw, temp_s = self._sample_cpu_and_temp()

        # Migration cost if action != allocate (0); simple proxy linked to memory
        migration_cost = 0.0 if action == 0 else 0.1 * float(job["memory_mb"])

        # Reward (penalties are positive inside, negative sign outside)
        reward = - (LAMBDA1 * temp_s + LAMBDA2 * sla_violation + LAMBDA3 * migration_cost)

        # Advance
        self.index += 1
        next_state = self._get_state()
        done = self.index >= len(self.jobs)

        info = {
            "job_id": job["job_id"],
            "start_time": start_time_str,
            "duration_sec": duration,
            "arrival_time_scaled": arrival_time,
            "cpu_percent": cpu_pct,
            "estimated_temp_raw": temp_raw,
            "estimated_temp_smoothed": temp_s,
            "memory_requested_MB": float(job["memory_mb"]),
            "action_id": action,
            "action_name": ACTION_NAMES.get(action, str(action)),
            "deadline_sec": deadline_sec,
            "elapsed_sec": elapsed,
            "sla_violation": sla_violation,
            "migration_cost": migration_cost,
            "reward": reward
        }
        return next_state, reward, done, info

# -----------------------------
# Training + Testing (5 folds)
# -----------------------------
for fold in range(5):
    test_start = fold * fold_size
    test_end = test_start + fold_size
    test_jobs = df.iloc[test_start:test_end]
    train_jobs = pd.concat([df.iloc[:test_start], df.iloc[test_end:]], ignore_index=True)

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # ---------- Training ----------
    train_env = OrchestrationEnv(train_jobs)
    rewards_per_episode = []

    for episode in range(1, EPISODES + 1):
        state = train_env.reset()
        ep_reward = 0.0

        while not train_env.done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            probs = policy(state_tensor)
            m = Categorical(probs)
            action = m.sample()

            next_state, reward, done, _ = train_env.step(action.item())
            ep_reward += reward

            # simple REINFORCE-style update (no baseline)
            loss = -m.log_prob(action) * torch.tensor(reward, dtype=torch.float32)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        rewards_per_episode.append({"episode": episode, "reward": ep_reward})

    # Save training reward curve per fold (for Fig. 3-style plot)
    pd.DataFrame(rewards_per_episode).to_csv(f"logs/train_rewards_fold_{fold+1}.csv", index=False)

    # ---------- Testing ----------
    test_env = OrchestrationEnv(test_jobs)
    test_logs = []
    lambda_components = []

    print(f"\n=== Testing Fold {fold+1}/5 ===")
    state = test_env.reset()
    total_test_reward = 0.0

    while not test_env.done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs = policy(state_tensor)
        m = Categorical(probs)
        action = m.sample()

        next_state, reward, done, log_info = test_env.step(action.item())
        total_test_reward += reward
        test_logs.append(log_info)

        lambda_components.append({
            "temp_component": LAMBDA1 * log_info["estimated_temp_smoothed"],
            "sla_component": LAMBDA2 * log_info["sla_violation"],
            "cost_component": LAMBDA3 * log_info["migration_cost"]
        })
        state = next_state

    # Save test logs & lambda components per fold
    pd.DataFrame(test_logs).to_csv(f"logs/drl_fold_{fold+1}_log.csv", index=False)
    pd.DataFrame(lambda_components).to_csv(f"logs/lambda_components_fold_{fold+1}.csv", index=False)
    pd.DataFrame([{"episode": 1, "reward": total_test_reward}]
                 ).to_csv(f"logs/test_rewards_fold_{fold+1}.csv", index=False)

    print(f"Fold {fold+1} complete. Logs saved: "
          f"drl_fold_{fold+1}_log.csv, lambda_components_fold_{fold+1}.csv, "
          f"train_rewards_fold_{fold+1}.csv, test_rewards_fold_{fold+1}.csv")
