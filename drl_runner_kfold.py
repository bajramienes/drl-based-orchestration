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

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

# Load dataset
df = pd.read_csv("workload.csv")
total_jobs = len(df)
fold_size = total_jobs // 5

GAMMA = 0.99
LR = 0.001
EPISODES = 1
LAMBDA1 = 0.1
LAMBDA2 = 2.0
LAMBDA3 = 0.001

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class OrchestrationEnv:
    def __init__(self, workload):
        self.jobs = workload.reset_index(drop=True)
        self.index = 0
        self.done = False
        self.start_sim_time = time.time()

    def reset(self):
        self.index = 0
        self.done = False
        self.start_sim_time = time.time()
        return self._get_state()

    def _get_state(self):
        if self.index >= len(self.jobs):
            self.done = True
            return np.zeros(4)
        job = self.jobs.iloc[self.index]
        return np.array([
            psutil.cpu_percent(interval=0.01) / 100,
            psutil.virtual_memory().percent / 100,
            40 + 0.5 * psutil.cpu_percent(interval=0.01),
            job["memory_mb"] / 4096
        ], dtype=np.float32)

    def step(self, action):
        job = self.jobs.iloc[self.index]
        arrival_time = job["arrival_time"] * 0.1

        while time.time() - self.start_sim_time < arrival_time:
            time.sleep(0.01)

        duration = job["duration"]
        start_time = datetime.now().strftime("%H:%M:%S")

        np.dot(np.random.rand(1300, 1300), np.random.rand(1300, 1300))

        cpu_percent = psutil.cpu_percent(interval=0.5)
        temp = 40 + 0.5 * cpu_percent
        sla_violation = 0 if action == 0 else np.random.choice([0, 1], p=[0.95, 0.05])
        cost = 0 if action == 0 else 0.1 * job["memory_mb"]
        reward = -LAMBDA1 * temp - LAMBDA2 * sla_violation - LAMBDA3 * cost

        self.index += 1
        next_state = self._get_state()
        done = self.index >= len(self.jobs)
        return next_state, reward, done, {
            "job_id": job["job_id"],
            "start_time": start_time,
            "duration_sec": duration,
            "cpu_percent": cpu_percent,
            "estimated_temp": temp,
            "memory_requested_MB": job["memory_mb"],
            "action_taken": action,
            "sla_violation": sla_violation,
            "migration_cost": cost,
            "reward": reward
        }

# Start 5-fold loop
for fold in range(5):
    test_start = fold * fold_size
    test_end = test_start + fold_size
    test_jobs = df.iloc[test_start:test_end]
    train_jobs = pd.concat([df.iloc[:test_start], df.iloc[test_end:]])

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    env = OrchestrationEnv(train_jobs)

    log_rows = []
    rewards_per_episode = []
    lambda_components = []

    print(f"\n=== Starting Fold {fold+1}/5 ===")

    for episode in range(EPISODES):
        total_reward = 0
        state = env.reset()
        while not env.done:
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            m = Categorical(probs)
            action = m.sample()

            next_state, reward, done, log_info = env.step(action.item())
            loss = -m.log_prob(action) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward
            log_rows.append(log_info)
            lambda_components.append({
                "temp_component": LAMBDA1 * log_info["estimated_temp"],
                "sla_component": LAMBDA2 * log_info["sla_violation"],
                "cost_component": LAMBDA3 * log_info["migration_cost"]
            })

            state = next_state

        rewards_per_episode.append({"episode": episode + 1, "reward": total_reward})

    pd.DataFrame(log_rows).to_csv(f"logs/drl_fold_{fold+1}_log.csv", index=False)
    pd.DataFrame(rewards_per_episode).to_csv(f"logs/rewards_fold_{fold+1}.csv", index=False)
    pd.DataFrame(lambda_components).to_csv(f"logs/lambda_components_fold_{fold+1}.csv", index=False)
    print(f"Fold {fold+1} complete. Logs saved.")
