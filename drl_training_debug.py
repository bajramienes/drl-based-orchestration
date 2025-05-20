
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

# Setup
os.makedirs("logs_debug", exist_ok=True)
df = pd.read_csv("workload.csv").iloc[:100]  # Use only first 100 jobs

EPISODES = 10
GAMMA = 0.99
LR = 0.001
LAMBDA1, LAMBDA2, LAMBDA3 = 0.1, 2.0, 0.001

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
        arrival_time = job["arrival_time"] * 0.05

        while time.time() - self.start_sim_time < arrival_time:
            time.sleep(0.01)

        # Simulate lighter job workload
        np.dot(np.random.rand(300, 300), np.random.rand(300, 300))

        cpu_percent = psutil.cpu_percent(interval=0.2)
        temp = 40 + 0.5 * cpu_percent
        sla_violation = 0 if action == 0 else np.random.choice([0, 1], p=[0.95, 0.05])
        cost = 0 if action == 0 else 0.1 * job["memory_mb"]
        reward = -LAMBDA1 * temp - LAMBDA2 * sla_violation - LAMBDA3 * cost

        self.index += 1
        next_state = self._get_state()
        done = self.index >= len(self.jobs)
        return next_state, reward, done

# Initialize
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=LR)
env = OrchestrationEnv(df)

rewards_per_episode = []
print("\n[DEBUG TRAINING] PPO on 100 jobs for 10 episodes...")

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    while not env.done:
        state_tensor = torch.FloatTensor(state)
        probs = policy(state_tensor)
        m = Categorical(probs)
        action = m.sample()

        next_state, reward, done = env.step(action.item())
        loss = -m.log_prob(action) * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward
        state = next_state

    rewards_per_episode.append({"episode": episode + 1, "reward": total_reward})
    print(f"Episode {episode + 1}: Total Reward = {round(total_reward, 2)}")

pd.DataFrame(rewards_per_episode).to_csv("logs_debug/reward_debug.csv", index=False)
print("\n✅ Debug training complete. Reward curve saved to logs_debug/reward_debug.csv")
