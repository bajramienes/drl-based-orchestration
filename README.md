
# DRL-Based Job Orchestration Framework

This repository contains the implementation, dataset, and result analysis scripts for the paper:

**"Reinforcement Learning for Energy-Efficient Job Orchestration: A Lightweight Evaluation Framework"**

## Overview

The framework evaluates a Proximal Policy Optimization (PPO)-based Deep Reinforcement Learning agent for real-time job orchestration, using CPU temperature, SLA violations, and migration overhead as optimization targets. A First-In-First-Out (FIFO) baseline is used for comparison.

All experiments are executed on a local machine using a synthetic workload of 500 jobs and evaluated through 5-fold cross-validation.

---

## Repository Structure

- `workload.csv` — Dataset of 500 job entries (arrival time, duration, memory)
- `drl_runner_kfold.py` — Main PPO agent script with 5-fold training
- `baseline_runner.py` — FIFO baseline scheduler
- `drl_training_debug.py` — Lightweight training to demonstrate reward curve
- `gen_all_charts_updated.py` — Generates all result charts in vector PDF format
- `/logs/` — DRL and baseline logs (can be regenerated)
- `/charts/` — PDF figures used in the paper (optional)


---
## Usage and Access

This code and dataset are provided for academic review and reproducibility purposes only. Redistribution, modification, or use in commercial projects is not permitted without explicit permission from the authors.


## Author

Enes Bajrami  
PhD Candidate in Software Engineering and Artificial Intelligence  
Faculty of Computer Science and Engineering - Ss. Cyril and Methodius University
enes.bajrami@students.finki.ukim.mk
