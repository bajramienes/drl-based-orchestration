# DRL-Based Job Orchestration Framework

This repository contains the source code, dataset, and chart generation scripts for the study:

**"Reinforcement Learning for Energy-Efficient Job Orchestration: A Lightweight Evaluation Framework"**

## Overview

The framework implements a Deep Reinforcement Learning (DRL) agent using Proximal Policy Optimization (PPO) for intelligent job orchestration in computing environments. The agent learns to optimize job scheduling based on CPU temperature, SLA violations, and migration overhead. A First-In-First-Out (FIFO) baseline scheduler is included for comparative analysis.

The system is evaluated using a synthetic workload of 500 jobs across 5-fold cross-validation. All experiments were conducted on local physical infrastructure using software-based monitoring tools.

---

## Repository Structure

- `workload.csv` — Synthetic dataset of 500 jobs (arrival time, duration, memory usage)
- `drl_runner.py` — Main script for 5-fold PPO training and testing
- `baseline_runner.py` — Baseline FIFO scheduler for comparison
- `drl_training_debug.py` — Lightweight script for initial PPO reward curve visualization
- `gen_all_charts_updated.py` — Chart generator for all figures in vector PDF format
- `/logs/` — Experiment logs for DRL and baseline (automatically generated)
- `/charts/` — All result charts used in the paper (automatically generated)

---

## How to Use

1. Run `drl_runner_kfold.py` to train and test the PPO agent across 5 folds.
2. Run `baseline_runner.py` to execute the baseline FIFO scheduler.
3. Use `gen_all_charts_updated.py` to generate all result visualizations (stored in `/charts`).
4. Optionally, run `drl_training_debug.py` for a quick debug-mode reward progression.

Ensure Python 3.8+ with dependencies: `pandas`, `numpy`, `torch`, `matplotlib`, `seaborn`, `psutil`.

---

## License and Use

This codebase and dataset are provided for academic and reproducibility purposes. Redistribution, modification, or commercial usage is not permitted without explicit written permission from the author(s).

---

## Contact

**Enes Bajrami**  
PhD Candidate in Software Engineering and Artificial Intelligence  
Faculty of Computer Science and Engineering, Ss. Cyril and Methodius University  
Email: enes.bajrami@students.finki.ukim.mk
