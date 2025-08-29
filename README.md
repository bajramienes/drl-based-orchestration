# DRL-Based Job Orchestration Framework
This repository provides the code, dataset, and result generation scripts for the paper: **"Reinforcement Learning for Energy-Efficient Job Orchestration: A Lightweight Evaluation Framework"**

## Repository Structure
- `baseline_runner.py` — FIFO baseline scheduler implementation  
- `drl_runner.py` — PPO-based Deep Reinforcement Learning (DRL) orchestration framework  
- `gen_charts.py` — Script to generate charts from experiment logs  
- `workload.xlsx` — Synthetic workload dataset (500 jobs: arrival_time, duration, memory_mb)  
- `logs/` — Execution logs for DRL folds and baseline runs (auto-generated)  
- `charts_compact/` — Compact figures combining multiple plots into single files  
- `baseline-log` / `drl-log` — Example log outputs from previous runs  
- `README.md` — Project description and usage instructions

## Requirements
- Python 3.8+  
- Dependencies:
  ```
  pandas
  numpy
  torch
  matplotlib
  seaborn
  psutil
  ```
Install with:
```bash
pip install -r requirements.txt
```

## How to Run
1) PPO orchestration (5-fold cross-validation):
```bash
python drl_runner.py
```
2) FIFO baseline:
```bash
python baseline_runner.py
```
3) Generate charts (PDF figures):
```bash
python gen_charts.py
```
Logs are written to `logs/` and figures to `charts_compact/`.

## Dataset
The workload file (`workload.xlsx`) contains 500 synthetic jobs with the following columns:
- `job_id` — unique identifier  
- `arrival_time` — scaled arrival timestamp  
- `duration` — execution time in seconds  
- `memory_mb` — memory requested in MB  
This dataset was used for all experiments in the paper.

## License
This code and dataset are provided for academic and reproducibility purposes. Redistribution, modification, or commercial use is not permitted without prior written permission from the authors.

## Contact
Enes Bajrami  
PhD Candidate in Software Engineering and Artificial Intelligence  
Faculty of Computer Science and Engineering, Ss. Cyril and Methodius University,
Skopje, North Macedonia  
enes.bajrami@students.finki.ukim.mk
