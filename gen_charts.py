import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

log_dir = r"C:\Users\Enes\Desktop\Framework\logs"
chart_dir = r"C:\Users\Enes\Desktop\Framework\charts"
os.makedirs(chart_dir, exist_ok=True)

# === 1. Reward per Episode (All Folds) ===
plt.figure()
plotted = False
for fold in range(1, 6):
    path = f"{log_dir}\\rewards_fold_{fold}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if len(df) > 1:
            plt.plot(df["episode"], df["reward"], label=f"Fold {fold}")
            plotted = True
if plotted:
    plt.title("Total Reward per Episode (All Folds)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{chart_dir}\\reward_all_folds.pdf", format="pdf")
    plt.close()

# === 2–6. Lambda Components (Line Plots) ===
for fold in range(1, 6):
    path = f"{log_dir}\\lambda_components_fold_{fold}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            plt.figure()
            plt.plot(df["temp_component"], label="λ1 · Temperature")
            plt.plot(df["sla_component"], label="λ2 · SLA Violations")
            plt.plot(df["cost_component"], label="λ3 · Migration Cost")
            plt.title(f"Lambda Components - Fold {fold}")
            plt.xlabel("Step")
            plt.ylabel("Component Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{chart_dir}\\lambda_components_fold_{fold}.pdf", format="pdf")
            plt.close()

# === 7. Average Lambda Contributions ===
temp_means, sla_means, cost_means = [], [], []
for fold in range(1, 6):
    path = f"{log_dir}\\lambda_components_fold_{fold}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            temp_means.append(df["temp_component"].mean())
            sla_means.append(df["sla_component"].mean())
            cost_means.append(df["cost_component"].mean())

if temp_means:
    df_avg = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(temp_means))],
        "λ1 · Temperature": temp_means,
        "λ2 · SLA Violations": sla_means,
        "λ3 · Migration Cost": cost_means
    })
    df_avg.set_index("Fold").plot(kind="bar", figsize=(8, 5))
    plt.title("Average Lambda Component Contribution per Fold")
    plt.ylabel("Mean Component Value")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{chart_dir}\\lambda_mean_contributions.pdf", format="pdf")
    plt.close()

# === 8. Temperature Distribution (Histogram) ===
temps = []
for fold in range(1, 6):
    path = f"{log_dir}\\drl_fold_{fold}_log.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        temps.extend(df["estimated_temp"])
if temps:
    plt.figure()
    sns.histplot(temps, bins=30, kde=True, color="steelblue")
    plt.title("Estimated CPU Temperature Distribution (DRL Agent)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Job Count")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}\\drl_temp_distribution.pdf", format="pdf")
    plt.close()

# === 9. Execution Time Comparison (with error bars + improved labels) ===
baseline = pd.read_csv(f"{log_dir}\\baseline_log.csv")
baseline_total = baseline["duration_sec"].sum()

drl_durations = []
for fold in range(1, 6):
    path = f"{log_dir}\\drl_fold_{fold}_log.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        drl_durations.append(df["duration_sec"].sum())

drl_mean = np.mean(drl_durations)
drl_std = np.std(drl_durations)

plt.figure()
bars = plt.bar(["Baseline", "DRL (Avg)"], [baseline_total, drl_mean],
               yerr=[0, drl_std], capsize=8, color=["gray", "green"])
plt.title("Total Execution Time Comparison (with Std Dev)")
plt.ylabel("Total Duration (s)")
plt.grid(True, axis='y')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2,
             height + (0.03 * height),  # 3% above bar height
             f"{int(height)}",
             ha='center',
             va='bottom',
             fontsize=9,
             fontweight='bold')

plt.tight_layout()
plt.savefig(f"{chart_dir}\\execution_time_comparison_with_std.pdf", format="pdf")
plt.close()

# === 10. Temperature Line Comparison ===
baseline_temp = baseline["estimated_temp"].reset_index(drop=True)
drl_temp = []
for fold in range(1, 6):
    path = f"{log_dir}\\drl_fold_{fold}_log.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        drl_temp.extend(df["estimated_temp"])
drl_temp = pd.Series(drl_temp).reset_index(drop=True)

plt.figure()
plt.plot(baseline_temp, label="Baseline", alpha=0.7)
plt.plot(drl_temp, label="DRL", alpha=0.7)
plt.title("Estimated CPU Temperature per Job: DRL vs Baseline")
plt.xlabel("Job Index")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{chart_dir}\\temp_comparison_line.pdf", format="pdf")
plt.close()

# === 11. Radar Chart (with SLA check) ===
metrics = ["Exec. Time", "Avg Temp", "SLA Violations", "CPU%"]
baseline_vals = [
    baseline["duration_sec"].sum(),
    baseline["estimated_temp"].mean(),
    baseline["sla_violation"].sum() if "sla_violation" in baseline.columns else 0,
    baseline["cpu_percent"].mean()
]

drl_vals = []
drl_temp_all, drl_sla_all, drl_cpu_all = [], [], []
for fold in range(1, 6):
    df = pd.read_csv(f"{log_dir}\\drl_fold_{fold}_log.csv")
    drl_temp_all.append(df["estimated_temp"].mean())
    drl_sla_all.append(df["sla_violation"].sum())
    drl_cpu_all.append(df["cpu_percent"].mean())
drl_vals = [
    np.mean(drl_durations),
    np.mean(drl_temp_all),
    np.sum(drl_sla_all),
    np.mean(drl_cpu_all)
]

baseline_vals = np.array(baseline_vals)
drl_vals = np.array(drl_vals)
vals_min = np.minimum(baseline_vals, drl_vals)
vals_max = np.maximum(baseline_vals, drl_vals)
baseline_norm = (baseline_vals - vals_min) / (vals_max - vals_min + 1e-6)
drl_norm = (drl_vals - vals_min) / (vals_max - vals_min + 1e-6)

labels = metrics
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
baseline_plot = np.concatenate((baseline_norm, [baseline_norm[0]]))
drl_plot = np.concatenate((drl_norm, [drl_norm[0]]))
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, baseline_plot, label="Baseline", linestyle='--', marker='o')
ax.plot(angles, drl_plot, label="DRL Proposed", linestyle='-', marker='o')
ax.fill(angles, drl_plot, alpha=0.1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
plt.title("Normalized Metric Comparison (Radar Chart)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{chart_dir}\\radar_comparison.pdf", format="pdf")
plt.close()

# === 12. Reward Training Curve (Combined from rewards_all.csv) ===
reward_all_path = f"{log_dir}\\rewards_all.csv"
if os.path.exists(reward_all_path):
    df_rewards_all = pd.read_csv(reward_all_path)
    if not df_rewards_all.empty:
        if "fold" in df_rewards_all.columns:
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=df_rewards_all, x="episode", y="reward", hue="fold", marker='o')
            plt.title("PPO Training Reward per Episode (All Folds)")
        else:
            plt.figure()
            plt.plot(df_rewards_all["episode"], df_rewards_all["reward"], marker='o', linestyle='-', color='blue')
            plt.title("PPO Training Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{chart_dir}\\reward_training_curve.pdf", format="pdf")
        plt.close()

print("All updated charts successfully generated in:", chart_dir)
