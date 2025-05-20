
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("charts", exist_ok=True)

# 0. Debug Reward Curve (logs_debug)
debug_file = "logs_debug/reward_debug.csv"
if os.path.exists(debug_file):
    df_debug = pd.read_csv(debug_file)
    plt.figure()
    plt.plot(df_debug["episode"], df_debug["reward"], marker='o', linestyle='-')
    plt.title("Debug PPO: Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("charts/reward_debug.pdf", format="pdf")
    plt.close()

# 1. Reward per Episode (All Folds Combined)
plt.figure()
empty_plot = True
for fold in range(1, 6):
    df = pd.read_csv(f"logs/rewards_fold_{fold}.csv")
    if len(df) > 1:
        plt.plot(df["episode"], df["reward"], label=f"Fold {fold}")
        empty_plot = False
if empty_plot:
    plt.text(0.5, 0.5, "Only 1 episode per fold. Increase EPISODES to plot learning curves.",
             ha='center', va='center', transform=plt.gca().transAxes)
plt.title("Total Reward per Episode (All Folds)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/reward_all_folds.pdf", format="pdf")
plt.close()

# 2–6: Lambda Components (One per Fold)
for fold in range(1, 6):
    df = pd.read_csv(f"logs/lambda_components_fold_{fold}.csv")
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
    plt.savefig(f"charts/lambda_components_fold_{fold}.pdf", format="pdf")
    plt.close()

# 7. Average Lambda Component Contributions (Bar Plot)
temp_means, sla_means, cost_means = [], [], []
for fold in range(1, 6):
    df = pd.read_csv(f"logs/lambda_components_fold_{fold}.csv")
    temp_means.append(df["temp_component"].mean())
    sla_means.append(df["sla_component"].mean())
    cost_means.append(df["cost_component"].mean())

df_avg = pd.DataFrame({
    "Fold": [f"Fold {i}" for i in range(1, 6)],
    "λ1 · Temperature": temp_means,
    "λ2 · SLA Violations": sla_means,
    "λ3 · Migration Cost": cost_means
})
df_avg.set_index("Fold").plot(kind="bar", figsize=(8, 5))
plt.title("Average Lambda Component Contribution per Fold")
plt.ylabel("Mean Component Value")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("charts/lambda_mean_contributions.pdf", format="pdf")
plt.close()

# 8. Estimated Temperature Distribution (All DRL Folds Combined)
temps = []
for fold in range(1, 6):
    df = pd.read_csv(f"logs/drl_fold_{fold}_log.csv")
    temps.extend(df["estimated_temp"].values)

plt.figure()
sns.histplot(temps, bins=30, kde=True, color="steelblue")
plt.title("Estimated CPU Temperature Distribution (DRL Agent)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Job Count")
plt.tight_layout()
plt.savefig("charts/drl_temp_distribution.pdf", format="pdf")
plt.close()

# 9. DRL vs Baseline: Total Execution Time
baseline = pd.read_csv("logs/baseline_log.csv")
baseline_total = baseline["duration_sec"].sum()

drl_total = 0
for fold in range(1, 6):
    df = pd.read_csv(f"logs/drl_fold_{fold}_log.csv")
    drl_total += df["duration_sec"].sum()
drl_avg = drl_total / 5

plt.figure()
plt.bar(["Baseline", "DRL (Avg)"], [baseline_total, drl_avg], color=["gray", "green"])
plt.title("Total Execution Time Comparison")
plt.ylabel("Total Duration (s)")
plt.tight_layout()
plt.savefig("charts/execution_time_comparison.pdf", format="pdf")
plt.close()

# 10. Temp Comparison: Line Chart (Baseline vs DRL)
baseline_temp = pd.read_csv("logs/baseline_log.csv")["estimated_temp"].reset_index(drop=True)
drl_temp_all = []
for fold in range(1, 6):
    df = pd.read_csv(f"logs/drl_fold_{fold}_log.csv")
    drl_temp_all.extend(df["estimated_temp"])
drl_temp_all = pd.Series(drl_temp_all).reset_index(drop=True)

plt.figure()
plt.plot(baseline_temp, label="Baseline", linestyle='-', alpha=0.7)
plt.plot(drl_temp_all, label="DRL", linestyle='-', alpha=0.7)
plt.title("Estimated CPU Temperature per Job: DRL vs Baseline (Line)")
plt.xlabel("Job Index")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/temp_comparison_line.pdf", format="pdf")
plt.close()

print("✅ All charts regenerated and saved in /charts folder.")
