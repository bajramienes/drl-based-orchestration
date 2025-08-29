import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Paths (edit if needed)
# ----------------------------
LOG_DIR    = r"C:\Users\Enes\Desktop\Framework\logs"
CHART_DIR  = r"C:\Users\Enes\Desktop\Framework\charts_compact"
os.makedirs(CHART_DIR, exist_ok=True)

def _read_many(pattern):
    files = sorted(glob.glob(os.path.join(LOG_DIR, pattern)))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ----------------------------
# Fig. 1 — Training reward per episode (all folds, one plot)
# ----------------------------
def fig01_training_reward():
    df = _read_many("train_rewards_fold_*.csv")
    if df.empty:
        print("No train_rewards_* files found.")
        return
    plt.figure()
    for name, g in df.groupby("_file"):
        g = g.sort_values("episode")
        label = name.replace("train_rewards_", "").replace(".csv","")  # e.g., fold_1
        plt.plot(g["episode"], g["reward"], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward per Episode (by fold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "fig01_training_reward.pdf"))
    plt.close()

# ----------------------------
# Fig. 2 — Lambda components, all folds in one 2x3 grid (a–e)
# ----------------------------
def fig02_lambda_components_grid():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "lambda_components_fold_*.csv")))
    if not files:
        print("No lambda_components_* files found.")
        return
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=False, sharey=False)
    axes = axes.flatten()
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for idx, f in enumerate(files[:6]):  # we have 5 folds; 6th subplot stays empty
        ax = axes[idx]
        df = pd.read_csv(f)
        x = np.arange(len(df))
        ax.plot(x, df["temp_component"], label="1 · Temperature")
        ax.plot(x, df["sla_component"],  label="2 · SLA Violations")
        ax.plot(x, df["cost_component"], label="3 · Migration Cost")
        fold_name = os.path.basename(f).replace(".csv","").replace("lambda_components_","")
        ax.set_title(f"{letters[idx]} {fold_name.replace('_',' ').title()}", fontsize=10)
        ax.set_xlabel("Step"); ax.set_ylabel("Component Value")
        ax.legend(fontsize=8)
    # Hide 6th if unused
    if len(files) < 6:
        axes[-1].axis("off")
    fig.suptitle("Lambda Components per Fold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(CHART_DIR, "fig02_lambda_components_grid.pdf"))
    plt.close(fig)

# ----------------------------
# Fig. 3 — Mean lambda component per fold (single bar chart)
# ----------------------------
def fig03_lambda_means():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "lambda_components_fold_*.csv")))
    if not files:
        print("No lambda_components_* files found.")
        return
    rows = []
    for f in files:
        df = pd.read_csv(f)
        rows.append({
            "fold": os.path.basename(f).split("_")[-1].split(".")[0],
            "temp": df["temp_component"].mean(),
            "sla":  df["sla_component"].mean(),
            "cost": df["cost_component"].mean()
        })
    mdf = pd.DataFrame(rows)
    x = np.arange(len(mdf))
    w = 0.25
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - w, mdf["temp"], width=w, label="1 · Temperature")
    plt.bar(x,       mdf["sla"], width=w, label="2 · SLA Violations")
    plt.bar(x + w, mdf["cost"], width=w, label="3 · Migration Cost")
    plt.xticks(x, [f"Fold {f}" for f in mdf["fold"]])
    plt.ylabel("Mean Component Value")
    plt.title("Average Lambda Component Contribution per Fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "fig03_lambda_means.pdf"))
    plt.close()

# ----------------------------
# Fig. 4 — DRL temperature histogram (smoothed)
# ----------------------------
def fig04_temp_hist():
    df = _read_many("drl_fold_*_log.csv")
    if df.empty:
        print("No drl_fold_*_log files found.")
        return
    # Prefer smoothed temp if present; else fall back
    temp_col = "estimated_temp_smoothed" if "estimated_temp_smoothed" in df.columns else \
               ("estimated_temp" if "estimated_temp" in df.columns else None)
    if temp_col is None:
        print("No temperature column found in DRL logs.")
        return
    temps = df[temp_col].values
    plt.figure()
    plt.hist(temps, bins=30)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Job Count")
    plt.title("Estimated CPU Temperature Distribution (DRL)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "fig04_temp_hist.pdf"))
    plt.close()

# ----------------------------
# Fig. 5 — Total execution time: Baseline vs DRL (DRL mean±std)
# ----------------------------
def fig05_exec_time():
    # DRL totals per fold
    drl_files = sorted(glob.glob(os.path.join(LOG_DIR, "drl_fold_*_log.csv")))
    drl_totals = []
    for f in drl_files:
        d = pd.read_csv(f)
        # Prefer elapsed_sec if logged; otherwise sum of duration_sec
        col = "elapsed_sec" if "elapsed_sec" in d.columns else "duration_sec"
        drl_totals.append(float(d[col].sum()))
    if not drl_totals:
        print("No DRL logs for exec time.")
        return
    drl_mean = float(np.mean(drl_totals))
    drl_std  = float(np.std(drl_totals, ddof=1)) if len(drl_totals) > 1 else 0.0

    # Baseline total
    bpath = os.path.join(LOG_DIR, "baseline_log.csv")
    if not os.path.exists(bpath):
        print("baseline_log.csv not found.")
        return
    bdf = pd.read_csv(bpath)
    b_col = "elapsed_sec" if "elapsed_sec" in bdf.columns else "duration_sec"
    baseline_total = float(bdf[b_col].sum())

    plt.figure()
    plt.bar(["Baseline", "DRL (mean)"], [baseline_total, drl_mean], yerr=[0.0, drl_std], capsize=6)
    plt.ylabel("Total Execution Time (s)")
    plt.title("Total Execution Time: Baseline vs DRL (mean±std across folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "fig05_exec_time.pdf"))
    plt.close()

# ----------------------------
# Fig. 6 — Job-wise temperature line: Baseline vs DRL
# ----------------------------
def fig06_jobwise_temp():
    drl = _read_many("drl_fold_*_log.csv")
    bpath = os.path.join(LOG_DIR, "baseline_log.csv")
    if drl.empty or not os.path.exists(bpath):
        print("Missing logs for job-wise temperature plot.")
        return
    base = pd.read_csv(bpath)
    # choose smoothed if present
    d_col = "estimated_temp_smoothed" if "estimated_temp_smoothed" in drl.columns else "estimated_temp"
    b_col = "estimated_temp_smoothed" if "estimated_temp_smoothed" in base.columns else "estimated_temp"

    d = drl.sort_values(["job_id"]).reset_index(drop=True)
    b = base.sort_values(["job_id"]).reset_index(drop=True)
    n = min(len(d), len(b))

    plt.figure()
    plt.plot(range(n), d[d_col].values[:n], label="DRL")
    plt.plot(range(n), b[b_col].values[:n], label="Baseline")
    plt.xlabel("Job Index (sorted by job_id)")
    plt.ylabel("Temperature (°C)")
    plt.title("Estimated CPU Temperature per Job: DRL vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "fig06_jobwise_temp.pdf"))
    plt.close()

# ----------------------------
# Fig. 7 — Radar over 4 metrics with explicit normalization
# ----------------------------
def fig07_radar():
    drl = _read_many("drl_fold_*_log.csv")
    bpath = os.path.join(LOG_DIR, "baseline_log.csv")
    if drl.empty or not os.path.exists(bpath):
        print("Missing logs for radar.")
        return
    base = pd.read_csv(bpath)

    # metric extraction
    d_exec = []
    for _, g in drl.groupby("_file"):
        col = "elapsed_sec" if "elapsed_sec" in g.columns else "duration_sec"
        d_exec.append(float(g[col].sum()))
    exec_drl = float(np.mean(d_exec))

    # choose temp columns
    d_tcol = "estimated_temp_smoothed" if "estimated_temp_smoothed" in drl.columns else "estimated_temp"
    b_tcol = "estimated_temp_smoothed" if "estimated_temp_smoothed" in base.columns else "estimated_temp"

    metrics = {
        "Baseline": {
            "exec_time": float(base["elapsed_sec"].sum() if "elapsed_sec" in base.columns else base["duration_sec"].sum()),
            "avg_temp":  float(base[b_tcol].mean()),
            "sla":       float(base["sla_violation"].sum() if "sla_violation" in base.columns else 0.0),
            "cpu_avg":   float(base["cpu_percent"].mean()) if "cpu_percent" in base.columns else 0.0
        },
        "DRL": {
            "exec_time": exec_drl,
            "avg_temp":  float(drl[d_tcol].mean()),
            "sla":       float(drl["sla_violation"].sum() if "sla_violation" in drl.columns else 0.0),
            "cpu_avg":   float(drl["cpu_percent"].mean()) if "cpu_percent" in drl.columns else 0.0
        }
    }

    names = ["Baseline", "DRL"]
    keys  = ["exec_time", "avg_temp", "sla", "cpu_avg"]
    invert = {"exec_time": True, "avg_temp": True, "sla": True, "cpu_avg": False}

    raw = {k: [metrics[n][k] for n in names] for k in keys}
    # invert where lower is better
    inv = {}
    for k in keys:
        if invert[k]:
            mx = max(raw[k])
            inv[k] = [mx - v for v in raw[k]]
        else:
            inv[k] = raw[k]
    # min-max normalize
    norm = {}
    for k in keys:
        arr = np.array(inv[k], dtype=float)
        mn, mx = arr.min(), arr.max()
        norm[k] = [1.0, 1.0] if mx - mn < 1e-12 else list((arr - mn) / (mx - mn))

    labels = ["Exec. Time", "Avg Temp", "SLA Violations", "CPU%"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    def vals(idx):
        v = [norm["exec_time"][idx], norm["avg_temp"][idx], norm["sla"][idx], norm["cpu_avg"][idx]]
        return v + [v[0]]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals(0), label="Baseline")
    ax.fill(angles, vals(0), alpha=0.1)
    ax.plot(angles, vals(1), label="DRL Proposed")
    ax.fill(angles, vals(1), alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Normalized Metric Comparison (Radar)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "fig07_radar.pdf"))
    plt.close()

def main():
    fig01_training_reward()
    fig02_lambda_components_grid()
    fig03_lambda_means()
    fig04_temp_hist()
    fig05_exec_time()
    fig06_jobwise_temp()
    fig07_radar()
    print(f"Done. Charts saved in: {CHART_DIR}")

if __name__ == "__main__":
    main()
