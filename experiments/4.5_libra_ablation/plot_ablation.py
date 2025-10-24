import os, json, glob
import numpy as np
import matplotlib.pyplot as plt

def load_runs():
    runs = []
    for path in glob.glob("results/ablation_*.json"):
        with open(path, "r") as f:
            runs.append((path, json.load(f)))
    return runs

def plot_acc_vs_F(runs, dataset, mixer="librakan"):
    xs, ys = [], []
    for _, d in runs:
        if d["dataset"]==dataset and d["mixer"]==mixer and not d["p_trainable"] and not d["lambda_trainable"]:
            xs.append(d["F"]); ys.append(d["best_acc"])
    if not xs: return
    o = np.argsort(xs); xs = np.array(xs)[o]; ys = np.array(ys)[o]
    plt.figure(); plt.plot(xs, ys, marker="o")
    plt.xlabel("F"); plt.ylabel("Best Acc"); plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/ablation_acc_vs_F_{dataset}.png")

def plot_activefreq_curve(runs, dataset, mixer="librakan"):
    for _, d in runs:
        if d["dataset"]==dataset and d["mixer"]==mixer:
            af = d["history"].get("active_freq", [])
            if not af: continue
            plt.figure(); plt.plot(af)
            plt.xlabel("epoch"); plt.ylabel("Active-Freq")
            os.makedirs("results", exist_ok=True)
            plt.tight_layout(); plt.savefig(f"results/ablation_activefreq_{dataset}.png")
            break

def main():
    runs = load_runs()
    for ds in ["cifar10","cifar100","mnist"]:
        plot_acc_vs_F(runs, ds)
        plot_activefreq_curve(runs, ds)

if __name__ == "__main__":
    main()
