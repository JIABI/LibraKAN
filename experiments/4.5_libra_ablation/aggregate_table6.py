import os, json, csv, glob

def main():
    rows = []
    for path in glob.glob("results/ablation_*.json"):
        with open(path, "r") as f:
            d = json.load(f)
        rows.append([
            os.path.basename(path),
            d["dataset"],
            d["mixer"],
            d["F"],
            d["nufft_dim"],
            "learn" if d["p_trainable"] else d["p_fixed"],
            "learn" if d["lambda_trainable"] else d["lambda_init"],
            d["es_beta"],
            d["spectral_scale"],
            d["best_acc"],
            d.get("last_active_freq", None),
        ])
    rows.sort()
    os.makedirs("results", exist_ok=True)
    out_csv = "results/table6.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id","dataset","mixer","F","dim","p","lambda","es_beta","spectral_scale","best_acc","active_freq"])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
