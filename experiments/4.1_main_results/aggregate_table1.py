
import os, json, csv, glob

def main():
    results = []
    for path in glob.glob("results/*.json"):
        with open(path, "r") as f:
            d = json.load(f)
        results.append([d.get("task",""), d.get("mixer",""), d.get("best_top1",0.0)])

    os.makedirs("results", exist_ok=True)
    out_csv = "results/table1.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Task","Mixer","Top1"])
        for r in results:
            w.writerow(r)
    print(f"Wrote {out_csv} with {len(results)} rows.")

if __name__ == "__main__":
    main()
