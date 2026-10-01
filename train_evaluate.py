import argparse, os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.data import bootstrap_upsample, load_data, split
from src.models import make_knn, make_logreg, make_qda, select_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.5, 1.0])
    ap.add_argument("--data", default="data/bridge_data.csv")
    ap.add_argument("--features", choices=["core", "all"], default="core",
                    help="core = age/magnitude/distance; all = + material + bridge type")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("models", exist_ok=True)

    X, y, names = load_data(a.data, a.features)
    print(f"features: {names}")
    print(f"{len(y)} samples, {y.sum()} damaged ({y.mean():.1%})")

    rows = []
    for ratio in a.ratios:
        for run in range(a.runs):
            rng = np.random.default_rng(run)
            Xtr, Xte, ytr, yte = split(X, y, seed=run)
            Xb, yb = bootstrap_upsample(Xtr, ytr, ratio, rng)
            k, _ = select_k(Xb, yb, seed=run)
            for name, m in [("LogReg", make_logreg()), ("QDA", make_qda()),
                            ("KNN", make_knn(k))]:
                m.fit(Xb, yb)
                p, s = m.predict(Xte), m.predict_proba(Xte)[:, 1]
                rows.append(dict(
                    model=name, ratio=ratio, run=run, k=k if name == "KNN" else np.nan,
                    train_acc=(m.predict(Xb) == yb).mean(),
                    test_acc=(p == yte).mean(),
                    balanced_acc=balanced_accuracy_score(yte, p),
                    recall_pos=recall_score(yte, p),
                    precision=precision_score(yte, p, zero_division=0),
                    f1=f1_score(yte, p), auc=roc_auc_score(yte, s)))
        print(f"ratio {ratio} done")

    df = pd.DataFrame(rows); df.to_csv(f"results/metrics_{a.features}.csv", index=False)
    cols = ["train_acc", "test_acc", "balanced_acc", "recall_pos", "precision", "f1", "auc"]
    summ = df.groupby(["model", "ratio"])[cols].mean().round(3)
    overall = df.groupby("model")[cols].mean().round(3)
    with open(f"results/summary_{a.features}.md", "w") as f:
        f.write("## Mean over all runs, by model\n\n" + overall.to_markdown() + "\n\n")
        f.write("## By model and #pos/#neg ratio\n\n" + summ.to_markdown() + "\n")
    print(overall.to_string()); print(); print(summ.to_string())

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a_, col, t in zip(ax, ["test_acc", "recall_pos", "f1"],
                          ["Test accuracy", "Recall (damaged class)", "F1 (damaged class)"]):
        for n in df.model.unique():
            g = df[df.model == n].groupby("ratio")[col].mean()
            a_.plot(g.index, g.values, marker="o", label=n)
        a_.set_xlabel("#pos/#neg after resampling"); a_.set_title(t); a_.grid(alpha=.3)
    ax[0].legend(); plt.tight_layout(); plt.savefig(f"results/metrics_vs_ratio_{a.features}.png", dpi=150)

    plt.figure(figsize=(5, 4))
    plt.boxplot([df[df.model == n].auc for n in df.model.unique()],
                tick_labels=list(df.model.unique())); plt.ylabel("Test ROC-AUC")
    plt.title("AUC across runs"); plt.tight_layout()
    plt.savefig(f"results/auc_boxplot_{a.features}.png", dpi=150)

    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(X[y == 0, 1], X[y == 0, 2], s=12, c="tab:blue", label="undamaged")
    plt.scatter(X[y == 1, 1], X[y == 1, 2], s=25, c="tab:red", label="damaged")
    plt.xlabel("Earthquake magnitude"); plt.ylabel("Distance to epicentre (miles)")
    plt.legend(); plt.title("Dataset"); plt.tight_layout()
    plt.savefig("results/data_scatter.png", dpi=150)

    best = df.groupby("model").f1.mean().idxmax()
    Xb, yb = bootstrap_upsample(X, y, 0.5, np.random.default_rng(0))
    k, _ = select_k(Xb, yb)
    final = {"LogReg": make_logreg(), "QDA": make_qda(), "KNN": make_knn(k)}[best].fit(Xb, yb)
    joblib.dump({"model": final, "name": best, "features": names}, f"models/bridge_model_{a.features}.joblib")
    print(f"\nSaved best model by F1: {best}")


if __name__ == "__main__":
    main()
