import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from src.data import bootstrap_balance, load_data, split
from src.models import make_knn, make_logreg, make_qda

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=100)
ap.add_argument("--features", choices=["core", "all"], default="all")
a = ap.parse_args()
os.makedirs("results", exist_ok=True)

X, y, names = load_data(features=a.features, target="damage_level")
print(f"features: {names}; class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
rows, cms = [], {}
for run in range(a.runs):
    rng = np.random.default_rng(run)
    Xtr, Xte, ytr, yte = split(X, y, seed=run)
    Xb, yb = bootstrap_balance(Xtr, ytr, rng)
    for n, m in [("LogReg", make_logreg()), ("QDA", make_qda()), ("KNN", make_knn(3))]:
        p = m.fit(Xb, yb).predict(Xte)
        r = recall_score(yte, p, average=None, labels=[1, 2, 3], zero_division=0)
        rows.append(dict(model=n, acc=(p == yte).mean(), macro_f1=f1_score(yte, p, average="macro"),
                         recall_L1=r[0], recall_L2=r[1], recall_L3=r[2]))
        cms[n] = cms.get(n, 0) + confusion_matrix(yte, p, labels=[1, 2, 3])

df = pd.DataFrame(rows)
summ = df.groupby("model").mean().round(3)
summ.to_markdown(f"results/multiclass_summary_{a.features}.md")
print(summ.to_string())

fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for a_, (n, cm) in zip(ax, cms.items()):
    a_.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            a_.text(j, i, f"{cm[i, j]}", ha="center", va="center")
    a_.set_xticks(range(3)); a_.set_yticks(range(3))
    a_.set_xticklabels([1, 2, 3]); a_.set_yticklabels([1, 2, 3])
    a_.set_title(n); a_.set_xlabel("predicted level"); a_.set_ylabel("true level")
fig.suptitle(f"Damage level: test-set counts summed over {a.runs} random splits (row = true level)", fontsize=10)
plt.tight_layout(); plt.savefig(f"results/multiclass_confusion_{a.features}.png", dpi=150)
