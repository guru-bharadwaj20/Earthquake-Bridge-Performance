import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)

from src.data import bootstrap_upsample, group_split, load_data, quake_groups
from src.models import make_knn, make_logreg, make_qda, select_k


# Parse experiment settings from the command line
ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=100)
ap.add_argument("--ratio", type=float, default=0.5)
ap.add_argument("--features", choices=["core", "all"], default="core")
a = ap.parse_args()

os.makedirs("results", exist_ok=True)


# Load features, labels, and earthquake group IDs
X, y, names = load_data(features=a.features)
g = quake_groups()

# Identify earthquakes that contain damaged samples
pos_ev = np.unique(g[y == 1])
print(
    f"{len(np.unique(g))} earthquake labels, "
    f"{len(pos_ev)} of them caused damage: {list(pos_ev)}"
)


# Run repeated experiments using earthquake-level train/test splits
rows = []
for run in range(a.runs):
    rng = np.random.default_rng(run)

    # Keep samples from the same earthquake in the same split
    tr, te = group_split(y, g, rng)

    # Upsample the minority class in the training data
    Xb, yb = bootstrap_upsample(X[tr], y[tr], a.ratio, rng)

    # Select the number of neighbours for KNN
    k, _ = select_k(Xb, yb, seed=run)

    # Train and evaluate all three models
    for name, m in [
        ("LogReg", make_logreg()),
        ("QDA", make_qda()),
        ("KNN", make_knn(k))
    ]:
        m.fit(Xb, yb)
        p, s = m.predict(X[te]), m.predict_proba(X[te])[:, 1]

        rows.append(dict(
            model=name,
            run=run,
            n_test_pos=int(y[te].sum()),
            test_acc=(p == y[te]).mean(),
            balanced_acc=balanced_accuracy_score(y[te], p),
            recall_pos=recall_score(y[te], p),
            precision=precision_score(y[te], p, zero_division=0),
            f1=f1_score(y[te], p),
            auc=roc_auc_score(y[te], s)
        ))


# Save metrics from the grouped-by-earthquake experiments
df = pd.DataFrame(rows)
df.to_csv(f"results/metrics_grouped_{a.features}.csv", index=False)

cols = ["test_acc", "balanced_acc", "recall_pos", "precision", "f1", "auc"]
grouped = df.groupby("model")[cols].mean()


# Load the corresponding random-split results for comparison
rand = pd.read_csv(f"results/metrics_{a.features}.csv")
rand = rand[rand.ratio == a.ratio].groupby("model")[cols].mean()

# Combine average results from both splitting strategies
both = pd.concat(
    {"random split": rand, "grouped by earthquake": grouped},
    axis=1
).round(3)

with open(f"results/grouped_vs_random_{a.features}.md", "w") as f:
    f.write(
        f"ratio {a.ratio}, {a.runs} runs, "
        f"2 of {len(pos_ev)} damaging earthquakes held out per run\n\n"
    )
    f.write(both.T.to_markdown() + "\n")

print(both.T.to_string())


# Plot F1 scores to compare the two splitting strategies
fig, ax = plt.subplots(figsize=(6.5, 4))
w = 0.38
models = list(grouped.index)
xs = np.arange(len(models))

ax.bar(
    xs - w / 2,
    rand.loc[models, "f1"],
    w,
    label="random split"
)
ax.bar(
    xs + w / 2,
    grouped.loc[models, "f1"],
    w,
    label="grouped by earthquake"
)

ax.set_xticks(xs)
ax.set_xticklabels(models)
ax.set_ylabel("Test F1 (damaged class)")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=.3)
ax.set_title(
    f"Random vs grouped split ({a.features} features, ratio {a.ratio})"
)

plt.tight_layout()
plt.savefig(
    f"results/grouped_vs_random_{a.features}.png",
    dpi=150
)
