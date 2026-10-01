import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CORE = ["age_yr", "magnitude", "distance_miles"]
TARGET = "damaged"


def load_data(path="data/bridge_data.csv", features="core", target=TARGET):
    df = pd.read_csv(path)
    X = df[CORE].copy()
    if features == "all":
        X["concrete"] = (df.material == "concrete").astype(int)
        for t in ["truss", "cable", "arch"]:
            X[f"type_{t}"] = (df.bridge_type == t).astype(int)
    return X.values.astype(float), df[target].values, list(X.columns)


def split(X, y, test_size=0.3, seed=0):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def bootstrap_upsample(X, y, ratio, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = int(round(ratio * len(neg)))
    idx = np.concatenate([neg, rng.choice(pos, size=n_pos, replace=True)])
    rng.shuffle(idx)
    return X[idx], y[idx]


def bootstrap_balance(X, y, rng, ratio=1.0):
    classes, counts = np.unique(y, return_counts=True)
    target = int(round(ratio * counts.max()))
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        idx.append(ci if c == classes[np.argmax(counts)] else rng.choice(ci, target, replace=True))
    idx = np.concatenate(idx); rng.shuffle(idx)
    return X[idx], y[idx]
