import argparse, joblib, numpy as np, pandas as pd

# Get bridge and earthquake details from the command line
ap = argparse.ArgumentParser()
ap.add_argument("--age", type=float, required=True, help="bridge age (years)")
ap.add_argument("--mag", type=float, required=True, help="earthquake magnitude")
ap.add_argument("--dist", type=float, required=True, help="distance to epicentre (miles)")
ap.add_argument("--material", choices=["steel", "concrete"])
ap.add_argument("--type", choices=["beam_girder", "truss", "cable", "arch"])
a = ap.parse_args()

# Use the full model only when material and bridge type are provided
full = a.material is not None and a.type is not None
bundle = joblib.load(f"models/bridge_model_{'all' if full else 'core'}.joblib")
names, model = bundle["features"], bundle["model"]

# Build the input feature vector
x = [a.age, a.mag, a.dist]
if full:
    x += [
        a.material == "concrete",
        a.type == "truss",
        a.type == "cable",
        a.type == "arch"
    ]
x = np.array([x], dtype=float)

print(f"\nModel: {bundle['name']}")
print("\nInput")
for n, v in zip(names, x[0]):
    print(f"  {n:<16}{v:>8.2f}")

# Show the internal calculation steps when using Logistic Regression
if bundle["name"] == "LogReg":
    sc, lr = (
        model.named_steps["standardscaler"],
        model.named_steps["logisticregression"]
    )

    # Standardize each input feature using the training data statistics
    z = sc.transform(x)[0]

    print("\nStep 1: scale each feature, (x - mean) / std")
    print(f"  {'feature':<16}{'mean':>8}{'std':>8}{'scaled':>9}")
    for n, m, s, v in zip(names, sc.mean_, sc.scale_, z):
        print(f"  {n:<16}{m:>8.2f}{s:>8.2f}{v:>9.3f}")

    # Calculate each feature's contribution to the logistic score
    print("\nStep 2: multiply by weights and add")
    print(f"  {'feature':<16}{'weight':>8}{'scaled':>9}{'w*x':>9}")
    for n, w, v in zip(names, lr.coef_[0], z):
        print(f"  {n:<16}{w:>8.3f}{v:>9.3f}{w * v:>9.3f}")

    # Calculate the final logit before applying the sigmoid
    logit = lr.intercept_[0] + lr.coef_[0] @ z
    print(f"  {'intercept':<16}{lr.intercept_[0]:>26.3f}")
    print(f"  {'total (logit)':<16}{logit:>26.3f}")

    # Convert the logit into a probability
    print("\nStep 3: sigmoid, 1 / (1 + e^-logit)")
    print(
        f"  1 / (1 + e^{-logit:.3f}) = "
        f"{1 / (1 + np.exp(-logit)):.3f}"
    )

# Predict the probability and final damage class
p = model.predict_proba(x)[0, 1]
print(
    f"\nP(damaged) = {p:.1%}  (threshold 50%)  ->  "
    f"{'DAMAGED' if p >= 0.5 else 'UNDAMAGED'}"
)

# Find the three records in the dataset most similar to the input
df = pd.read_csv("data/bridge_data.csv")
cols = ["age_yr", "magnitude", "distance_miles"]
d = (
    ((df[cols] - x[0, :3]) / df[cols].std())
    .pow(2)
    .sum(axis=1)
    .pow(0.5)
)

near = df.loc[
    d.nsmallest(3).index,
    ["earthquake"] + cols + ["damaged"]
]

print("\nMost similar records in the dataset")
print(near.to_string(index=False))
print()
