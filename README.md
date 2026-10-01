# Predicting Bridge Performance under Earthquakes (Problem #27)

UE24CS352A – Machine Learning mini-project. Binary classification: given a
bridge's **age**, the earthquake's **magnitude** and the **distance to the
epicentre**, predict whether the bridge was **damaged (1)** or **undamaged (0)**.
An optional extension adds material/bridge type and predicts the damage level.

Models: Logistic Regression, Quadratic Discriminant Analysis (QDA), K-Nearest
Neighbours (k chosen by cross-validation).

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python train_evaluate.py --features core   # age, magnitude, distance (as in the report)
python train_evaluate.py --features all    # + material and bridge type
python train_multiclass.py                 # extension: damage LEVEL 1/2/3
python predict.py --age 53 --mag 6.9 --dist 60
python predict.py --age 53 --mag 6.9 --dist 60 --material steel --type beam_girder
```

Each binary run does 100 random splits x 3 resampling ratios (~1 min). Use
`--runs 20` for a quick check. Outputs go to `results/` (tables, plots) and
`models/` (best model by mean F1, used by `predict.py`).

## Project layout

```
data/bridge_data.csv     249 records, all columns from the source table
src/data.py              loading, stratified split, bootstrap up-sampling
src/models.py            model factories + CV selection of k
train_evaluate.py        binary experiment (damaged / undamaged)
train_multiclass.py      extension: damage level 1/2/3
predict.py               CLI for the live demo
results/                 generated metrics and figures
```

## Dataset

249 bridge/earthquake pairs, **21 damaged (8.4 %)**. Columns: bridge name,
earthquake, material (steel/concrete), bridge type (beam/girder, truss, cable,
arch), age, magnitude, distance to epicentre (miles), damaged (0/1), damage
level (1 none, 2 moderate/severe: 11 records, 3 partial/full collapse: 10 records).
Damaged examples come from government reports of historic earthquakes;
undamaged examples are other USGS-catalogued earthquakes near the same
bridges, assuming "no recorded damage = undamaged". Source: Jiang & Zhang,
*Predicting Bridge Performance under Earthquakes with Supervised Learning*
(Stanford CS229, 2018), https://github.com/jzy95310/CS229-Fall-2018-Final-Report
(`Data/ds2.csv`, cleaned and re-labelled with readable category names).

## Method (and what we changed vs. the reference report)

1. **Stratified 70/30 split first**, so the test set keeps the true ~1:10 ratio.
2. **Bootstrap up-sampling of damaged examples on the training set only**
   (target #pos/#neg of 0.2, 0.5, 1.0). Resampling before splitting would leak
   duplicate positives into the test set.
3. **Feature scaling** for Logistic Regression and KNN (distance and magnitude
   are on very different scales).
4. **QDA with `reg_param=0.1`** since only ~15 positives are available to
   estimate a covariance matrix.
5. **k for KNN chosen by 5-fold CV on F1**, not accuracy.
6. **Metrics beyond accuracy**: with 8 % positives, always predicting
   "undamaged" gives ~92 % accuracy, so we report recall, precision, F1,
   balanced accuracy and ROC-AUC. Everything is averaged over 100 random
   splits.

## Results (held-out test sets, mean of 100 runs x 3 ratios)

**Binary, core features (age, magnitude, distance)**

| Model  | Test acc | Balanced acc | Recall (damaged) | Precision | F1    | AUC   |
|--------|----------|--------------|------------------|-----------|-------|-------|
| LogReg | 0.972    | 0.942        | 0.906            | 0.803     | 0.841 | 0.994 |
| QDA    | 0.969    | 0.925        | 0.873            | 0.793     | 0.819 | 0.992 |
| KNN    | 0.972    | 0.895        | 0.802            | 0.868     | 0.817 | 0.906 |

**Binary, all features (+ material, bridge type)**

| Model  | Test acc | Balanced acc | Recall (damaged) | Precision | F1    | AUC   |
|--------|----------|--------------|------------------|-----------|-------|-------|
| LogReg | 0.974    | 0.934        | 0.886            | 0.833     | 0.849 | 0.988 |
| QDA    | 0.968    | 0.907        | 0.834            | 0.803     | 0.804 | 0.986 |
| KNN    | 0.955    | 0.800        | 0.616            | 0.801     | 0.672 | 0.803 |

Adding material and bridge type gives no clear gain for Logistic Regression
(F1 up slightly, AUC down slightly) and hurts KNN, possibly because the binary
features distort distances when there are so few positives (not tested). For Logistic
Regression and QDA, higher up-sampling ratios raise recall but lower
precision (KNN improves on both).

**Multi-class damage level (all features, minority levels bootstrapped up to the majority count; KNN uses a fixed k=3)**

| Model  | Accuracy | Macro-F1 | Recall L1 | Recall L2 | Recall L3 |
|--------|----------|----------|-----------|-----------|-----------|
| LogReg | 0.936    | 0.662    | 0.966     | 0.630     | 0.550     |
| QDA    | 0.927    | 0.621    | 0.962     | 0.553     | 0.513     |
| KNN    | 0.923    | 0.554    | 0.971     | 0.357     | 0.380     |

With only 11 and 10 examples in levels 2 and 3, these numbers are noisy.
Full tables and plots are in `results/`.

## Limitations

- Only 21 positives, so results vary a lot between splits.
- Negatives rest on the "no report = no damage" assumption, which may be wrong.
- Positives come from famous large earthquakes, so magnitude/distance separate
  the classes easily (AUC ~0.99). Performance on typical new events would
  likely be lower.
- Bootstrap resampling duplicates data; it does not add information.
- The same earthquake appears in many rows, and our splits are random by row,
  so train and test can contain records from the same event. This likely
  inflates scores; a split grouped by earthquake would be a stricter test.
- Our numbers differ from the reference report (e.g. its Logistic Regression
  test accuracy was 73.8 %). We did not trace the exact cause; our evaluation
  differs in the ways listed under Method.

## Team

UE24CS352A, Section C, Team 3, Problem statement #27

| Name | SRN |
|------|-----|
| Guru R Bharadwaj | PES1UG24CS177 |
| Chukkapalli Rohan | PES1UG24CS135 |

Task split is in [CONTRIBUTING.md](CONTRIBUTING.md).
