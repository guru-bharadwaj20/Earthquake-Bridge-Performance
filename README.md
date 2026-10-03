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
python train_multiclass.py                 # extension: damage_LEVEL 1/2/3
python grouped_split.py                    # test on earthquakes the model has never seen
python predict.py --age 53 --mag 6.9 --dist 60
python predict.py --age 53 --mag 6.9 --dist 60 --material steel --type beam_girder
```

Each binary run does 100 random splits x 3 resampling ratios (~1 min). Use
`--runs 20` for a quick check. Outputs go to `results/` (tables, plots) and
`models/` (best model by mean F1, used by `predict.py`).

## Project layout

```
data/bridge_data.csv     249 records, all columns from the source table
data/raw/ds2_original.csv  untouched copy of ds2.csv from the reference repo
sources/                 reference paper, poster and assignment guidelines
src/data.py              loading, stratified split, bootstrap up-sampling
src/models.py            model factories + CV selection of k
train_evaluate.py        binary experiment (damaged / undamaged)
train_multiclass.py      extension: damage level 1/2/3
grouped_split.py         train/test split grouped by earthquake
predict.py               CLI for the live demo
results/                 generated metrics and figures
docs/                    write-up (PDF) and review slides
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
| KNN    | 0.973    | 0.896        | 0.804            | 0.874     | 0.821 | 0.906 |

**Binary, all features (+ material, bridge type)**

| Model  | Test acc | Balanced acc | Recall (damaged) | Precision | F1    | AUC   |
|--------|----------|--------------|------------------|-----------|-------|-------|
| LogReg | 0.974    | 0.934        | 0.886            | 0.833     | 0.849 | 0.988 |
| QDA    | 0.968    | 0.906        | 0.833            | 0.802     | 0.803 | 0.986 |
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
| QDA    | 0.927    | 0.622    | 0.962     | 0.553     | 0.513     |
| KNN    | 0.923    | 0.554    | 0.971     | 0.357     | 0.380     |

With only 11 and 10 examples in levels 2 and 3, these numbers are noisy.
Full tables and plots are in `results/`.

## Comparison with the reference report

We checked `data/bridge_data.csv` against the original `ds2.csv` row by row. All 249
rows match on age, magnitude, distance and labels, and the Roman numeral codes map to
material, bridge type and damage level as described in the notes at the bottom of that file.

| Model  | Report test acc | Report acc on damaged | Our test acc | Our recall on damaged |
|--------|-----------------|-----------------------|--------------|-----------------------|
| LogReg | 0.738           | 0.827                 | 0.972        | 0.906                 |
| QDA    | 0.948           | 0.995                 | 0.969        | 0.873                 |
| KNN    | 0.806           | 0.299                 | 0.973        | 0.804                 |

The report picks QDA as the best model. In our runs QDA and Logistic Regression are close,
with Logistic Regression slightly ahead on F1 and AUC. The report also picks k = 2 for KNN by
accuracy, while we pick k by F1 and scale the features first, which is the likely reason KNN
does much better on the damaged class here. The report does not say whether it resampled
before or after splitting, so we cannot fully explain the gap for Logistic Regression.

## Testing on unseen earthquakes

All 21 damaged rows come from just 7 earthquakes (Alaska 1964, San Fernando 1971,
Miyagi 1978, Loma Prieta 1989, Northridge 1994, Kobe 1995, Tohoku 2011), and every row
from those events is labelled damaged. With a random split the model can see Northridge in
training and then be tested on another Northridge row, which mostly checks whether it
remembers the magnitude 6.7.

`grouped_split.py` holds out whole earthquakes instead. In each of 100 runs it puts 2 of
the 7 damaging earthquakes and 30 % of the other earthquake labels in the test set, so no
event is in both. A few names were spelled two ways in the data (e.g. "1971, San Fernendo"
and "1971 San Fernando"), so they are normalised before grouping. Up-sampling ratio is 0.5.

| Model  | Split   | Test acc | Balanced acc | Recall (damaged) | Precision | F1    | AUC   |
|--------|---------|----------|--------------|------------------|-----------|-------|-------|
| LogReg | random  | 0.972    | 0.939        | 0.898            | 0.807     | 0.840 | 0.994 |
| LogReg | grouped | 0.948    | 0.932        | 0.911            | 0.724     | 0.762 | 0.990 |
| QDA    | random  | 0.968    | 0.926        | 0.875            | 0.781     | 0.816 | 0.993 |
| QDA    | grouped | 0.964    | 0.920        | 0.867            | 0.794     | 0.800 | 0.994 |
| KNN    | random  | 0.974    | 0.898        | 0.808            | 0.880     | 0.828 | 0.901 |
| KNN    | grouped | 0.932    | 0.804        | 0.637            | 0.739     | 0.620 | 0.814 |

KNN loses the most, which fits the idea that it was partly matching rows from the same
event. Logistic Regression keeps its recall and AUC but raises more false alarms. QDA barely
changes and is the best model on this stricter test, which agrees with the reference report's
choice. With only 7 damaging events, two held out per run, these numbers still swing a lot
from run to run.

## Limitations

- Only 21 positives, so results vary a lot between splits.
- Negatives rest on the "no report = no damage" assumption, which may be wrong.
- Positives come from famous large earthquakes, so magnitude/distance separate
  the classes easily (AUC ~0.99). Performance on typical new events would
  likely be lower.
- Bootstrap resampling duplicates data; it does not add information.
- The same earthquake appears in many rows, so a random row split lets the
  model see an event in training and then be tested on it. The grouped split
  below measures how much this matters.
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
