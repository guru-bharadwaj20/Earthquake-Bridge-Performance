# Contributing

Rohan built the data pipeline, models and experiments. Guru takes the repo, verification, the extra grouped-split experiment and the written deliverables.

## Part 1: Chukkapalli Rohan (PES1UG24CS135)

Data, models, experiments and the demo CLI.

| # | Task | Subtask | Status |
|---|------|---------|--------|
| 1 | Data | Collect the 249 bridge/earthquake records from the reference dataset into `bridge_data.csv` | Done |
| 2 | | Clean and relabel material and bridge type with readable names | Done |
| 3 | | Add the damage level column (1 none, 2 moderate/severe, 3 collapse) | Done |
| 4 | Preprocessing | Stratified 70/30 split | Done |
| 5 | | Bootstrap up-sampling of damaged examples on the training set only | Done |
| 6 | | Feature scaling for LogReg and KNN, one-hot material and bridge type | Done |
| 7 | Models | Logistic Regression, regularised QDA, KNN with k picked by 5-fold CV on F1 | Done |
| 8 | Binary experiment | 100 splits x 3 ratios on core and all features | Done |
| 9 | | Accuracy, balanced accuracy, recall, precision, F1, ROC AUC | Done |
| 10 | | Plots: data scatter, metrics vs ratio, AUC boxplot | Done |
| 11 | Extension | Multi-class damage level with confusion matrices | Done |
| 12 | Demo | `predict.py` CLI and saved best models | Done |
| 13 | Docs | First README with setup, run steps, results and limitations | Done |
| 14 | Review | Code walkthrough and live demo of `predict.py` during the review | Pending |

## Part 2: Guru R Bharadwaj (PES1UG24CS177)

Repo, checks, one more experiment and the final deliverables.

| # | Task | Subtask | Status |
|---|------|---------|--------|
| 1 | Repo setup | Private GitHub repo, push code, share with faculty and TAs | Pending |
| 2 | | LICENSE and CONTRIBUTING | Done |
| 3 | | README: team names, SRNs and the problem statement number | Done |
| 4 | Verification | Fresh setup from README and rerun every script | Done |
| 5 | | Check rerun numbers against the tables in `results/` | Done |
| 6 | | Compare our results and method with the reference paper | Done |
| 7 | Grouped split | Split train/test by earthquake so no event appears in both | Done |
| 8 | | Rerun the three models and compare with the random split | Done |
| 9 | | Add the table and a short note to the README | Done |
| 10 | Write-up | Two page PDF: problem, dataset, approach, implementation, conclusions | Done |
| 11 | Slides | Deck for the review using the results and plots | Done |
| 12 | Review | Present the slides and take the Q&A on results and limitations | Pending |

## Workflow

- Small commits with clear messages
- No virtual environments or `__pycache__` in the repo
- Update the status column when a task is finished
