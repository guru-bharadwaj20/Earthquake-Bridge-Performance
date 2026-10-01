# Contributing

Two people, one repo. Rohan built the data pipeline, models and experiments. Guru takes the repo, verification, the extra grouped-split experiment and the written deliverables.

✅ done &nbsp;&nbsp; ❌ not done

## Part 1: Chukkapalli Rohan (PES1UG24CS135)

Data, models, experiments and the demo CLI.

| # | Task | Subtask | Status |
|---|------|---------|:------:|
| 1 | Data | Collect the 249 bridge/earthquake records from the reference dataset into `bridge_data.csv` | ✅ |
| 2 | | Clean and relabel material and bridge type with readable names | ✅ |
| 3 | | Add the damage level column (1 none, 2 moderate/severe, 3 collapse) | ✅ |
| 4 | Preprocessing | Stratified 70/30 split | ✅ |
| 5 | | Bootstrap up-sampling of damaged examples on the training set only | ✅ |
| 6 | | Feature scaling for LogReg and KNN, one-hot material and bridge type | ✅ |
| 7 | Models | Logistic Regression, regularised QDA, KNN with k picked by 5-fold CV on F1 | ✅ |
| 8 | Binary experiment | 100 splits x 3 ratios on core and all features | ✅ |
| 9 | | Accuracy, balanced accuracy, recall, precision, F1, ROC AUC | ✅ |
| 10 | | Plots: data scatter, metrics vs ratio, AUC boxplot | ✅ |
| 11 | Extension | Multi-class damage level with confusion matrices | ✅ |
| 12 | Demo | `predict.py` CLI and saved best models | ✅ |
| 13 | Docs | First README with setup, run steps, results and limitations | ✅ |
| 14 | Review | Code walkthrough and live demo of `predict.py` during the review | ❌ |

## Part 2: Guru R Bharadwaj (PES1UG24CS177)

Repo, checks, one more experiment and the final deliverables.

| # | Task | Subtask | Status |
|---|------|---------|:------:|
| 1 | Repo setup | Private GitHub repo, push code, share with faculty and TAs | ❌ |
| 2 | | LICENSE and CONTRIBUTING | ❌ |
| 3 | | README: team names, SRNs and the problem statement number | ✅ |
| 4 | Verification | Fresh setup from README and rerun every script | ❌ |
| 5 | | Check rerun numbers against the tables in `results/` | ❌ |
| 6 | | Compare our results and method with the reference paper | ❌ |
| 7 | Grouped split | Split train/test by earthquake so no event appears in both | ❌ |
| 8 | | Rerun the three models and compare with the random split | ❌ |
| 9 | | Add the table and a short note to the README | ❌ |
| 10 | Write-up | Two page PDF: problem, dataset, approach, implementation, conclusions | ❌ |
| 11 | Slides | Deck for the review using the results and plots | ❌ |
| 12 | Review | Present the slides and take the Q&A on results and limitations | ❌ |

## Workflow

- Branch off `main` for each task, e.g. `grouped-split`, `write-up`
- Keep commits small with clear messages
- Open a PR and get the other person to look at it before merging
- Don't commit virtual environments or `__pycache__`
- Flip ❌ to ✅ in the table in the same PR that finishes the task
