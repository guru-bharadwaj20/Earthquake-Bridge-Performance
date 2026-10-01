ratio 0.5, 100 runs, 2 of 7 damaging earthquakes held out per run

|                                           |   KNN |   LogReg |   QDA |
|:------------------------------------------|------:|---------:|------:|
| ('random split', 'test_acc')              | 0.974 |    0.972 | 0.968 |
| ('random split', 'balanced_acc')          | 0.898 |    0.939 | 0.926 |
| ('random split', 'recall_pos')            | 0.808 |    0.898 | 0.875 |
| ('random split', 'precision')             | 0.88  |    0.807 | 0.781 |
| ('random split', 'f1')                    | 0.828 |    0.84  | 0.816 |
| ('random split', 'auc')                   | 0.901 |    0.994 | 0.993 |
| ('grouped by earthquake', 'test_acc')     | 0.932 |    0.948 | 0.964 |
| ('grouped by earthquake', 'balanced_acc') | 0.804 |    0.932 | 0.92  |
| ('grouped by earthquake', 'recall_pos')   | 0.637 |    0.911 | 0.867 |
| ('grouped by earthquake', 'precision')    | 0.739 |    0.724 | 0.794 |
| ('grouped by earthquake', 'f1')           | 0.62  |    0.762 | 0.8   |
| ('grouped by earthquake', 'auc')          | 0.814 |    0.99  | 0.994 |
