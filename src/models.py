from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

def make_qda():
    return QuadraticDiscriminantAnalysis(reg_param=0.1)

def make_knn(k):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))

def select_k(X, y, k_range=range(1, 16), seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = {k: cross_val_score(make_knn(k), X, y, cv=cv, scoring="f1").mean()
              for k in k_range}
    return max(scores, key=scores.get), scores
