from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_logistic(X_train, y_train, balanced=True):
    if balanced:
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000
        )
    else:
        model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

