from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def make_model(name, numeric_cols, categorical_cols, seed):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ], remainder="drop")

    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", clf),
    ])
