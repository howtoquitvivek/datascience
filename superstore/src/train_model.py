
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from features import read_superstore, add_basic_features, select_features

def build_regressor(X_num, X_cat):
    pre = ColumnTransformer([
        ("num","passthrough", X_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X_cat)
    ])
    # Try two models; keep RF by default (robust)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    lr = LinearRegression()
    # Default to RF; you can gridsearch later
    model = rf
    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe

def build_classifier(X_num, X_cat):
    pre = ColumnTransformer([
        ("num","passthrough", X_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X_cat)
    ])
    # Try RandomForest then LogisticRegression
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("model", clf)])
    return pipe

def main(args):
    df = read_superstore(args.csv)
    df = add_basic_features(df)

    task = args.task  # "regress" or "classify"
    target = args.target

    X, y, X_num, X_cat = select_features(df, target, task=task)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task=="classify" else None
    )

    if task == "regress":
        model = build_regressor(X_num, X_cat)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred,)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    else:
        model = build_classifier(X_num, X_cat)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        metrics = {"accuracy": float(acc), "f1": float(f1)}

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{target}_{task}.joblib")
    dump(model, model_path)

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to:", model_path)
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to superstore CSV")
    p.add_argument("--target", type=str, default="Profit", help="Target column (e.g., Profit, Sales, IsProfitable)")
    p.add_argument("--task", type=str, choices=["regress","classify"], default="regress")
    args = p.parse_args()
    main(args)
