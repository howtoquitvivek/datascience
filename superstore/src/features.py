from __future__ import annotations
import pandas as pd
import numpy as np

DATE_COLS = ["Order Date", "Ship Date"]

CATEGORICAL = [
    "Ship Mode","Segment","Country","City","State","Region",
    "Category","Sub-Category","Product ID","Customer ID",
]
NUMERIC = ["Sales","Quantity","Discount","Profit","Postal Code"]

def read_superstore(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse dates (US format like 11/8/2016 -> month/day/year)
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", format="%m/%d/%Y")
    # Clean strings
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip()
    # Basic dedupe
    df = df.drop_duplicates()
    return df

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Order Date" in out and "Ship Date" in out:
        out["ShipDelayDays"] = (out["Ship Date"] - out["Order Date"]).dt.days
    if "Order Date" in out:
        out["OrderYear"] = out["Order Date"].dt.year
        out["OrderMonth"] = out["Order Date"].dt.month
        out["OrderDow"] = out["Order Date"].dt.dayofweek
    # Profitability flag
    if "Profit" in out:
        out["IsProfitable"] = (out["Profit"] > 0).astype(int)
    # Handle missing numerics
    for c in ["Sales","Quantity","Discount","Profit","Postal Code","ShipDelayDays"]:
        if c in out.columns:
            out[c] = out[c].fillna(out[c].median())
    # Clip weird discounts
    if "Discount" in out:
        out["Discount"] = out["Discount"].clip(lower=0, upper=0.8)
    return out

def select_features(df: pd.DataFrame, target: str, task: str = "regress"):
    X_cols_num = [c for c in ["Sales","Quantity","Discount","Postal Code",
                              "ShipDelayDays","OrderYear","OrderMonth","OrderDow"]
                  if c in df.columns and c != target]
    X_cols_cat = [c for c in CATEGORICAL if c in df.columns and c != target]
    X = df[X_cols_num + X_cols_cat].copy()
    y = df[target].copy() if target in df.columns else None

    # Remove leakage: exclude Profit or Sales only if they are not the target
    leakage = {"Profit","Sales"}
    X = X[[c for c in X.columns if (c not in leakage) or (c == target)]]

    if task == "classify":
        # For classification target, ensure y is 0/1
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in dataframe.")
        if sorted(pd.unique(y.dropna())) not in ([0,1], [0], [1]):
            raise ValueError("For classification, target must be binary 0/1.")
    return X, y, X_cols_num, X_cols_cat
