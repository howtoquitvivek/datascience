
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

def parse_dates(df):
    for c in ["Order Date","Ship Date"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
            except Exception:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def add_features(df):
    out = df.copy()
    if "Order Date" in out:
        out["OrderYear"] = out["Order Date"].dt.year
        out["OrderMonth"] = out["Order Date"].dt.to_period("M").astype(str)
    if "Ship Date" in out and "Order Date" in out:
        out["ShipDelayDays"] = (out["Ship Date"] - out["Order Date"]).dt.days
    if "Profit" in out:
        out["IsProfitable"] = (out["Profit"] > 0).astype(int)
    return out

def save_plot(fig, outdir, name):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(outdir, name), bbox_inches="tight")
    plt.close(fig)

def main(csv_path):
    outdir = "reports/eda_figures"
    df = pd.read_csv(csv_path)
    df = parse_dates(df)
    df = add_features(df)

    # Summary
    df.describe(include="all").to_csv("reports/eda_describe.csv")

    # Sales over time (monthly)
    if "OrderMonth" in df:
        g = df.groupby("OrderMonth")["Sales"].sum().sort_index()
        fig, ax = plt.subplots()
        g.plot(ax=ax)
        ax.set_title("Monthly Sales")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        save_plot(fig, outdir, "monthly_sales.png")

    # Profit distribution
    if "Profit" in df:
        fig, ax = plt.subplots()
        df["Profit"].dropna().plot(kind="hist", bins=50, ax=ax)
        ax.set_title("Profit Distribution")
        ax.set_xlabel("Profit")
        save_plot(fig, outdir, "profit_distribution.png")

    # Sales by Category
    if "Category" in df:
        g = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        g.plot(kind="bar", ax=ax)
        ax.set_title("Sales by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Sales")
        save_plot(fig, outdir, "sales_by_category.png")

    # Top 10 Sub-Categories by Sales
    if "Sub-Category" in df:
        g = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        g.plot(kind="bar", ax=ax)
        ax.set_title("Top 10 Sub-Categories by Sales")
        ax.set_xlabel("Sub-Category")
        ax.set_ylabel("Sales")
        save_plot(fig, outdir, "top10_subcategories_sales.png")

    # Ship delay by ship mode
    if "Ship Mode" in df and "ShipDelayDays" in df:
        g = df.groupby("Ship Mode")["ShipDelayDays"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots()
        g.plot(kind="bar", ax=ax)
        ax.set_title("Average Ship Delay by Ship Mode")
        ax.set_xlabel("Ship Mode")
        ax.set_ylabel("Avg Delay (days)")
        save_plot(fig, outdir, "ship_delay_by_shipmode.png")

    print("EDA complete. Figures saved to", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to superstore CSV")
    args = ap.parse_args()
    main(args.csv)
