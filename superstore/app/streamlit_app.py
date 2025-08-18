import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Superstore Dashboard", layout="wide")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    for c in ["Order Date","Ship Date"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
            except Exception:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

st.sidebar.header("Data")
data_path = st.sidebar.text_input("CSV path", value="data/superstore.csv")
df = load_data(data_path)

st.title("📊 Superstore Dashboard")
st.caption("Filter → KPIs → Charts")

# Filters
min_date = df["Order Date"].min() if "Order Date" in df else None
max_date = df["Order Date"].max() if "Order Date" in df else None
if min_date is not None and max_date is not None:
    date_range = st.sidebar.date_input("Order Date range", value=(min_date, max_date))
    if isinstance(date_range, tuple) and len(date_range)==2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["Order Date"]>=start) & (df["Order Date"]<=end)]

state_opt = ["All"] + sorted(df["State"].dropna().unique().tolist()) if "State" in df else ["All"]
state = st.sidebar.selectbox("State", options=state_opt, index=0)
if state != "All" and "State" in df:
    df = df[df["State"]==state]

category_opt = ["All"] + sorted(df["Category"].dropna().unique().tolist()) if "Category" in df else ["All"]
category = st.sidebar.selectbox("Category", options=category_opt, index=0)
if category != "All" and "Category" in df:
    df = df[df["Category"]==category]

# KPIs
total_sales = df["Sales"].sum() if "Sales" in df else 0.0
total_profit = df["Profit"].sum() if "Profit" in df else 0.0
orders = df.shape[0]
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Orders", f"{orders:,}")

# Charts
if "Order Date" in df and "Sales" in df:
    m = df.copy()
    m["Month"] = m["Order Date"].dt.to_period("M").astype(str)
    monthly = m.groupby("Month", as_index=False)["Sales"].sum().sort_values("Month")
    fig = px.line(monthly, x="Month", y="Sales", title="Monthly Sales")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    if "Category" in df and "Sales" in df:
        cat_sales = df.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
        st.plotly_chart(px.bar(cat_sales, x="Category", y="Sales", title="Sales by Category"), use_container_width=True)
with right:
    if "Sub-Category" in df and "Sales" in df:
        sub_sales = df.groupby("Sub-Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(10)
        st.plotly_chart(px.bar(sub_sales, x="Sub-Category", y="Sales", title="Top 10 Sub-Categories"), use_container_width=True)

# Profitability by State (top 15)
if "State" in df and "Profit" in df:
    state_profit = df.groupby("State", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False).head(15)
    st.plotly_chart(px.bar(state_profit, x="State", y="Profit", title="Top States by Profit"), use_container_width=True)

st.caption("Tip: Train a model with `python src/train_model.py --csv data/superstore.csv --target Profit`.")
