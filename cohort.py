# cohort.py
import pandas as pd
import numpy as np
from typing import Tuple

# --- Utilities ---
def _month_start(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return s.values.astype("datetime64[M]")

def _std_rename(df: pd.DataFrame, date_col, customer_col, order_col):
    rename_map = {}
    if date_col: rename_map[date_col] = "order_date"
    if customer_col: rename_map[customer_col] = "customer_id"
    if order_col: rename_map[order_col] = "order_id"
    return df.rename(columns=rename_map)

# --- Ingestion & Cohorts ---
def prepare_orders(df: pd.DataFrame, date_col: str, customer_col: str, order_col: str,
                   quantity_col=None, revenue_col=None, unit_price_col=None) -> pd.DataFrame:
    df = _std_rename(df.copy(), date_col, customer_col, order_col)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["month"] = _month_start(df["order_date"])
    df = df.dropna(subset=["order_date", "customer_id", "order_id"])

    if quantity_col and quantity_col in df.columns:
        df["quantity"] = pd.to_numeric(df[quantity_col], errors="coerce").fillna(0)
    else:
        df["quantity"] = 1.0

    if revenue_col and revenue_col in df.columns:
        revenue = pd.to_numeric(df[revenue_col], errors="coerce")
    elif unit_price_col and unit_price_col in df.columns:
        revenue = pd.to_numeric(df[unit_price_col], errors="coerce") * df["quantity"]
    else:
        revenue = np.nan
    df["revenue"] = revenue

    orders = (
        df.groupby(["customer_id", "order_id"], as_index=False)
          .agg(order_date=("order_date","min"),
               month=("month","min"),
               quantity=("quantity","sum"),
               revenue=("revenue","sum"))
    )
    return orders

def build_cohorts(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    first_purchase = orders.groupby("customer_id")["month"].min().rename("cohort_month")
    orders = orders.merge(first_purchase, on="customer_id", how="left")
    year_diff = orders["month"].dt.year - orders["cohort_month"].dt.year
    month_diff = orders["month"].dt.month - orders["cohort_month"].dt.month
    orders["cohort_index"] = year_diff * 12 + month_diff
    return orders

def retention_tables(orders: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    active = (orders.groupby(["cohort_month","cohort_index"])["customer_id"]
                    .nunique().rename("active").reset_index())
    sizes = (orders.groupby("cohort_month")["customer_id"].nunique()
                   .rename("cohort_size").reset_index())
    df = active.merge(sizes, on="cohort_month", how="left")
    df["retention_rate"] = df["active"] / df["cohort_size"]
    counts = df.pivot(index="cohort_month", columns="cohort_index", values="active").fillna(0).astype(int)
    rates = df.pivot(index="cohort_month", columns="cohort_index", values="retention_rate").fillna(0.0)
    return counts, rates, sizes.set_index("cohort_month")

def revenue_table(orders: pd.DataFrame) -> pd.DataFrame:
    rev = (orders.groupby(["cohort_month","cohort_index"])["revenue"]
                 .sum().reset_index())
    return rev.pivot(index="cohort_month", columns="cohort_index", values="revenue").fillna(0.0)

def aov_table(orders: pd.DataFrame) -> pd.DataFrame:
    aov = (orders.groupby(["cohort_month","cohort_index"])["revenue"]
                 .mean().reset_index())
    return aov.pivot(index="cohort_month", columns="cohort_index", values="revenue").astype(float)

def items_per_order_table(orders: pd.DataFrame) -> pd.DataFrame:
    ipo = (orders.groupby(["cohort_month","cohort_index"])["quantity"]
                 .mean().reset_index())
    return ipo.pivot(index="cohort_month", columns="cohort_index", values="quantity").astype(float)

def order_frequency_table(orders: pd.DataFrame) -> pd.DataFrame:
    tmp = orders.groupby(["cohort_month","cohort_index","customer_id"])["order_id"].nunique().reset_index(name="orders")
    freq = tmp.groupby(["cohort_month","cohort_index"])["orders"].mean().reset_index()
    return freq.pivot(index="cohort_month", columns="cohort_index", values="orders").fillna(0.0)

def cohort_sizes_series(orders: pd.DataFrame) -> pd.Series:
    return orders.groupby("cohort_month")["customer_id"].nunique().sort_index()

def arpu_table(orders: pd.DataFrame) -> pd.DataFrame:
    counts, _, _ = retention_tables(orders)
    rev = revenue_table(orders)
    rev = rev.reindex(index=counts.index, columns=counts.columns, fill_value=0.0)
    active = counts.astype(float).replace(0, np.nan)
    return (rev / active).fillna(0.0)

def repeat_rate_over_time(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    max_k = int(orders["cohort_index"].max()) if not orders["cohort_index"].empty else 0
    rows = []
    for cohort, group in orders.groupby("cohort_month"):
        custs = group["customer_id"].unique()
        sub = orders[orders["customer_id"].isin(custs)]
        for k in range(0, max_k+1):
            subk = sub[sub["cohort_index"] <= k]
            orders_per_customer = subk[subk["cohort_month"]==cohort].groupby("customer_id")["order_id"].nunique()
            pct = (orders_per_customer >= 2).mean() if len(orders_per_customer)>0 else 0.0
            rows.append({"cohort_month": cohort, "cohort_index": k, "repeat_rate": pct})
    df = pd.DataFrame(rows)
    return df.pivot(index="cohort_month", columns="cohort_index", values="repeat_rate").fillna(0.0) if not df.empty else pd.DataFrame()

def first_vs_returning_revenue(orders: pd.DataFrame):
    first = orders[orders["cohort_index"] == 0]["revenue"].sum(skipna=True)
    returning = orders[orders["cohort_index"] > 0]["revenue"].sum(skipna=True)
    overall = pd.Series({"first_purchase": first, "returning": returning})
    cohort_rev = orders.groupby(["cohort_month","cohort_index"])["revenue"].sum().reset_index()
    cohort_rev["type"] = cohort_rev["cohort_index"].apply(lambda x: "first" if x==0 else "returning")
    pivot = cohort_rev.pivot_table(index=["cohort_month","cohort_index"], columns="type", values="revenue", fill_value=0.0)
    return overall, pivot

def rfm(df_orders: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
    if as_of_date is None:
        as_of_date = df_orders["order_date"].max()
    r = df_orders.groupby("customer_id")["order_date"].max().apply(lambda d: (as_of_date - d).days)
    f = df_orders.groupby("customer_id")["order_id"].nunique()
    m = df_orders.groupby("customer_id")["revenue"].sum()
    rfm = pd.DataFrame({"RecencyDays": r, "Frequency": f, "Monetary": m})
    rfm["R_Score"] = pd.qcut(rfm["RecencyDays"].rank(method="first"), 5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["Segment_Label"] = rfm.apply(lambda row:
        "Champions" if row["R_Score"]>=4 and row["F_Score"]>=4 and row["M_Score"]>=4 else
        "Loyal" if row["R_Score"]>=4 and row["F_Score"]>=3 else
        "Big Spenders" if row["R_Score"]>=3 and row["M_Score"]>=4 else
        "At Risk" if row["R_Score"]<=2 and row["F_Score"]<=2 else "Others", axis=1)
    return rfm.reset_index()

def auto_insights(rate_wide: pd.DataFrame, revenue_wide: pd.DataFrame) -> list:
    tips = []
    try:
        if not rate_wide.empty and 3 in rate_wide.columns:
            best = rate_wide.sort_values(by=3, ascending=False).head(1)
            if not best.empty:
                cohort = best.index[0]
                val = float(best[3].iloc[0])*100
                tips.append(f"Highest 3-month retention is {val:.1f}% for cohort {cohort.strftime('%Y-%m')}.")
        avg_curve = rate_wide.mean(axis=0)
        if len(avg_curve)>=2 and avg_curve.iloc[1] < avg_curve.iloc[0]*0.6:
            tips.append("Sharp drop after month 1 — consider onboarding or 2nd-order incentives.")
        if not revenue_wide.empty and 1 in revenue_wide.columns:
            top = revenue_wide[1].idxmax()
            tips.append(f"Cohort {top.strftime('%Y-%m')} has the highest month-1 revenue.")
    except Exception:
        pass
    return tips
