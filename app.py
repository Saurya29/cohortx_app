# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from cohort import (
    prepare_orders, build_cohorts, retention_tables, revenue_table,
    aov_table, items_per_order_table, order_frequency_table,
    rfm, auto_insights, cohort_sizes_series, arpu_table,
    repeat_rate_over_time, first_vs_returning_revenue
)

st.set_page_config(page_title="CohortX Pro — Customer Analytics", layout="wide")
st.markdown("# CohortX Pro — Customer Analytics")
st.caption("Upload your CSV and explore cohorts, revenue, ARPU, repeat rate, and mix.")

# --- Sidebar: upload & filters ---
with st.sidebar:
    st.header("1) Upload Data")
    file = st.file_uploader("CSV or Parquet", type=["csv","parquet"])
    st.caption("If none uploaded, a sample dataset will be used.")
    st.header("2) Global Filters")
    date_from = st.date_input("Start date", value=None)
    date_to = st.date_input("End date", value=None)
    min_cohort_size = st.number_input("Min cohort size", 1, 10000, 20, 1)
    st.header("3) Downloads")
    allow_downloads = st.checkbox("Enable table downloads", value=True)

@st.cache_data(show_spinner=False)
def load_df(file):
    if file is None:
        return None
    if file.name.endswith(".parquet"):
        return pd.read_parquet(file)
    return pd.read_csv(file)

def load_sample():
    return pd.read_csv("data/sample_transactions.csv")

# Load
if file is not None:
    raw = load_df(file)
else:
    raw = load_sample()
    st.info("Using bundled sample data. Upload to use your own.")

# Auto schema detection
cols_lower = {c.lower(): c for c in raw.columns}
def pick(*names):
    for n in names:
        if n in cols_lower: return cols_lower[n]
    return None

date_col = pick("order_date","date","created_at")
cust_col = pick("cust_id","customer_id","user_id")
order_col = pick("order_id","invoice_id","transaction_id")
qty_col = pick("qty_ordered","quantity","qty")
rev_col = pick("total","value","revenue","grand_total","amount")
unit_price_col = pick("price","unit_price","unitprice")

category_col = pick("category","product_category","cat")
payment_col = pick("payment_method","payment","pay_method")
gender_col = pick("gender","sex")
region_col = pick("region")
state_col = pick("state")
city_col = pick("city")

# Preview
with st.expander("Preview DataFrame"):
    st.dataframe(raw.head(15), use_container_width=True)

# Apply filters
df_dim = raw.copy()
if category_col:
    cats = ["All"] + sorted(list(pd.Series(df_dim[category_col].astype(str).unique()).dropna()))
    sel_cat = st.sidebar.selectbox("Category", cats, index=0)
    if sel_cat != "All":
        df_dim = df_dim[df_dim[category_col].astype(str) == sel_cat]

if payment_col:
    pays = ["All"] + sorted(list(pd.Series(df_dim[payment_col].astype(str).unique()).dropna()))
    sel_pay = st.sidebar.selectbox("Payment Method", pays, index=0)
    if sel_pay != "All":
        df_dim = df_dim[df_dim[payment_col].astype(str) == sel_pay]

if gender_col:
    gens = ["All"] + sorted(list(pd.Series(df_dim[gender_col].astype(str).unique()).dropna()))
    sel_gen = st.sidebar.selectbox("Gender", gens, index=0)
    if sel_gen != "All":
        df_dim = df_dim[df_dim[gender_col].astype(str) == sel_gen]

# Date filter
if date_col:
    df_dim[date_col] = pd.to_datetime(df_dim[date_col], errors="coerce")
    if date_from:
        df_dim = df_dim[df_dim[date_col] >= pd.to_datetime(date_from)]
    if date_to:
        df_dim = df_dim[df_dim[date_col] <= pd.to_datetime(date_to)]

# Build cohort-ready orders
orders = prepare_orders(
    df_dim, date_col=date_col, customer_col=cust_col, order_col=order_col,
    quantity_col=qty_col, revenue_col=rev_col, unit_price_col=unit_price_col
)
orders = build_cohorts(orders)

# KPIs
total_customers = orders["customer_id"].nunique()
total_orders = orders["order_id"].nunique()
total_revenue = orders["revenue"].sum(skipna=True)
repeat_customers = (orders.groupby("customer_id")["order_id"].nunique()>1).sum()
repeat_rate = repeat_customers / total_customers if total_customers else 0.0
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{total_customers:,}")
col2.metric("Orders", f"{total_orders:,}")
col3.metric("Revenue", f"{total_revenue:,.0f}")
col4.metric("Repeat Rate", f"{repeat_rate*100:.1f}%")

# Tables
counts_wide, rate_wide, cohort_sizes = retention_tables(orders)
eligible = cohort_sizes[cohort_sizes["cohort_size"] >= min_cohort_size].index
counts_wide = counts_wide.loc[counts_wide.index.isin(eligible)]
rate_wide = rate_wide.loc[rate_wide.index.isin(eligible)]
rev_wide = revenue_table(orders)
aov_wide = aov_table(orders)
ipo_wide = items_per_order_table(orders)
freq_wide = order_frequency_table(orders)

ins = auto_insights(rate_wide, rev_wide)
if ins:
    st.success(" • ".join(ins))

cohort_trend = cohort_sizes_series(orders)
arpu_wide = arpu_table(orders)
repeat_wide = repeat_rate_over_time(orders)
first_return_overall, first_return_pivot = first_vs_returning_revenue(orders)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Retention", "Revenue", "Frequency & AOV", "RFM Segments", "Demographics", "Metrics & Mix"
])

with tab1:
    st.subheader("Retention Heatmap (Rate)")
    if not rate_wide.empty:
        hm = rate_wide.reset_index().melt(id_vars="cohort_month", var_name="cohort_index", value_name="rate")
        hm["cohort_month"] = hm["cohort_month"].astype(str)
        fig = px.density_heatmap(hm, x="cohort_index", y="cohort_month", z="rate",
                                 histfunc="avg", labels={"rate":"Retention"})
        fig.update_coloraxes(colorbar_title="Rate")
        st.plotly_chart(fig, use_container_width=True, key="retention_heatmap")

        st.markdown("### Cohort size (customers acquired)")
        cs = cohort_trend.reset_index()
        cs.columns = ["cohort_month","customers"]
        cs["cohort_month"] = cs["cohort_month"].astype(str)
        fig_cs = px.bar(cs, x="cohort_month", y="customers", labels={"customers":"Customers","cohort_month":"Cohort"})
        fig_cs.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cs, use_container_width=True, key="cohort_size_trend")

        st.subheader("Retention Curves")
        options = [str(i) for i in rate_wide.index.astype(str).tolist()]
        sel = st.multiselect("Select cohorts", options, default=options[:3])
        curve = rate_wide.copy(); curve.index = curve.index.astype(str)
        curve = curve.loc[sel] if sel else curve
        curve = curve.T.reset_index().rename(columns={"index":"cohort_index"}).melt(id_vars="cohort_index", var_name="cohort", value_name="retention")
        fig2 = px.line(curve, x="cohort_index", y="retention", color="cohort")
        st.plotly_chart(fig2, use_container_width=True, key="retention_curves")
    else:
        st.warning("No cohorts after filters. Try lowering min cohort size.")

with tab2:
    st.subheader("Revenue Heatmap")
    if not rev_wide.empty:
        hm2 = rev_wide.reset_index().melt(id_vars="cohort_month", var_name="cohort_index", value_name="revenue")
        hm2["cohort_month"] = hm2["cohort_month"].astype(str)
        fig3 = px.density_heatmap(hm2, x="cohort_index", y="cohort_month", z="revenue", histfunc="avg", labels={"revenue":"Revenue"})
        fig3.update_coloraxes(colorbar_title="Revenue")
        st.plotly_chart(fig3, use_container_width=True, key="revenue_heatmap")
        st.caption("Revenue is aggregated per cohort-month.")
    else:
        st.info("Revenue not available (no revenue column detected).")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average Order Value (AOV)")
        ha = aov_wide.reset_index().melt(id_vars="cohort_month", var_name="cohort_index", value_name="aov")
        ha["cohort_month"] = ha["cohort_month"].astype(str)
        fig4 = px.density_heatmap(ha, x="cohort_index", y="cohort_month", z="aov", histfunc="avg", labels={"aov":"AOV"})
        st.plotly_chart(fig4, use_container_width=True, key="aov_heatmap")
    with c2:
        st.subheader("Items per Order")
        hi = ipo_wide.reset_index().melt(id_vars="cohort_month", var_name="cohort_index", value_name="items")
        hi["cohort_month"] = hi["cohort_month"].astype(str)
        fig5 = px.density_heatmap(hi, x="cohort_index", y="cohort_month", z="items", histfunc="avg", labels={"items":"Items/Order"})
        st.plotly_chart(fig5, use_container_width=True, key="items_per_order_heatmap")
    st.subheader("Average Orders per Customer (Frequency)")
    hf = freq_wide.reset_index().melt(id_vars="cohort_month", var_name="cohort_index", value_name="avg_orders")
    hf["cohort_month"] = hf["cohort_month"].astype(str)
    fig6 = px.density_heatmap(hf, x="cohort_index", y="cohort_month", z="avg_orders", histfunc="avg", labels={"avg_orders":"Avg Orders/Customer"})
    st.plotly_chart(fig6, use_container_width=True, key="avg_orders_heatmap")

with tab4:
    st.subheader("RFM Segmentation")
    rfm_df = rfm(orders)
    st.dataframe(rfm_df.head(20), use_container_width=True)
    seg_counts = rfm_df["Segment_Label"].value_counts().reset_index()
    seg_counts.columns = ["Segment","Customers"]
    fig7 = px.bar(seg_counts, x="Segment", y="Customers")
    st.plotly_chart(fig7, use_container_width=True, key="rfm_bar")

with tab5:
    st.subheader("Demographics & Mix")
    if category_col and category_col in raw.columns and rev_col and rev_col in raw.columns:
        cat_rev = raw.groupby(category_col)[rev_col].sum().reset_index().sort_values(rev_col, ascending=False).head(15)
        fig8 = px.bar(cat_rev, x=category_col, y=rev_col)
        st.plotly_chart(fig8, use_container_width=True, key="category_mix_tab5")
    if payment_col and payment_col in raw.columns:
        pay_ct = raw[payment_col].value_counts().reset_index()
        pay_ct.columns = ["Payment Method","Orders"]
        fig9 = px.pie(pay_ct, names="Payment Method", values="Orders")
        st.plotly_chart(fig9, use_container_width=True, key="payment_mix_tab5")
    if gender_col and gender_col in raw.columns:
        g_ct = raw[gender_col].value_counts().reset_index()
        g_ct.columns = ["Gender","Customers"]
        fig10 = px.bar(g_ct, x="Gender", y="Customers")
        st.plotly_chart(fig10, use_container_width=True, key="gender_bar_tab5")

with tab6:
    st.subheader("Metrics & Mix")
    st.markdown("### ARPU (Revenue ÷ Active Customers)")
    if not arpu_wide.empty:
        arpu_h = arpu_wide.reset_index().melt(id_vars="cohort_month", var_name="cohort_index", value_name="arpu")
        arpu_h["cohort_month"] = arpu_h["cohort_month"].astype(str)
        fig_arpu = px.density_heatmap(arpu_h, x="cohort_index", y="cohort_month", z="arpu", histfunc="avg", labels={"arpu":"ARPU"})
        st.plotly_chart(fig_arpu, use_container_width=True, key="arpu_heatmap_tab6")
    else:
        st.info("ARPU not available (needs revenue + active customer data).")

    st.markdown("### First vs Returning Revenue")
    try:
        overall = first_return_overall
        total_rev = overall.sum()
        if total_rev>0:
            labels = ["First Purchase", "Returning"]
            vals = [float(overall["first_purchase"]), float(overall["returning"])]
            fig_pie = px.pie(values=vals, names=labels, title="Revenue Share")
            st.plotly_chart(fig_pie, use_container_width=True, key="first_vs_returning_pie_tab6")
        if not first_return_pivot.empty:
            fr = first_return_pivot.reset_index()
            sample_cohorts = fr["cohort_month"].drop_duplicates().astype(str).tolist()[:8]
            fr_small = fr[fr["cohort_month"].astype(str).isin(sample_cohorts)]
            st.dataframe(fr_small.head(20), use_container_width=True)
    except Exception:
        st.info("Not enough revenue data to compute first vs returning split.")

    st.markdown("### Repeat Purchase Rate (cumulative share of customers with >=2 orders)")
    if not repeat_wide.empty:
        rpt = repeat_wide.reset_index().melt(id_vars="cohort_month", var_name="cohort_index", value_name="repeat_rate")
        rpt["cohort_month"] = rpt["cohort_month"].astype(str)
        fig_rpt = px.line(rpt, x="cohort_index", y="repeat_rate", color="cohort_month", labels={"repeat_rate":"Repeat Rate"})
        st.plotly_chart(fig_rpt, use_container_width=True, key="repeat_rate_line_tab6")
    else:
        st.info("Repeat rate requires at least two orders per some customers.")

    st.markdown("### Category & Payment Mix (if available)")
    if category_col and category_col in raw.columns and rev_col and rev_col in raw.columns:
        cat_rev = raw.groupby(category_col)[rev_col].sum().reset_index().sort_values(rev_col, ascending=False).head(10)
        fig_cat = px.bar(cat_rev, x=category_col, y=rev_col, labels={rev_col:"Revenue"})
        st.plotly_chart(fig_cat, use_container_width=True, key="category_mix_tab6")
    if payment_col and payment_col in raw.columns:
        pay_ct = raw[payment_col].value_counts().reset_index()
        pay_ct.columns = ["Payment Method","Orders"]
        fig_pay = px.pie(pay_ct, names="Payment Method", values="Orders")
        st.plotly_chart(fig_pay, use_container_width=True, key="payment_mix_tab6")

st.markdown("---")
st.subheader("Raw Data / Export")
st.dataframe(raw.head(100), use_container_width=True)
if allow_downloads:
    st.download_button("Download retention rates (CSV)", rate_wide.to_csv().encode("utf-8"), "retention_rates.csv")
    st.download_button("Download retention counts (CSV)", counts_wide.to_csv().encode("utf-8"), "retention_counts.csv")
    if not rev_wide.empty:
        st.download_button("Download revenue by cohort (CSV)", rev_wide.to_csv().encode("utf-8"), "revenue_by_cohort.csv")
    try:
        with pd.ExcelWriter("cohortx_export.xlsx", engine="openpyxl") as xw:
            rate_wide.to_excel(xw, sheet_name="retention_rates")
            counts_wide.to_excel(xw, sheet_name="retention_counts")
            aov_wide.to_excel(xw, sheet_name="aov")
            ipo_wide.to_excel(xw, sheet_name="items_per_order")
            freq_wide.to_excel(xw, sheet_name="frequency")
        with open("cohortx_export.xlsx","rb") as f:
            st.download_button("Download Excel Report", f.read(), "cohortx_export.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.info("Excel export requires openpyxl installed.")
