# CohortX — Customer Retention Dashboard

A minimal, interview-ready **cohort analysis** web app built with **Streamlit**.

- Upload your CSV, map columns, and instantly get **retention heatmaps**, **cohort curves**, **AOV**, and **items per order**.
- Clean UX, fast, and deployable in minutes.

## Quick Start (Local)

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown (usually http://localhost:8501).

## Dataset Requirements

Your CSV can be line-items or orders. The app will aggregate to orders. Minimum columns:
- **order_date** (date or datetime)
- **customer_id**
- **order_id**

Optional:
- **quantity** (defaults to 1)
- **revenue** (or **unit_price** × quantity)

You can **rename/choose** the exact column names in the sidebar.

## Features

- Retention **heatmap** by **cohort month × months since acquisition**
- **Retention curves** for selected cohorts
- **AOV** heatmap (if revenue available)
- **Items per order** heatmap
- **CSV downloads** for computed tables
- Small **sample dataset**: `data/sample_transactions.csv`

## Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo.
2. Go to share.streamlit.io → **New app**.
3. Select your repo, set **Main file path** to `app.py`.
4. Deploy.

## Deploy (Hugging Face Spaces)

Create a **Streamlit** Space, upload these files, set `app.py` as the entry file.

## Structure

```
cohortx_app/
  app.py
  cohort.py
  requirements.txt
  data/
    sample_transactions.csv
```

---

Made for fast demos and real discussions with interviewers.
