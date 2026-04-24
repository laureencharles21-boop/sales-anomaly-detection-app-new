# ===============================
# IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Merchant Anomaly Detection",
    layout="wide"
)

st.title("📊 Merchant Anomaly Detection Dashboard - New")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("Transactions_New.csv")
    merchant_df = pd.read_csv("merchant_New.csv")
    business_df = pd.read_csv("business_New.csv")
    status_df = pd.read_csv("status_New.csv")

    # Clean column names
    df.columns = df.columns.str.strip()
    merchant_df.columns = merchant_df.columns.str.strip()
    business_df.columns = business_df.columns.str.strip()
    status_df.columns = status_df.columns.str.strip()

    # Merge datasets
    df = df.merge(merchant_df, on="Merchant_ID", how="left")
    df = df.merge(business_df, on="Business_ID", how="left")
    df = df.merge(status_df, on="Status_Code", how="left")

    # Filter only valid transactions
    df = df[df["Status_Code"] == 1]

    # Preprocessing
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
    df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce')

    df = df.dropna()

    return df

# Load data
with st.spinner("Loading data..."):
    df = load_data()

# ===============================
# SIDEBAR - MERCHANT SELECTION
# ===============================
st.sidebar.header("🔎 Filters")

merchant_list = sorted(df["Merchant"].unique())
selected_merchant = st.sidebar.selectbox("Select Merchant", merchant_list)

# ===============================
# FILTER DATA
# ===============================
merchant_data = df[df["Merchant"] == selected_merchant]

daily_data = merchant_data.groupby("Date")["Amount"].sum().reset_index()

# ===============================
# ANOMALY DETECTION (TIME SERIES)
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(daily_data[["Amount"]])

model = IsolationForest(contamination=0.05, random_state=42)
daily_data["Anomaly"] = model.fit_predict(X_scaled)

anomalies = daily_data[daily_data["Anomaly"] == -1]

# ===============================
# KPIs
# ===============================
col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{daily_data['Amount'].sum():,.0f}")
col2.metric("Total Transactions", len(merchant_data))
col3.metric("Anomaly Days", len(anomalies))

# ===============================
# DAILY SALES PLOT
# ===============================
st.subheader(f"📈 Daily Sales Pattern - {selected_merchant}")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(daily_data["Date"], daily_data["Amount"], label="Sales", linewidth=2)

# Highlight anomalies
ax.scatter(anomalies["Date"], anomalies["Amount"], color='red', label="Anomaly", zorder=5)

# Annotate anomalies
for i in range(len(anomalies)):
    ax.annotate("Anomaly",
                (anomalies.iloc[i]["Date"], anomalies.iloc[i]["Amount"]),
                textcoords="offset points",
                xytext=(0,8),
                ha='center',
                color='red')

ax.set_xlabel("Date")
ax.set_ylabel("Amount")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ===============================
# ANOMALY TABLE
# ===============================
st.subheader("⚠️ Detected Anomaly Days")
st.dataframe(anomalies)

# ===============================
# DRILL-DOWN: BUSINESS LEVEL
# ===============================
st.subheader("🔍 Business Contribution")

business_data = merchant_data.groupby("Business")["Amount"].sum().reset_index()

fig2, ax2 = plt.subplots(figsize=(8,4))

ax2.bar(business_data["Business"], business_data["Amount"])

ax2.set_xlabel("Business")
ax2.set_ylabel("Total Amount")
ax2.set_title("Sales by Business Category")

plt.xticks(rotation=45)

st.pyplot(fig2)

# ===============================
# BUSINESS-LEVEL ANOMALY DETECTION
# ===============================
st.subheader("🚨 Business-Level Anomaly Detection")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(business_data[["Amount"]])

model = IsolationForest(contamination=0.2, random_state=42)
business_data["Anomaly"] = model.fit_predict(X_scaled)

st.dataframe(business_data)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("✅ Built with Streamlit | Unsupervised Anomaly Detection using Isolation Forest")