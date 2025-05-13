import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import os

DATA_DIR = "./csv_data"  # Or wherever your feature CSVs are
record_names = [
    "slp01a", "slp01b", "slp02a", "slp02b", "slp03", "slp04",
    "slp14", "slp16", "slp32", "slp37", "slp41", "slp45",
    "slp48", "slp59", "slp60", "slp61", "slp66", "slp67x"
]
LABELS = ["H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A", "W", "1", "2", "3", "4", "R"]

dfs = []
for record in record_names:
    csv_path = os.path.join(DATA_DIR, f"{record}_features.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        dfs.append(df)
    else:
        print(f"⚠️ CSV file not found: {csv_path}")

df_all = pd.concat(dfs, ignore_index=True).fillna(0)
# Assuming df_all = pd.concat(dfs, ignore_index=True).fillna(0)

st.title("📊 Statistical Analysis of Extracted Dataset")

# 1. Dataset Shape
st.markdown(f"**Dataset Shape:** {df_all.shape[0]} samples, {df_all.shape[1]} columns")

# 2. Descriptive Stats
st.subheader("📈 Descriptive Statistics")
st.dataframe(df_all.describe().transpose())

# 3. Label Distribution
st.subheader("🔢 Label Distribution")
label_counts = df_all[LABELS].sum().sort_values(ascending=False)
st.bar_chart(label_counts)


# 6. Missing Values (Before fillna if you want)
# st.subheader("🛠 Missing Values")
# st.dataframe(df_all.isna().sum().sort_values(ascending=False))

# 7. Optional: Pairplot (very slow with many features)
# sns.pairplot(df_all[top_varied].join(df_all[LABELS].iloc[:, :1]))  # e.g., use 1 label column
# st.pyplot()
