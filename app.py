import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Job Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Job Market Analytics Dashboard")

df = pd.read_csv("data/Job Posts.csv")

col1,col2,col3=st.columns(3)

with col1:
    st.metric("Total Jobs", len(df))

with col2:
    st.metric("Unique Titles", df["Job_Title"].nunique())

with col3:
    st.metric("Dataset Rows", df.shape[0])

st.dataframe(df.head())
