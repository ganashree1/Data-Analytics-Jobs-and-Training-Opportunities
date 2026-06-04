import streamlit as st
import pandas as pd
import plotly.express as px
from utils.analytics import top_skills

df=pd.read_csv("data/Job Posts.csv")

skills=top_skills(df)

skill_df=pd.DataFrame(
    skills,
    columns=["Skill","Count"]
)

fig=px.bar(
    skill_df,
    x="Skill",
    y="Count",
    title="Top Skills"
)

st.plotly_chart(fig,
                use_container_width=True)
