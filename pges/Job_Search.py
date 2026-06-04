from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

text=" ".join(df["Job_Description"])

wc=WordCloud(
    width=1200,
    height=500,
    background_color="white"
).generate(text)

fig,ax=plt.subplots(figsize=(12,5))
ax.imshow(wc)
ax.axis("off")

st.pyplot(fig)
