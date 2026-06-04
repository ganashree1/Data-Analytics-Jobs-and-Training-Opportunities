st.metric("Jobs", total_jobs)
st.metric("Skills", total_skills)
st.metric("Companies", total_companies)

px.bar()
px.pie()
px.treemap()
px.sunburst()

keyword=st.text_input("Search Jobs")

filtered=df[
df["Job_Title"].str.contains(
keyword,
case=False,
na=False
)
]

st.dataframe(filtered)
