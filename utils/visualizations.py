# visualizations.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def plot_job_titles(job_title_df):
    """
    Bar chart for top job titles
    """

    fig = px.bar(
        job_title_df,
        x="Count",
        y="Job_Title",
        orientation="h",
        title="Top Job Titles"
    )

    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def plot_skill_frequency(skill_df):
    """
    Top skills bar chart
    """

    fig = px.bar(
        skill_df.head(15),
        x="Skill",
        y="Count",
        title="Top Skills in Demand"
    )

    fig.update_layout(
        xaxis_title="Skills",
        yaxis_title="Frequency"
    )

    return fig


def plot_keyword_frequency(keyword_df):
    """
    Keyword frequency chart
    """

    fig = px.bar(
        keyword_df.head(15),
        x="Keyword",
        y="Score",
        title="Top Keywords from Job Descriptions"
    )

    return fig


def plot_technology_demand(tech_df):
    """
    Technology demand visualization
    """

    fig = px.pie(
        tech_df.head(10),
        names="Technology",
        values="Count",
        title="Technology Demand Distribution"
    )

    return fig


def plot_skill_treemap(skill_df):
    """
    Treemap visualization
    """

    fig = px.treemap(
        skill_df.head(20),
        path=["Skill"],
        values="Count",
        title="Skill Demand Treemap"
    )

    return fig


def plot_skill_sunburst(skill_df):
    """
    Sunburst chart
    """

    fig = px.sunburst(
        skill_df.head(15),
        path=["Skill"],
        values="Count",
        title="Skill Distribution"
    )

    return fig


def create_wordcloud(df):
    """
    Generate WordCloud
    """

    text = " ".join(
        df["Job_Description"]
        .astype(str)
        .tolist()
    )

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(wordcloud)

    ax.axis("off")

    plt.tight_layout()

    return fig


def plot_description_length(df):
    """
    Job description length analysis
    """

    lengths = (
        df["Job_Description"]
        .astype(str)
        .apply(len)
    )

    length_df = pd.DataFrame({
        "Length": lengths
    })

    fig = px.histogram(
        length_df,
        x="Length",
        nbins=30,
        title="Job Description Length Distribution"
    )

    return fig


def plot_top_skills_donut(skill_df):
    """
    Donut chart
    """

    fig = go.Figure(
        data=[
            go.Pie(
                labels=skill_df.head(10)["Skill"],
                values=skill_df.head(10)["Count"],
                hole=0.5
            )
        ]
    )

    fig.update_layout(
        title="Top Skills (Donut Chart)"
    )

    return fig


def plot_dashboard_kpis(total_jobs,
                        unique_titles,
                        total_skills):
    """
    KPI indicator cards
    """

    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total_jobs,
            title={"text": "Total Jobs"}
        )
    )

    return fig


def plot_skill_vs_demand(skill_df):
    """
    Scatter plot
    """

    fig = px.scatter(
        skill_df.head(20),
        x="Skill",
        y="Count",
        size="Count",
        title="Skill Demand Analysis"
    )

    return fig


def plot_top_10_skills(skill_df):
    """
    Horizontal bar chart
    """

    fig = px.bar(
        skill_df.head(10),
        x="Count",
        y="Skill",
        orientation="h",
        title="Top 10 Skills"
    )

    return fig


def plot_career_recommendation(skill_df):
    """
    Career recommendation chart
    """

    top = skill_df.head(8)

    fig = px.funnel(
        top,
        x="Count",
        y="Skill",
        title="Career Growth Opportunities"
    )

    return fig


def generate_all_charts(
        job_title_df,
        skill_df,
        keyword_df,
        tech_df,
        df):
    """
    Return all dashboard charts
    """

    charts = {
        "job_titles":
            plot_job_titles(job_title_df),

        "skills":
            plot_skill_frequency(skill_df),

        "keywords":
            plot_keyword_frequency(keyword_df),

        "technology":
            plot_technology_demand(tech_df),

        "treemap":
            plot_skill_treemap(skill_df),

        "sunburst":
            plot_skill_sunburst(skill_df),

        "description_length":
            plot_description_length(df),

        "donut":
            plot_top_skills_donut(skill_df),

        "scatter":
            plot_skill_vs_demand(skill_df),

        "career":
            plot_career_recommendation(skill_df)
    }

    return charts
