# analytics.py

import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def get_basic_stats(df):
    """
    Basic dataset statistics
    """

    stats = {
        "Total Jobs": len(df),
        "Unique Job Titles": df["Job_Title"].nunique()
        if "Job_Title" in df.columns else 0,
        "Total Columns": len(df.columns),
        "Missing Values": df.isnull().sum().sum()
    }

    return stats


def job_title_distribution(df, top_n=20):
    """
    Most common job titles
    """

    if "Job_Title" not in df.columns:
        return pd.DataFrame()

    return (
        df["Job_Title"]
        .value_counts()
        .head(top_n)
        .reset_index()
        .rename(
            columns={
                "index": "Job_Title",
                "Job_Title": "Count"
            }
        )
    )


def skill_frequency(df):
    """
    Skill frequency analysis
    """

    all_skills = []

    if "Skills" not in df.columns:
        return pd.DataFrame()

    for skills in df["Skills"]:

        if isinstance(skills, list):
            all_skills.extend(skills)

    skill_counts = Counter(all_skills)

    result = pd.DataFrame(
        skill_counts.items(),
        columns=["Skill", "Count"]
    )

    result = result.sort_values(
        by="Count",
        ascending=False
    )

    return result


def top_keywords(df, top_n=20):
    """
    Extract top keywords using TF-IDF
    """

    if "Cleaned_Description" not in df.columns:
        return pd.DataFrame()

    corpus = df["Cleaned_Description"].fillna("")

    vectorizer = TfidfVectorizer(
        max_features=1000
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)

    scores = tfidf_matrix.sum(axis=0)

    keyword_scores = [
        (
            word,
            scores[0, idx]
        )
        for word, idx
        in vectorizer.vocabulary_.items()
    ]

    keywords_df = pd.DataFrame(
        keyword_scores,
        columns=["Keyword", "Score"]
    )

    keywords_df = keywords_df.sort_values(
        by="Score",
        ascending=False
    )

    return keywords_df.head(top_n)


def description_length_analysis(df):
    """
    Analyze job description lengths
    """

    if "Job_Description" not in df.columns:
        return {}

    lengths = (
        df["Job_Description"]
        .astype(str)
        .apply(len)
    )

    return {
        "Average Length": round(lengths.mean(), 2),
        "Maximum Length": lengths.max(),
        "Minimum Length": lengths.min()
    }


def technology_demand(df):
    """
    Technology demand analysis
    """

    technologies = [
        "python",
        "java",
        "sql",
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "tensorflow",
        "pytorch",
        "react",
        "angular",
        "nodejs",
        "mongodb",
        "spark",
        "hadoop"
    ]

    result = {}

    descriptions = (
        df["Job_Description"]
        .astype(str)
        .str.lower()
    )

    for tech in technologies:

        count = descriptions.str.contains(
            tech,
            na=False
        ).sum()

        result[tech] = count

    return pd.DataFrame(
        result.items(),
        columns=["Technology", "Count"]
    ).sort_values(
        by="Count",
        ascending=False
    )


def ai_career_insights(df):
    """
    Generate career insights
    """

    skill_df = skill_frequency(df)

    top_skills = skill_df.head(5)["Skill"].tolist()

    insights = {
        "Top Skills": top_skills,
        "Recommendation":
            "Focus on the most demanded skills "
            "for better employability.",
        "Trending Domain":
            top_skills[0] if len(top_skills) > 0 else "N/A"
    }

    return insights


def dashboard_summary(df):
    """
    Complete dashboard summary
    """

    summary = {
        "basic_stats": get_basic_stats(df),
        "top_job_titles": job_title_distribution(df),
        "top_skills": skill_frequency(df),
        "top_keywords": top_keywords(df),
        "description_stats":
            description_length_analysis(df),
        "technology_demand":
            technology_demand(df),
        "career_insights":
            ai_career_insights(df)
    }

    return summary


# Testing
if __name__ == "__main__":

    from preprocessing import preprocess_pipeline

    df = preprocess_pipeline(
        "data/Job Posts.csv"
    )

    print("\nBasic Stats")
    print(get_basic_stats(df))

    print("\nTop Job Titles")
    print(job_title_distribution(df).head())

    print("\nTop Skills")
    print(skill_frequency(df).head())

    print("\nTop Keywords")
    print(top_keywords(df).head())

    print("\nTechnology Demand")
    print(technology_demand(df).head())

    print("\nAI Insights")
    print(ai_career_insights(df))
