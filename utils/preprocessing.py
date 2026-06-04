# preprocessing.py

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (first run only)
nltk.download('punkt')
nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))


def load_data(file_path):
    """
    Load CSV dataset
    """
    df = pd.read_csv(file_path)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove completely empty rows
    df = df.dropna(how='all')

    return df


def clean_text(text):
    """
    Clean job description text
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def remove_stopwords(text):
    """
    Remove stop words
    """

    words = word_tokenize(text)

    filtered_words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(filtered_words)


def preprocess_descriptions(df):
    """
    Create cleaned job description column
    """

    if "Job_Description" not in df.columns:
        return df

    df["Cleaned_Description"] = (
        df["Job_Description"]
        .astype(str)
        .apply(clean_text)
        .apply(remove_stopwords)
    )

    return df


def extract_skills(text):
    """
    Extract common technical skills
    """

    skills = [
        "python",
        "java",
        "c++",
        "sql",
        "mysql",
        "mongodb",
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "linux",
        "git",
        "github",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "nlp",
        "data science",
        "pandas",
        "numpy",
        "power bi",
        "tableau",
        "excel",
        "spark",
        "hadoop",
        "react",
        "angular",
        "nodejs",
        "javascript",
        "html",
        "css"
    ]

    text = str(text).lower()

    found_skills = []

    for skill in skills:
        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))


def create_skill_column(df):
    """
    Create extracted skills column
    """

    df["Skills"] = df["Job_Description"].astype(str).apply(extract_skills)

    return df


def get_top_skills(df):
    """
    Get most frequent skills
    """

    skill_count = {}

    for skills in df["Skills"]:

        for skill in skills:

            if skill in skill_count:
                skill_count[skill] += 1
            else:
                skill_count[skill] = 1

    result = (
        pd.DataFrame(
            skill_count.items(),
            columns=["Skill", "Count"]
        )
        .sort_values(
            by="Count",
            ascending=False
        )
    )

    return result


def preprocess_pipeline(file_path):
    """
    Complete preprocessing pipeline
    """

    df = load_data(file_path)

    df = preprocess_descriptions(df)

    df = create_skill_column(df)

    return df


# Testing
if __name__ == "__main__":

    FILE_PATH = "data/Job Posts.csv"

    df = preprocess_pipeline(FILE_PATH)

    print("Dataset Shape:", df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nTop Skills:")
    print(get_top_skills(df).head(10))
