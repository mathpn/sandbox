import re
import sqlite3
from datetime import datetime

import pandas as pd
from unidecode import unidecode
from sklearn.datasets import load_iris
from airflow.decorators import dag, task


def normalize_col_names(col_names: list[str]) -> list[str]:
    col_names = [re.sub("[^a-zA-Z0-9\s]", " ", unidecode(col).lower()) for col in col_names]
    col_names = ["_".join(col.split()).strip() for col in col_names]
    return col_names


@task()
def extract():
    dataset = load_iris()
    df = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
    df.columns = normalize_col_names(df.columns)
    return df


@task()
def transform(df):
    return df


@task()
def load(df: pd.DataFrame) -> None:
    conn = sqlite3.connect("./database.db")
    df.to_sql("iris", conn, if_exists="replace")


@dag(
    schedule=None,
    catchup=False,
    start_date=datetime.today(),
    tags=["learning"],
)
def ingest_iris():
    """
    # Iris ETL
    Super simple ETL to load the Iris dataset, transform it and load it to
    a SQLite database.
    """
    # NOTE passing data between tasks is not recommended
    # it's done here only for learning purposes
    df = extract()
    df = transform(df)
    load(df)


ingest_iris()
