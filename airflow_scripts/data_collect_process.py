from airflow import DAG
from datetime import datetime


def fetch_new_data():
    data = ["test"]
    return data


def preprocess_data(data):

    return data.append("add data")


def collect_and_preprocess_data(**kwargs):
    # 데이터 수집 및 전처리 로직 구현
    new_data = fetch_new_data()  # 새로운 데이터 수집
    processed_data = preprocess_data(new_data)  # 전처리
    return processed_data
