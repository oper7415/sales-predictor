import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st

def load_sheet(sheet_url):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.get_worksheet(0)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)
    try:
    df = load_sheet(SHEET_URL)
except Exception as e:
    st.error(f"❌ Google Sheets 로딩 중 오류 발생: {e}")
    st.stop()


def get_target_options(df):
    return [col for col in df.columns if df[col].dtype in [np.float64, np.int64, int, float]]

def train_model(df, target_col):
    df_clean = df.dropna().copy()
    label_encoders = {}
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            label_encoders[col] = le

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    model._encoders = label_encoders
    return model

def predict_target(model, input_data):
    input_df = pd.DataFrame([input_data])
    for col, le in model._encoders.items():
        if col in input_df:
            input_df[col] = le.transform([input_df[col][0]])
    return model.predict(input_df)[0]
