import streamlit as st
import pandas as pd
from utils import load_sheet, train_model, predict_target, get_target_options

st.set_page_config(page_title="📊 범용 매출 예측기", layout="wide")
st.title("📊 범용 매출 예측 시스템")

SHEET_URL = "https://docs.google.com/spreadsheets/d/1skrmBW_94nArCMl3W9IVkgLdprJtb6S2wVg88aSoWb8/edit?usp=sharing"

try:
    df = load_sheet(SHEET_URL)
    st.success("✅ 구글시트 데이터 로드 완료")
except Exception as e:
    st.error(f"❌ Google Sheets 로딩 중 오류 발생: {e}")
    st.stop()

st.subheader("🔍 실제 데이터 미리보기")
st.dataframe(df, use_container_width=True)

target_col = st.selectbox("🎯 예측할 항목 선택", options=get_target_options(df))

st.subheader("📥 예측을 위한 변수 입력")
input_data = {}
for col in df.columns:
    if col != target_col:
        if df[col].dtype == 'object':
            input_data[col] = st.selectbox(f"{col}", options=df[col].unique())
        else:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

model = train_model(df, target_col)
prediction = predict_target(model, input_data)

st.markdown("---")
st.subheader("📈 예측 결과")
st.success(f"📌 **예측값 ({target_col}): {round(prediction, 2)}**")
