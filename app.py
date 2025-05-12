import streamlit as st
import pandas as pd
from utils import load_sheet, train_model, predict_target, get_target_options

st.set_page_config(page_title="📊 범용 예측 시스템", layout="wide")
st.title("📊 날짜 기반 자동 채움 예측 시스템")

# ✅ Google Sheet URL
SHEET_URL = "https://docs.google.com/spreadsheets/d/1skrmBW_94nArCMl3W9IVkgLdprJtb6S2wVg88aSoWb8/edit?usp=sharing"

# ✅ Load data
try:
    df = load_sheet(SHEET_URL)
    st.success("✅ 구글시트 데이터 로드 완료")
except Exception as e:
    st.error(f"❌ Google Sheets 로딩 중 오류 발생: {e}")
    st.stop()

# ✅ 날짜 선택
st.subheader("📅 예측할 날짜 선택")
selected_date = st.selectbox("날짜를 선택하세요", df["날짜"].unique())

# ✅ 해당 날짜의 데이터 추출
selected_row = df[df["날짜"] == selected_date]
if selected_row.empty:
    st.error("선택한 날짜에 해당하는 데이터가 없습니다.")
    st.stop()

# ✅ 예측 대상 선택
target_col = st.selectbox("🎯 예측할 항목 선택", options=get_target_options(df))

# ✅ 입력값 자동 추출
input_data = {}
for col in df.columns:
    if col != target_col:
        input_data[col] = selected_row.iloc[0][col]

# ✅ 모델 학습 및 예측
model = train_model(df, target_col)
prediction = predict_target(model, input_data)

# ✅ 결과 출력
st.markdown("---")
st.subheader("📈 예측 결과")
st.write(f"선택한 날짜: `{selected_date}`")
st.success(f"📌 **예측값 ({target_col}): {round(prediction, 2)}**")
