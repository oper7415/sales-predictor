import streamlit as st
import pandas as pd
import datetime
from utils import load_sheet, train_model, predict_target, get_target_options

st.set_page_config(page_title="📊 범용 예측 시스템", layout="wide")
st.title("📊 날짜 기반 예측 (미래 날짜 포함)")

# ✅ Google Sheet URL
SHEET_URL = "https://docs.google.com/spreadsheets/d/1skrmBW_94nArCMl3W9IVkgLdprJtb6S2wVg88aSoWb8/edit?usp=sharing"

# ✅ Load data
try:
    df = load_sheet(SHEET_URL)
    st.success("✅ 구글시트 데이터 로드 완료")
except Exception as e:
    st.exception(e)  # ❗ 이 줄을 st.error → st.exception 으로 바꾸면 자세한 오류 확인 가능
    st.stop()


# ✅ 날짜 입력 (직접 입력 가능)
st.subheader("📅 예측할 날짜 선택")
selected_date = st.date_input("날짜를 선택하세요", value=datetime.date.today())
selected_date_str = selected_date.strftime("%m월 %d일")

# ✅ 예측할 항목 선택
target_col = st.selectbox("🎯 예측할 항목 선택", options=get_target_options(df))

# ✅ 시트에 해당 날짜가 있는지 확인
selected_row = df[df["날짜"] == selected_date_str]

st.subheader("📥 입력값 확인 및 수정")

input_data = {}
if not selected_row.empty:
    st.info("✅ 시트에서 해당 날짜의 데이터를 불러왔습니다.")
    for col in df.columns:
        if col != target_col:
            value = selected_row.iloc[0][col]
            input_data[col] = st.number_input(f"{col}", value=float(value))
else:
    st.warning("⚠️ 해당 날짜가 시트에 없어 수동 입력이 필요합니다.")
    weekday_str = selected_date.strftime("%a")
    weekday_map = {
        "Mon": "월", "Tue": "화", "Wed": "수",
        "Thu": "목", "Fri": "금", "Sat": "토", "Sun": "일"
    }
    yoil = weekday_map.get(weekday_str, weekday_str)
    st.text(f"자동 계산된 요일: {yoil}")
    for col in df.columns:
        if col == "날짜":
            continue
        elif col == "요일":
            input_data[col] = st.text_input("요일", value=yoil)
        elif col != target_col:
            input_data[col] = st.number_input(f"{col}", value=0.0)

# ✅ 모델 학습 및 예측
model = train_model(df, target_col)
prediction = predict_target(model, input_data)

# ✅ 예측 결과 출력
st.markdown("---")
st.subheader("📈 예측 결과")
st.write(f"선택한 날짜: `{selected_date_str}`")
st.success(f"📌 **예측값 ({target_col}): {round(prediction, 2)}**")
