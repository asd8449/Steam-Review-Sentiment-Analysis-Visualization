import streamlit as st
import pandas as pd
from lib.steam_crawler import get_reviews_from_steam
from lib.steam_api import get_game_name
import os

st.set_page_config(page_title="데이터 수집", page_icon="📥")
st.title("📥 스팀 리뷰 데이터 수집")
st.markdown("스팀 상점의 게임 ID (App ID)를 입력하여 리뷰를 수집합니다. 수집된 데이터는 `data/steam_reviews_{appid}.csv` 파일로 저장됩니다.")

app_id_input = st.text_input("스팀 App ID를 입력하세요", help="스팀 상점 페이지 URL에서 찾을 수 있습니다. (예: 578080)")
if app_id_input:
    game_name = get_game_name(app_id_input)
    st.info(f"선택된 게임: **{game_name}**")

if st.button("리뷰 수집 시작", type="primary"):
    if app_id_input and app_id_input.isdigit():
        app_id = int(app_id_input)
        output_path = f"data/steam_reviews_{app_id}.csv"
        try:
            with st.spinner(f"'{get_game_name(str(app_id))}'의 리뷰를 수집 중입니다..."):
                review_df = get_reviews_from_steam(app_id)
            if review_df is not None and not review_df.empty:
                review_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                st.success(f"총 {len(review_df)}개의 리뷰를 수집하여 '{output_path}'에 저장했습니다.")
                st.dataframe(review_df.head())
            else:
                st.warning("수집된 리뷰가 없거나 데이터를 가져오는 데 실패했습니다.")
        except Exception as e:
            st.error(f"리뷰 수집 중 오류가 발생했습니다: {e}")
    else:
        st.error("유효한 숫자 형식의 App ID를 입력해주세요.")

st.markdown("---")
st.subheader("📁 수집된 데이터 파일 목록")
data_dir = 'data'
if not os.path.exists(data_dir): os.makedirs(data_dir)
data_files = [f for f in os.listdir(data_dir) if f.startswith('steam_reviews_') and f.endswith('.csv')]
if data_files:
    selected_file = st.selectbox("확인할 데이터 파일을 선택하세요:", data_files)
    if selected_file:
        df = pd.read_csv(os.path.join(data_dir, selected_file))
        st.dataframe(df)
else:
    st.info("아직 수집된 데이터 파일이 없습니다.")