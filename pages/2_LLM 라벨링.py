import streamlit as st
import pandas as pd
from myLangchainService import LLMSentimentAnalyzer, load_corpus_from_csv
import os

st.set_page_config(page_title="LLM 레이블링", page_icon="🤖")
st.title("🤖 LLM 기반 감성 레이블링")
st.markdown("`data` 폴더에 있는 원본 리뷰 파일(.csv)을 선택하여 LLM으로 감성 레이블을 생성합니다. **이 과정에서 'Posted:', 'EARLY ACCESS REVIEW' 등의 접두사가 자동으로 제거됩니다.**")
data_dir = 'data'
if not os.path.exists(data_dir): os.makedirs(data_dir)
source_files = [f for f in os.listdir(data_dir) if f.startswith('steam_reviews_') and f.endswith('.csv')]
if not source_files:
    st.warning("먼저 '데이터 수집' 메뉴에서 리뷰를 수집해주세요.")
    st.stop()

selected_file = st.selectbox("레이블링할 파일을 선택하세요:", source_files)
df = None
default_col = 'review'
if selected_file:
    file_path = os.path.join(data_dir, selected_file)
    st.markdown("---")
    st.subheader("📄 데이터 미리보기 및 정보")
    try:
        df = pd.read_csv(file_path)
        st.info(f"총 데이터 개수: **{len(df)}**개")
        st.write("**데이터 상위 5개 행:**")
        st.dataframe(df.head())
        st.write("**컬럼 목록:**"); st.code(', '.join(df.columns))
        if 'review' in df.columns: default_col = 'review'
        elif 'document' in df.columns: default_col = 'document'
        elif df.columns.any(): default_col = df.columns[0]
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

st.markdown("---")
st.subheader("⚙️ 레이블링 설정 및 시작")
if df is not None:
    with st.form('llm_labeling_form'):
        column_name = st.text_input('리뷰 텍스트가 있는 컬럼명', value=default_col)
        start_index = st.number_input("시작 인덱스", 0, max_value=len(df)-1, key="start_idx")
        num_to_label = st.number_input("레이블링할 문서 수", value=100, min_value=1, key="num_label")
        st.write("### LLM 서버 설정")
        server_endpoint = st.text_input("LLM 서버 엔드포인트", value="http://127.0.0.1:1234/v1")
        model_name = st.text_input("LLM 모델명", value="google/gemma-3-12b")
        submitted = st.form_submit_button('레이블링 시작', type="primary")
else:
    st.warning("파일을 선택하고 미리보기를 확인한 후 설정을 진행해주세요.")

def start_sentiment_labeling(corpus, result_container, status_placeholder, sa):
    labels = []
    for i, review_text in enumerate(corpus):
        if not isinstance(review_text, str) or not review_text.strip():
            labels.append("중립"); result_container.write(f'[중립] (내용 없음)')
            continue
        response = sa.analyze_sentiment(review_text)
        result_container.write(f'[{response}] {review_text}')
        labels.append(response)
        status_placeholder.info(f'🔄 진행 중: {i+1} / {len(corpus)}개 문서 레이블링 완료...')
    return labels

if 'submitted' in locals() and submitted:
    status = st.empty()
    status.info('데이터 로딩 및 정제 중...')
    corpus = load_corpus_from_csv(file_path, column_name)
    if corpus is None:
        st.error(f"컬럼 '{column_name}'을 찾을 수 없거나 데이터가 없습니다."); st.stop()
    if start_index + num_to_label > len(corpus):
        st.warning(f"요청 범위가 데이터 크기를 초과하여 가능한 범위까지만 분석합니다.")
        num_to_label = len(corpus) - start_index
    end_index = start_index + num_to_label
    corpus_subset = corpus[start_index:end_index]
    st.info(f"정제된 텍스트 총 {len(corpus)}개 중 {start_index}번부터 {end_index-1}번까지, 총 {len(corpus_subset)}개 레이블링 시작.")
    result_box = st.container(height=400, border=True)
    try:
        sa = LLMSentimentAnalyzer(server_endpoint, model_name)
        label_list = start_sentiment_labeling(corpus_subset, result_box, status, sa)
        status.success('✅ 레이블링이 완료되었습니다! 결과를 저장합니다.')
        original_df = pd.read_csv(file_path).iloc[start_index:end_index]
        labeled_df = pd.DataFrame({'document': corpus_subset, 'label': label_list})
        if 'rating' in original_df.columns:
            labeled_df['rating'] = original_df['rating'].values
            labeled_df = labeled_df[['document', 'rating', 'label']]
        output_filename = f"labeled_{os.path.basename(file_path)}"
        output_path = os.path.join(data_dir, output_filename)
        labeled_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        st.success(f"'{output_path}'에 저장 완료!"); st.dataframe(labeled_df)
    except Exception as e:
        st.error(f"LLM API 호출 중 오류 발생: {e}"); st.warning("LLM 서버가 실행 중인지 확인하세요.")