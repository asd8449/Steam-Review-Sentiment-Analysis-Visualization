import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
import pickle
from tensorflow.keras.models import load_model
from lib.model_trainer import train_model as train_sklearn_model
from lib.keras_model_manager import train_lstm_model, test_lstm_model, predict_lstm
from lib.model_tester import test_model as test_sklearn_model, predict_sklearn
from myLangchainService import LLMSentimentAnalyzer

st.set_page_config(page_title="모델 학습 및 테스트", page_icon="🏋️")
st.title("🏋️ 모델 학습 및 테스트")
st.markdown("레이블링된 데이터를 사용하여 감성 분석 모델을 학습시키고, 그 성능을 테스트합니다.")

def get_file_lists():
    data_dir, models_dir = 'data', 'models'
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    labeled_files = sorted([f for f in os.listdir(data_dir) if f.startswith('labeled_') and f.endswith('.csv')])
    sklearn_model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pkl')])
    lstm_model_dirs = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path) and 'best_model.keras' in os.listdir(item_path):
            lstm_model_dirs.append(item)
    lstm_model_dirs.sort()
    return labeled_files, sklearn_model_files, lstm_model_dirs

@st.cache_data
def get_class_distribution(file_paths):
    if not file_paths: return None
    try:
        df_list = [pd.read_csv(path) for path in file_paths]
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.dropna(subset=['label'], inplace=True)
        return combined_df['label'].value_counts()
    except Exception: return None

tab1, tab2 = st.tabs(["모델 학습", "모델 테스트"])
with tab1:
    st.header("모델 학습")
    model_type = st.radio("학습할 모델 종류 선택:", ("Scikit-learn (로지스틱 회귀)", "Deep Learning (LSTM)"), horizontal=True)
    st.markdown("---"); st.subheader("1. 학습 데이터 선택")
    labeled_files, sklearn_model_files, _ = get_file_lists()
    if not labeled_files:
        st.warning("먼저 'LLM 레이블링' 메뉴에서 'labeled_*.csv' 데이터를 생성해주세요.")
    else:
        display_options = labeled_files
        if model_type == "Scikit-learn (로지스틱 회귀)":
             trained_source_files = set()
             for model_file in sklearn_model_files:
                if model_file.startswith('model_labeled_'):
                    source_name = model_file.replace('model_', '').replace('.pkl', '.csv')
                    trained_source_files.add(source_name)
             options_with_status = [f"{f} (학습 완료 ✔️)" if f in trained_source_files else f for f in labeled_files]
             display_options = options_with_status
        selected_options = st.multiselect("학습 데이터 선택 (다중 선택 가능):", options=display_options, key=f"multiselect_{model_type}")
        selected_files = [opt.split(' ')[0] for opt in selected_options]
        if selected_files:
            file_paths_to_check = [os.path.join('data', fname) for fname in selected_files]
            with st.spinner("데이터 분포 계산 중..."): distribution = get_class_distribution(file_paths_to_check)
            st.subheader("2. 선택된 데이터 분포 확인")
            if distribution is not None and not distribution.empty:
                st.dataframe(distribution)
                rare_classes = distribution[distribution < 2].index.tolist()
                if rare_classes: st.warning(f"⚠️ **경고**: '{', '.join(rare_classes)}' 클래스는 샘플이 1개뿐이라 학습에서 **자동 제외**됩니다.")
            else: st.error("데이터 분포를 계산할 수 없습니다. 파일을 확인해주세요.")
            st.subheader("3. 모델 설정 및 학습")
            use_class_weight = st.checkbox("데이터 불균형 보정 (Class Weight) 적용", value=True, help="소수 클래스에 가중치를 부여하여 모델 성능을 향상시킵니다.")
            if model_type == "Scikit-learn (로지스틱 회귀)":
                default_name = f"model_{os.path.splitext(selected_files[0])[0]}.pkl" if len(selected_files) == 1 else f"model_combined_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                model_name = st.text_input("저장할 모델 파일명:", value=default_name)
                if st.button("Scikit-learn 모델 학습 시작", type="primary"):
                    train_paths = [os.path.join('data', fname) for fname in selected_files]
                    save_path = os.path.join('models', model_name)
                    try:
                        with st.spinner("모델 학습 중..."): accuracy, report_df, warning_msg = train_sklearn_model(train_paths, save_path, use_class_weight)
                        if warning_msg: st.warning(warning_msg)
                        st.success(f"모델 학습 완료! '{save_path}'에 저장."); st.metric("모델 정확도(Accuracy)", f"{accuracy:.4f}")
                        st.text("Classification Report:"); st.dataframe(report_df)
                        st.info("'모델 테스트' 탭에서 성능을 검증할 수 있습니다.")
                    except Exception as e: st.error(f"모델 학습 중 오류 발생: {e}")
            elif model_type == "Deep Learning (LSTM)":
                epochs = st.number_input("학습 에포크(Epochs) 수:", min_value=1, max_value=100, value=15)
                default_dir_name = f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M')}"
                model_dir_name = st.text_input("저장할 모델 디렉토리명:", value=default_dir_name)
                if st.button("LSTM 모델 학습 시작", type="primary"):
                    train_paths = [os.path.join('data', fname) for fname in selected_files]
                    save_dir = os.path.join('models', model_dir_name)
                    status, log_container = st.empty(), st.container(height=300, border=True)
                    log_placeholder = log_container.text("학습 대기 중...")
                    st.markdown("### 학습 과정 로그")
                    try:
                        history, accuracy, report_df, warning_msg = train_lstm_model(train_paths, save_dir, epochs, use_class_weight, status, log_placeholder)
                        status.success("✅ 딥러닝 모델 학습 완료!")
                        if warning_msg: st.warning(warning_msg)
                        st.metric("최종 테스트 정확도", f"{accuracy:.4f}"); st.text("Classification Report:"); st.dataframe(report_df)
                        st.subheader("학습 손실 및 정확도 그래프")
                        fig, loss_ax = plt.subplots(figsize=(10, 5)); acc_ax = loss_ax.twinx()
                        loss_ax.plot(history.history['loss'], 'y', label='train loss'); loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
                        acc_ax.plot(history.history['accuracy'], 'b', label='train acc'); acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
                        loss_ax.set_xlabel('epoch'); loss_ax.set_ylabel('loss'); acc_ax.set_ylabel('accuracy')
                        loss_ax.legend(loc='upper right'); acc_ax.legend(loc='lower right'); st.pyplot(fig)
                    except Exception as e: status.error(f"모델 학습 중 오류 발생: {e}")
with tab2:
    st.header("모델 테스트")
    sub_tab1, sub_tab2 = st.tabs(["파일로 전체 테스트", "텍스트 직접 입력 테스트"])
    with sub_tab1:
        st.subheader("파일 전체 성능 테스트")
        test_model_type = st.radio("테스트할 모델 종류:", ("Scikit-learn", "Deep Learning (LSTM)"), horizontal=True, key="bulk_test_model_type")
        st.markdown("---"); labeled_files, sklearn_model_files, lstm_model_dirs = get_file_lists()
        if test_model_type == "Scikit-learn":
            if not sklearn_model_files: st.warning("학습된 Scikit-learn 모델이 없습니다.")
            elif not labeled_files: st.warning("테스트할 데이터가 없습니다.")
            else:
                selected_model = st.selectbox("모델(.pkl) 선택:", sklearn_model_files, key="sklearn_model_select_bulk")
                test_file = st.selectbox("데이터 선택:", labeled_files, key="sklearn_test_file_bulk")
                if st.button("Scikit-learn 모델 테스트 시작", type="primary"):
                    model_path, test_file_path = os.path.join('models', selected_model), os.path.join('data', test_file)
                    try:
                        with st.spinner("모델 성능 테스트 중..."): accuracy, report_df, conf_fig, result_df = test_sklearn_model(model_path, test_file_path)
                        st.success("테스트 완료!"); st.metric("테스트 정확도", f"{accuracy:.4f}")
                        col1, col2 = st.columns(2)
                        with col1: st.text("Classification Report:"); st.dataframe(report_df)
                        with col2: st.text("Confusion Matrix:"); st.pyplot(conf_fig)
                        st.subheader("예측 결과 샘플"); st.dataframe(result_df.head())
                    except Exception as e: st.error(f"테스트 중 오류 발생: {e}")
        elif test_model_type == "Deep Learning (LSTM)":
            if not lstm_model_dirs: st.warning("학습된 LSTM 모델이 없습니다.")
            elif not labeled_files: st.warning("테스트할 데이터가 없습니다.")
            else:
                selected_model_dir = st.selectbox("모델 디렉토리 선택:", lstm_model_dirs, key="lstm_model_select_bulk")
                test_file = st.selectbox("데이터 선택:", labeled_files, key="lstm_test_file_bulk")
                if st.button("LSTM 모델 테스트 시작", type="primary"):
                    model_dir_path, test_file_path = os.path.join('models', selected_model_dir), os.path.join('data', test_file)
                    try:
                        with st.spinner("모델 성능 테스트 중..."): accuracy, report_df, conf_fig, result_df = test_lstm_model(model_dir_path, test_file_path)
                        st.success("테스트 완료!"); st.metric("테스트 정확도", f"{accuracy:.4f}")
                        col1, col2 = st.columns(2)
                        with col1: st.text("Classification Report:"); st.dataframe(report_df)
                        with col2: st.text("Confusion Matrix:"); st.pyplot(conf_fig)
                        st.subheader("예측 결과 샘플"); st.dataframe(result_df.head())
                    except Exception as e: st.error(f"LSTM 모델 테스트 중 오류 발생: {e}")
    with sub_tab2:
        st.subheader("실시간 텍스트 감성 판별")
        test_model_type_live = st.radio("판별 모델 종류:", ("Scikit-learn", "Deep Learning (LSTM)"), horizontal=True, key="live_test_model_type")
        _, sklearn_model_files, lstm_model_dirs = get_file_lists()
        model_path_live, selected_model_name = None, None
        if test_model_type_live == "Scikit-learn":
            if not sklearn_model_files: st.warning("학습된 Scikit-learn 모델이 없습니다.")
            else: selected_model_name = st.selectbox("모델(.pkl) 선택:", sklearn_model_files, key="sklearn_model_select_live"); model_path_live = os.path.join('models', selected_model_name)
        else:
            if not lstm_model_dirs: st.warning("학습된 LSTM 모델이 없습니다.")
            else: selected_model_name = st.selectbox("모델 디렉토리 선택:", lstm_model_dirs, key="lstm_model_select_live"); model_path_live = os.path.join('models', selected_model_name)
        st.markdown("---")
        with st.expander("LLM 비교 설정"):
            run_with_llm = st.checkbox("학습된 모델과 LLM 동시 판별", value=True)
            llm_server_endpoint = st.text_input("LLM 서버 엔드포인트", value="http://127.0.0.1:1234/v1", key="llm_endpoint_live")
            llm_model_name = st.text_input("LLM 모델명", value="google/gemma-3-12b", key="llm_model_live")
        if model_path_live:
            review_text = st.text_area("판별할 리뷰 텍스트 입력:", height=150, key="live_text_input")
            if 'last_prediction' not in st.session_state: st.session_state.last_prediction = None
            if st.button("판별 시작", type="primary", key="live_predict_button"):
                if not review_text.strip(): st.error("리뷰를 입력해주세요."); st.session_state.last_prediction = None
                else:
                    st.session_state.last_prediction = {'text': review_text, 'model_pred': None, 'llm_pred': None}
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🤖 학습된 모델 판별 결과")
                        with st.spinner("분석 중..."):
                            try:
                                if test_model_type_live == "Scikit-learn": pred_label, pred_score = predict_sklearn(model_path_live, review_text)
                                else: pred_label, pred_score = predict_lstm(model_path_live, review_text)
                                st.session_state.last_prediction['model_pred'] = pred_label
                                if pred_label == '긍정': st.success(f"결과: **{pred_label}** (신뢰도: {pred_score:.2%})")
                                elif pred_label == '부정': st.error(f"결과: **{pred_label}** (신뢰도: {pred_score:.2%})")
                                else: st.info(f"결과: **{pred_label}** (신뢰도: {pred_score:.2%})")
                            except Exception as e: st.error(f"판별 중 오류: {e}")
                    if run_with_llm:
                        with col2:
                            st.subheader("💬 LLM 판별 결과")
                            with st.spinner("분석 중..."):
                                try:
                                    sa = LLMSentimentAnalyzer(llm_server_endpoint, llm_model_name)
                                    llm_label = sa.analyze_sentiment(review_text)
                                    st.session_state.last_prediction['llm_pred'] = llm_label
                                    if llm_label == '긍정': st.success(f"결과: **{llm_label}**")
                                    elif llm_label == '부정': st.error(f"결과: **{llm_label}**")
                                    else: st.info(f"결과: **{llm_label}**")
                                except Exception as e: st.error(f"LLM 호출 중 오류: {e}"); st.warning("LLM 서버가 실행 중인지 확인하세요.")
            if st.session_state.last_prediction and st.session_state.last_prediction['text']:
                st.markdown("---"); st.subheader("학습 데이터로 추가 (피드백)")
                options = ['긍정', '중립', '부정']
                default_pred = st.session_state.last_prediction.get('llm_pred') or st.session_state.last_prediction.get('model_pred')
                default_index = options.index(default_pred) if default_pred in options else 0
                final_label = st.radio("이 리뷰의 최종 레이블 선택:", options, index=default_index, horizontal=True, key="final_label_radio")
                if test_model_type_live == "Scikit-learn":
                    if st.button("피드백 반영 및 즉시 재학습", help="현재 선택된 모델을 모든 레이블 데이터와 이 피드백으로 즉시 재학습하여 덮어씁니다."):
                        feedback_file_path = 'data/feedback_labeled_data.csv'
                        new_data = pd.DataFrame({'document': [st.session_state.last_prediction['text']], 'label': [final_label]})
                        try:
                            if os.path.exists(feedback_file_path):
                                feedback_df = pd.read_csv(feedback_file_path)
                                if st.session_state.last_prediction['text'] not in feedback_df['document'].values:
                                    pd.concat([feedback_df, new_data], ignore_index=True).to_csv(feedback_file_path, index=False, encoding='utf-8-sig')
                            else: new_data.to_csv(feedback_file_path, index=False, encoding='utf-8-sig')
                            with st.spinner(f"'{selected_model_name}' 모델 즉시 재학습 중..."):
                                all_labeled_files = [os.path.join('data', f) for f in labeled_files if f.startswith('labeled_')]
                                if os.path.exists(feedback_file_path): all_labeled_files.append(feedback_file_path)
                                accuracy, report_df, warning_msg = train_sklearn_model(all_labeled_files, model_path_live, use_class_weight=True)
                            st.success(f"'{selected_model_name}' 모델이 새 피드백으로 업데이트되었습니다!"); st.info("다시 판별을 시도하여 성능 향상을 확인해보세요.")
                            st.session_state.last_prediction = None; st.rerun()
                        except Exception as e: st.error(f"즉시 재학습 중 오류 발생: {e}")
                else:
                    if st.button("이 리뷰를 학습 데이터에 추가"):
                        feedback_file_path = 'data/feedback_labeled_data.csv'
                        new_data = pd.DataFrame({'document': [st.session_state.last_prediction['text']], 'label': [final_label]})
                        try:
                            if os.path.exists(feedback_file_path):
                                feedback_df = pd.read_csv(feedback_file_path)
                                if st.session_state.last_prediction['text'] not in feedback_df['document'].values:
                                    pd.concat([feedback_df, new_data], ignore_index=True).to_csv(feedback_file_path, index=False, encoding='utf-8-sig')
                                    st.success(f"피드백이 '{feedback_file_path}'에 추가되었습니다.")
                                else: st.warning("이미 동일한 리뷰가 피드백 데이터에 존재합니다.")
                            else:
                                new_data.to_csv(feedback_file_path, index=False, encoding='utf-8-sig')
                                st.success(f"새 피드백 파일 '{feedback_file_path}' 생성 및 데이터 추가 완료.")
                            st.info("LSTM 모델 성능 개선을 원하시면, '모델 학습' 탭에서 이 피드백 데이터를 포함하여 재학습시키세요.")
                            st.session_state.last_prediction = None; st.rerun()
                        except Exception as e: st.error(f"피드백 저장 중 오류 발생: {e}")