import streamlit as st
import pandas as pd
import os
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import platform
import json
import re
from lib.steam_api import get_game_name

st.set_page_config(page_title="결과 시각화", page_icon="📊")

def prepare_wordcloud_text(reviews_series):
    okt, processed_text_list = Okt(), []
    for review in reviews_series.dropna():
        tokens = okt.pos(str(review))
        words = [word for word, pos in tokens if (pos == 'Noun' or pos == 'Adjective') and len(word) > 1]
        processed_text_list.append(" ".join(set(words)))
    return " ".join(processed_text_list)

@st.cache_resource
def get_font_path():
    system = platform.system()
    if system == "Windows": font_path = "c:/Windows/Fonts/malgun.ttf"
    elif system == "Darwin": font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    else:
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        if not os.path.exists(font_path):
            st.warning("나눔고딕 폰트가 설치되지 않았습니다. `sudo apt-get install fonts-nanum*` 명령어로 설치해주세요.")
            return None
    if not os.path.exists(font_path):
        st.warning(f"지정된 폰트 경로를 찾을 수 없습니다: {font_path}"); return None
    return font_path

@st.cache_resource
def load_sklearn_model_and_vectorizer(model_path):
    model = joblib.load(model_path)
    vectorizer_path = os.path.join(os.path.dirname(model_path), os.path.basename(model_path).replace('model_', 'vectorizer_'))
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

@st.cache_resource
def load_lstm_model_and_dependencies(model_dir_path):
    model = load_model(os.path.join(model_dir_path, 'best_model.keras'))
    with open(os.path.join(model_dir_path, 'tokenizer.pkl'), 'rb') as f: tokenizer = pickle.load(f)
    with open(os.path.join(model_dir_path, 'params.json'), 'r', encoding='utf-8') as f: params = json.load(f)
    return model, tokenizer, params

def predict_emotions(model_type, model_objects, data_df):
    if 'document' not in data_df.columns: st.error("'document' 컬럼을 찾을 수 없습니다."); return None
    data_df.dropna(subset=['document'], inplace=True)
    reviews = data_df['document'].astype(str)
    if model_type == "Scikit-learn":
        model, vectorizer = model_objects
        preds = model.predict(vectorizer.transform(reviews))
    else:
        model, tokenizer, params = model_objects
        okt = Okt()
        tokenized_reviews = [okt.morphs(r, stem=True) for r in reviews]
        seqs = tokenizer.texts_to_sequences(tokenized_reviews)
        padded_seqs = pad_sequences(seqs, maxlen=params['max_len'], padding='post')
        predictions = model.predict(padded_seqs)
        pred_indices = np.argmax(predictions, axis=1)
        preds = [params['final_text_labels'][i] for i in pred_indices]
    data_df['predicted_label'] = preds
    return data_df

def load_stopwords_from_file(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0: return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.json'): return set(json.load(f))
            elif filepath.endswith('.txt'): return {line.strip() for line in f if line.strip()}
    except json.JSONDecodeError:
        st.error(f"'{os.path.basename(filepath)}' 파일의 JSON 형식이 잘못되었습니다. `[\"단어1\", \"단어2\"]` 형식인지 확인해주세요.")
        return set()
    except Exception as e:
        st.error(f"'{os.path.basename(filepath)}' 파일 로드 중 오류 발생: {e}")
        return set()
    return set()

# <<-- 수정된 부분 시작: save_stopwords_to_file 함수 강화 -->>
def save_stopwords_to_file(filepath, stopwords_set):
    """불용어 세트를 JSON 파일로 저장합니다. 경로가 없어도 처리합니다."""
    try:
        dir_path = os.path.dirname(filepath)
        # 디렉토리 경로가 존재할 경우에만 폴더 생성 시도
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(stopwords_set)), f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        st.error(f"'{filepath}' 파일 저장 중 오류 발생: {e}")
        return False
# <<-- 수정된 부분 끝 -->>

font_path = get_font_path()
if font_path:
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False

st.title("📊 감성 분석 결과 시각화")
st.markdown("학습된 모델과 리뷰 데이터를 사용하여 감성 분석을 수행하고, 결과를 시각화합니다.")

models_dir, data_dir = 'models', 'data'
if not os.path.exists(models_dir): os.makedirs(models_dir)
if not os.path.exists(data_dir): os.makedirs(data_dir)
sklearn_model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pkl')])
lstm_model_dirs = sorted([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and 'best_model.keras' in os.listdir(os.path.join(models_dir, d))])
data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

st.subheader("1. 데이터 선택")
if not data_files: st.warning("분석할 데이터 파일이 'data' 폴더에 없습니다."); st.stop()
selected_data = st.selectbox("분석할 데이터 파일을 선택하세요:", data_files)
data_path = os.path.join(data_dir, selected_data)
app_id_match = re.search(r'(\d+)', selected_data)
app_id = app_id_match.group(1) if app_id_match else None
if app_id: st.info(f"분석 대상 게임: **{get_game_name(app_id)}** (App ID: {app_id})")

st.subheader("2. 모델 선택")
model_type = st.radio("시각화에 사용할 모델 종류:", ("Scikit-learn", "Deep Learning (LSTM)"), horizontal=True)
model_to_load = None
if model_type == "Scikit-learn":
    if not sklearn_model_files: st.warning("학습된 Scikit-learn 모델이 없습니다.")
    else: selected_model_file = st.selectbox("모델(.pkl)을 선택하세요:", sklearn_model_files); model_to_load = os.path.join(models_dir, selected_model_file)
else:
    if not lstm_model_dirs: st.warning("학습된 LSTM 모델이 없습니다.")
    else: selected_model_dir = st.selectbox("모델 디렉토리를 선택하세요:", lstm_model_dirs); model_to_load = os.path.join(models_dir, selected_model_dir)

st.subheader("3. 워드클라우드 설정")
with st.expander("불용어 목록 수정 및 관리"):
    default_stopwords_set = load_stopwords_from_file("default_stop_words.json")
    st.markdown(f"**기본 불용어 ({len(default_stopwords_set)}개)**: `default_stop_words.json`")
    st.write(sorted(list(default_stopwords_set)))
    game_stopwords_path = os.path.join("stopwords", f"{app_id}.json") if app_id else None
    game_stopwords_set = load_stopwords_from_file(game_stopwords_path) if game_stopwords_path else set()
    st.markdown(f"**'{get_game_name(app_id)}' 게임별 불용어 ({len(game_stopwords_set)}개)**")
    game_stopwords_input = st.text_area("이 게임에만 적용할 불용어 수정:", value=", ".join(sorted(list(game_stopwords_set))), key=f"game_stopwords_{app_id}")
    if st.button("게임별 불용어 저장"):
        if game_stopwords_path:
            new_game_stopwords = {word.strip() for word in re.split(r'[,\n]', game_stopwords_input) if word.strip()}
            save_stopwords_to_file(game_stopwords_path, new_game_stopwords)
            st.success(f"'{game_stopwords_path}' 파일에 저장되었습니다!"); st.rerun()
        else: st.error("App ID를 파일명에서 찾을 수 없어 저장할 수 없습니다.")
    custom_stopwords_input = st.text_area("이번 분석에만 일회성으로 사용할 불용어 입력:")

if model_to_load and selected_data:
    if st.button("분석 및 시각화 시작", type="primary"):
        game_stopwords_final = {word.strip() for word in re.split(r'[,\n]', game_stopwords_input) if word.strip()}
        temp_stopwords_final = {word.strip() for word in re.split(r'[,\n]', custom_stopwords_input) if word.strip()}
        final_stopwords = default_stopwords_set.union(game_stopwords_final).union(temp_stopwords_final)
        st.write(f"**총 {len(final_stopwords)}개의 불용어 적용:** (기본 + 게임별 + 임시)")
        try:
            with st.spinner("모델 로드 중..."):
                if model_type == "Scikit-learn": model_objects = load_sklearn_model_and_vectorizer(model_to_load)
                else: model_objects = load_lstm_model_and_dependencies(model_to_load)
            with st.spinner("데이터 예측 중..."):
                df = pd.read_csv(data_path)
                result_df = predict_emotions(model_type, model_objects, df)
            if result_df is not None:
                st.subheader("감성 분포")
                label_counts = result_df['predicted_label'].value_counts()
                fig1, ax1 = plt.subplots()
                colors = ['#4CAF50' if x=='긍정' else '#F44336' if x=='부정' else '#FFC107' for x in label_counts.index]
                ax1.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax1.axis('equal'); st.pyplot(fig1)
                st.subheader("워드 클라우드")
                pos_reviews_text = prepare_wordcloud_text(result_df[result_df['predicted_label'] == '긍정']['document'])
                neu_reviews_text = prepare_wordcloud_text(result_df[result_df['predicted_label'] == '중립']['document'])
                neg_reviews_text = prepare_wordcloud_text(result_df[result_df['predicted_label'] == '부정']['document'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<h4 style='text-align: center; color: green;'>긍정 리뷰</h4>", unsafe_allow_html=True)
                    if pos_reviews_text.strip():
                        wc_pos = WordCloud(width=400, height=300, background_color='white', font_path=font_path, stopwords=final_stopwords).generate(pos_reviews_text)
                        fig_pos, ax_pos = plt.subplots(); ax_pos.imshow(wc_pos, interpolation='bilinear'); ax_pos.axis('off'); st.pyplot(fig_pos)
                    else: st.info("긍정 리뷰가 없습니다.")
                with col2:
                    st.markdown("<h4 style='text-align: center; color: gray;'>중립 리뷰</h4>", unsafe_allow_html=True)
                    if neu_reviews_text.strip():
                        wc_neu = WordCloud(width=400, height=300, background_color='white', colormap='Greys', font_path=font_path, stopwords=final_stopwords).generate(neu_reviews_text)
                        fig_neu, ax_neu = plt.subplots(); ax_neu.imshow(wc_neu, interpolation='bilinear'); ax_neu.axis('off'); st.pyplot(fig_neu)
                    else: st.info("중립 리뷰가 없습니다.")
                with col3:
                    st.markdown("<h4 style='text-align: center; color: red;'>부정 리뷰</h4>", unsafe_allow_html=True)
                    if neg_reviews_text.strip():
                        wc_neg = WordCloud(width=400, height=300, background_color='black', colormap='Reds', font_path=font_path, stopwords=final_stopwords).generate(neg_reviews_text)
                        fig_neg, ax_neg = plt.subplots(); ax_neg.imshow(wc_neg, interpolation='bilinear'); ax_neg.axis('off'); st.pyplot(fig_neg)
                    else: st.info("부정 리뷰가 없습니다.")
        except Exception as e: st.error(f"시각화 중 오류 발생: {e}")