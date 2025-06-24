import streamlit as st
import os
import json
import re

DEFAULT_STOPWORDS_PATH = "default_stop_words.json"
GAME_STOPWORDS_DIR = "stopwords"

def load_stopwords_from_file(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0: return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.json'): return set(json.load(f))
            elif filepath.endswith('.txt'): return {line.strip() for line in f if line.strip()}
    except json.JSONDecodeError:
        st.warning(f"'{os.path.basename(filepath)}' 파일이 비어있거나 JSON 형식이 아닙니다. 저장 시 덮어쓰게 됩니다.")
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
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(stopwords_set)), f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        st.error(f"'{filepath}' 파일 저장 중 오류 발생: {e}")
        return False
# <<-- 수정된 부분 끝 -->>

st.set_page_config(page_title="불용어 관리", page_icon="⚙️")
st.title("⚙️ 불용어 관리")
st.markdown("이 페이지에서 워드클라우드 등 텍스트 분석에 사용될 공통 불용어와 게임별 불용어를 관리할 수 있습니다.")

st.subheader("1. 기본 불용어 관리 (`default_stop_words.json`)")
default_stopwords = load_stopwords_from_file(DEFAULT_STOPWORDS_PATH)
default_stopwords_input = st.text_area("공통 불용어 수정 (쉼표 또는 줄바꿈으로 구분):", value=", ".join(sorted(list(default_stopwords))), height=200, key="default_stopwords_area")
if st.button("기본 불용어 목록 저장"):
    new_default_stopwords = {word.strip() for word in re.split(r'[,\n]', default_stopwords_input) if word.strip()}
    if save_stopwords_to_file(DEFAULT_STOPWORDS_PATH, new_default_stopwords):
        st.success(f"'{DEFAULT_STOPWORDS_PATH}' 파일에 성공적으로 저장되었습니다!")
        st.cache_data.clear()

st.markdown("---")
st.subheader("2. 게임별 불용어 관리 (`stopwords/[AppID].json`)")
if not os.path.exists(GAME_STOPWORDS_DIR): os.makedirs(GAME_STOPWORDS_DIR)
game_stopword_files = [f for f in os.listdir(GAME_STOPWORDS_DIR) if f.endswith('.json')]
if not game_stopword_files:
    st.info("'stopwords' 폴더에 게임별 불용어 파일이 없습니다. 시각화 페이지에서 생성할 수 있습니다.")
else:
    app_id_options = [os.path.splitext(f)[0] for f in game_stopword_files]
    selected_app_id = st.selectbox("수정할 게임 선택:", app_id_options)
    if selected_app_id:
        game_stopwords_path = os.path.join(GAME_STOPWORDS_DIR, f"{selected_app_id}.json")
        game_stopwords = load_stopwords_from_file(game_stopwords_path)
        game_stopwords_input = st.text_area(f"App ID '{selected_app_id}'의 게임별 불용어 수정:", value=", ".join(sorted(list(game_stopwords))), height=150, key=f"game_stopwords_area_{selected_app_id}")
        if st.button(f"'{selected_app_id}' 게임 불용어 저장"):
            new_game_stopwords = {word.strip() for word in re.split(r'[,\n]', game_stopwords_input) if word.strip()}
            if save_stopwords_to_file(game_stopwords_path, new_game_stopwords):
                st.success(f"'{game_stopwords_path}' 파일에 성공적으로 저장되었습니다!")
                st.cache_data.clear()