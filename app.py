import streamlit as st
import os

st.set_page_config(
    page_title="스팀 리뷰 분석 대시보드",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 스팀 리뷰 분석 대시보드")

st.markdown("""
이 대시보드는 스팀 게임 리뷰 데이터의 수집부터 분석, 시각화까지 전 과정을 지원합니다.
왼쪽 사이드바에서 원하는 기능을 선택하여 작업을 시작하세요.

### 🚀 주요 기능

1.  **📥 데이터 수집**: App ID를 사용하여 특정 게임의 스팀 리뷰를 실시간으로 수집합니다.
2.  **🤖 LLM 레이블링**: 수집된 데이터를 LLM을 통해 '긍정', '부정', '중립'으로 자동 분류합니다.
3.  **🏋️ 모델 학습 및 테스트**: 레이블링된 데이터로 Scikit-learn 또는 Deep Learning(LSTM) 모델을 학습하고 성능을 검증합니다.
4.  **📊 시각화**: 학습된 모델을 활용하여 데이터의 감성 분포를 파이 차트와 워드 클라우드로 시각화합니다.
5.  **🔍 실시간 판별**: 텍스트를 직접 입력하여 학습된 모델이 리뷰 감성을 즉시 판별하도록 합니다.

### 📂 폴더 구조
- **`data/`**: 수집된 원본 리뷰(steam_reviews_*.csv), LLM 레이블링된 데이터(labeled_*.csv)가 저장됩니다.
- **`models/`**: 학습된 머신러닝 모델(Scikit-learn)과 디렉토리(LSTM)가 저장됩니다.
- **`result/`**: 모델 테스트 결과(test_results_*.csv)가 저장됩니다.
- **`pages/`**: 각 기능별 Streamlit 페이지 파일이 들어있습니다.
- **`stopwords/`**: 게임별 불용어 파일([ID].json)이 저장됩니다.

---
**시작하려면 왼쪽 사이드바에서 메뉴를 선택하세요.**
""")

# 필수 폴더 생성
for folder in ['data', 'models', 'result', 'stopwords']:
    if not os.path.exists(folder):
        os.makedirs(folder)

st.sidebar.success("위에서 작업을 선택하세요.")