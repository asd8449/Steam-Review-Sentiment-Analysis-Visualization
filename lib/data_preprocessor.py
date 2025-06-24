import pandas as pd
import numpy as np
from konlpy.tag import Okt
from tqdm import tqdm

def normalize_label(label):
    """지저분한 레이블을 '긍정', '부정', '중립'으로 표준화합니다."""
    label_str = str(label)
    if '부정' in label_str: return '부정'
    elif '긍정' in label_str: return '긍정'
    elif '중립' in label_str: return '중립'
    else: return np.nan

def preprocess_for_keras(df: pd.DataFrame, label_map: dict = None) -> (pd.DataFrame, dict, str):
    """Keras 모델 학습을 위해 데이터프레임을 전처리합니다."""
    warning_message = ""
    df.dropna(subset=['document', 'label'], inplace=True)
    initial_rows = len(df)
    df['label'] = df['label'].apply(normalize_label)
    df.dropna(subset=['label'], inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        warning_message = f"info: '긍정', '중립', '부정'으로 표준화할 수 없는 데이터 {dropped_rows}개를 제외했습니다."

    df['document'] = df['document'].astype(str).str.replace(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', "", regex=True)
    df['document'] = df['document'].str.lstrip()
    df['document'].replace('', np.nan, inplace=True)
    df.dropna(subset=['document'], inplace=True)
    df.drop_duplicates(subset=['document'], inplace=True)

    okt = Okt()
    tqdm.pandas(desc="형태소 분석 중")
    df['tokenized'] = df['document'].progress_apply(lambda x: okt.morphs(x, stem=True))

    if label_map is None:
        unique_labels = sorted(df['label'].unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}
    
    df['label_encoded'] = df['label'].map(label_map)
    df.dropna(subset=['label_encoded'], inplace=True)
    df['label_encoded'] = df['label_encoded'].astype(int)
    return df, label_map, warning_message