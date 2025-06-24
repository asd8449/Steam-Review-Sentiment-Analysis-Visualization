import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from typing import List, Tuple

def normalize_label(label):
    label_str = str(label)
    if '부정' in label_str: return '부정'
    elif '긍정' in label_str: return '긍정'
    elif '중립' in label_str: return '중립'
    else: return np.nan

def train_model(file_paths: List[str], model_save_path: str, use_class_weight: bool) -> Tuple[float, pd.DataFrame, str]:
    warnings = []
    if not file_paths: raise ValueError("학습할 파일이 제공되지 않았습니다.")
    df_list = [pd.read_csv(file_path) for file_path in file_paths]
    df = pd.concat(df_list, ignore_index=True)
    df.dropna(subset=['document', 'label'], inplace=True)
    initial_rows = len(df)
    df['label'] = df['label'].apply(normalize_label)
    df.dropna(subset=['label'], inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        warnings.append(f"info: '긍정', '중립', '부정'으로 표준화할 수 없는 데이터 {dropped_rows}개를 제외했습니다.")

    label_counts = df['label'].value_counts()
    rare_labels = label_counts[label_counts < 2].index.tolist()
    if rare_labels:
        warnings.append(f"경고: 다음 레이블은 샘플 수가 1개뿐이므로 학습에서 제외됩니다: {', '.join(rare_labels)}")
        df = df[~df['label'].isin(rare_labels)]
    if len(df) < 10: raise ValueError("학습 데이터가 너무 적습니다.")
    if len(df['label'].unique()) < 2: raise ValueError("학습 클래스가 2개 미만입니다.")

    X_train, X_test, y_train, y_test = train_test_split(
        df['document'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2), stop_words=None)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    class_weight = 'balanced' if use_class_weight else None
    if use_class_weight:
        warnings.append("info: 데이터 불균형 보정을 위해 'class_weight=balanced' 옵션을 적용했습니다.")
    
    model = LogisticRegression(random_state=42, C=5, penalty='l2', solver='lbfgs', class_weight=class_weight)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, digits=4, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    joblib.dump(model, model_save_path)
    dir_name, file_name = os.path.split(model_save_path)
    vectorizer_path = os.path.join(dir_name, file_name.replace('model_', 'vectorizer_'))
    joblib.dump(vectorizer, vectorizer_path)
    return accuracy, report_df, "\n".join(warnings)