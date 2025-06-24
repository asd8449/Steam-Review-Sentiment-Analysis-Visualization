import pandas as pd
import numpy as np
import pickle
import json
import os
import streamlit as st
from .data_preprocessor import preprocess_for_keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns
from konlpy.tag import Okt

class StreamlitCallback(Callback):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.progress_log = ""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_str = f"Epoch {epoch + 1}/{self.params['epochs']}"
        for k, v in logs.items(): log_str += f" - {k}: {v:.4f}"
        self.progress_log += log_str + "\n"
        self.placeholder.text(self.progress_log)

def train_lstm_model(file_paths: list, model_save_dir: str, epochs: int, use_class_weight: bool, st_status_placeholder, st_log_placeholder):
    warnings = []
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    st_status_placeholder.info("1/5: 데이터 로드 및 전처리 중...")
    df_list = [pd.read_csv(file_path) for file_path in file_paths]
    raw_df = pd.concat(df_list, ignore_index=True)
    processed_df, label_map, preprocess_warning = preprocess_for_keras(raw_df)
    if preprocess_warning: warnings.append(preprocess_warning)
    st_status_placeholder.info("2/5: 데이터 무결성 검사 중...")
    label_counts = processed_df['label_encoded'].value_counts()
    rare_labels_encoded = label_counts[label_counts < 2].index.tolist()
    if rare_labels_encoded:
        reverse_label_map = {v: k for k, v in label_map.items()}
        rare_labels_text = [reverse_label_map[code] for code in rare_labels_encoded]
        warnings.append(f"경고: 다음 레이블은 샘플 수가 1개뿐이므로 학습에서 제외됩니다: {', '.join(rare_labels_text)}")
        processed_df = processed_df[~processed_df['label_encoded'].isin(rare_labels_encoded)]
    st_status_placeholder.info("3/5: 레이블 재정렬 중...")
    final_label_codes = sorted(processed_df['label_encoded'].unique())
    final_code_map = {old_code: new_code for new_code, old_code in enumerate(final_label_codes)}
    processed_df['label_final'] = processed_df['label_encoded'].map(final_code_map)
    num_classes = len(final_label_codes)
    if num_classes < 2: raise ValueError("학습 클래스가 2개 미만입니다.")
    X_data, y_data = list(processed_df['tokenized']), list(processed_df['label_final'])
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42, stratify=y_data)
    class_weight_dict = None
    if use_class_weight:
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(weights))
        warnings.append(f"info: Class Weight 적용: {class_weight_dict}")
    st_status_placeholder.info("4/5: 토크나이저 생성 및 패딩 중...")
    vocab_size, max_len = 20000, 60
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post')
    y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes=num_classes)
    st_status_placeholder.info("5/5: 모델 설계 및 학습 중...")
    model = Sequential([Embedding(vocab_size, 128, input_length=max_len), LSTM(128, dropout=0.2, recurrent_dropout=0.2), Dense(num_classes, activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
    checkpoint_filepath = os.path.join(model_save_dir, 'best_model.keras')
    mc = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True)
    st_callback = StreamlitCallback(st_log_placeholder)
    history = model.fit(X_train_pad, y_train_one_hot, epochs=epochs, batch_size=256, validation_split=0.1, callbacks=[es, mc, st_callback], verbose=0, class_weight=class_weight_dict)
    loaded_model = load_model(checkpoint_filepath)
    loss, accuracy = loaded_model.evaluate(X_test_pad, y_test_one_hot, verbose=0)
    predicts = loaded_model.predict(X_test_pad)
    y_pred = np.argmax(predicts, axis=1)
    final_text_labels = sorted([k for k, v in label_map.items() if v in final_label_codes], key=lambda x: label_map[x])
    report_dict = classification_report(y_test, y_pred, target_names=final_text_labels, digits=4, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    with open(os.path.join(model_save_dir, 'tokenizer.pkl'), 'wb') as f: pickle.dump(tokenizer, f)
    params = {'max_len': max_len, 'vocab_size': vocab_size, 'label_map': label_map, 'final_text_labels': final_text_labels}
    with open(os.path.join(model_save_dir, 'params.json'), 'w', encoding='utf-8') as f: json.dump(params, f, ensure_ascii=False, indent=4)
    return history, accuracy, report_df, "\n".join(warnings)

def test_lstm_model(model_dir_path: str, test_file_path: str):
    model = load_model(os.path.join(model_dir_path, 'best_model.keras'))
    with open(os.path.join(model_dir_path, 'tokenizer.pkl'), 'rb') as f: tokenizer = pickle.load(f)
    with open(os.path.join(model_dir_path, 'params.json'), 'r', encoding='utf-8') as f: params = json.load(f)
    max_len, label_map, final_text_labels = params['max_len'], params['label_map'], params['final_text_labels']
    num_classes = len(final_text_labels)
    test_df = pd.read_csv(test_file_path)
    processed_test_df, _, _ = preprocess_for_keras(test_df, label_map=label_map)
    processed_test_df = processed_test_df[processed_test_df['label'].isin(final_text_labels)].copy()
    final_code_map = {label: i for i, label in enumerate(final_text_labels)}
    processed_test_df['label_final'] = processed_test_df['label'].map(final_code_map)
    X_test, y_test = list(processed_test_df['tokenized']), list(processed_test_df['label_final'])
    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post')
    y_test_one_hot = to_categorical(y_test, num_classes=num_classes)
    loss, accuracy = model.evaluate(X_test_pad, y_test_one_hot, verbose=0)
    predicts = model.predict(X_test_pad)
    y_pred = np.argmax(predicts, axis=1)
    report_dict = classification_report(y_test, y_pred, target_names=final_text_labels, digits=4, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=final_text_labels, yticklabels=final_text_labels, ax=ax)
    ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels'); ax.set_title('Confusion Matrix')
    result_df = processed_test_df.copy()
    pred_labels_text = [final_text_labels[i] for i in y_pred]
    result_df['predicted_label'] = pred_labels_text
    return accuracy, report_df, fig, result_df

def predict_lstm(model_dir_path: str, text: str):
    model = load_model(os.path.join(model_dir_path, 'best_model.keras'))
    with open(os.path.join(model_dir_path, 'tokenizer.pkl'), 'rb') as f: tokenizer = pickle.load(f)
    with open(os.path.join(model_dir_path, 'params.json'), 'r', encoding='utf-8') as f: params = json.load(f)
    max_len, final_text_labels = params['max_len'], params['final_text_labels']
    okt = Okt()
    tokenized_text = okt.morphs(text, stem=True)
    padded_sequence = pad_sequences(tokenizer.texts_to_sequences([tokenized_text]), maxlen=max_len, padding='post')
    prediction_scores = model.predict(padded_sequence)
    pred_index = np.argmax(prediction_scores)
    pred_label = final_text_labels[pred_index]
    pred_score = prediction_scores[0][pred_index]
    return pred_label, pred_score