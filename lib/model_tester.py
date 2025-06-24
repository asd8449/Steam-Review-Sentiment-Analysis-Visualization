import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def test_model(model_path, test_file_path, result_save_path=None):
    try:
        model = joblib.load(model_path)
        dir_name, file_name = os.path.split(model_path)
        vectorizer_path = os.path.join(dir_name, file_name.replace('model_', 'vectorizer_'))
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"모델 또는 벡터라이저 파일을 찾을 수 없습니다: {e}.")
    test_df = pd.read_csv(test_file_path)
    test_df = test_df.dropna(subset=['document', 'label'])
    X_test, y_test = test_df['document'], test_df['label']
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, digits=4, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    result_df = test_df.copy()
    result_df['predicted_label'] = y_pred
    if result_save_path:
        os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
        result_df.to_csv(result_save_path, index=False, encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels'); ax.set_title('Confusion Matrix')
    return accuracy, report_df, fig, result_df

def predict_sklearn(model_path: str, text: str):
    model = joblib.load(model_path)
    dir_name, file_name = os.path.split(model_path)
    vectorizer_path = os.path.join(dir_name, file_name.replace('model_', 'vectorizer_'))
    vectorizer = joblib.load(vectorizer_path)
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    proba = model.predict_proba(vectorized_text)
    pred_label = prediction[0]
    class_index = list(model.classes_).index(pred_label)
    pred_score = proba[0][class_index]
    return pred_label, pred_score