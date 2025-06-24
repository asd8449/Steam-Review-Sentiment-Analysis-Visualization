import pandas as pd
from lib.lm_studio import get_llm_response
from tqdm import tqdm

def label_data(input_csv_path, output_csv_path, start_index=0, num_rows=100):
    # CSV 파일 읽기
    df = pd.read_csv(input_csv_path)
    
    # 레이블링할 데이터 슬라이싱
    subset_df = df.iloc[start_index : start_index + num_rows]

    system_prompt = "You are a helpful assistant that classifies the sentiment of a given text into 'positive', 'negative', or 'neutral'. Respond with only one word."
    
    labels = []
    for index, row in tqdm(subset_df.iterrows(), total=subset_df.shape[0], desc="Labeling data"):
        review_text = row['review']
        user_prompt = f"Classify the sentiment of the following text: '{review_text}'"
        
        try:
            label = get_llm_response(system_prompt, user_prompt).strip().lower()
            if label not in ['positive', 'negative', 'neutral']:
                label = 'unknown' # 유효하지 않은 응답 처리
            labels.append(label)
        except Exception as e:
            print(f"Error labeling row {index}: {e}")
            labels.append('error')
    
    # 결과 데이터프레임 생성
    result_df = pd.DataFrame({
        'document': subset_df['review'],
        'label': labels
    })
    
    # CSV 파일로 저장
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Labeled data saved to {output_csv_path}")