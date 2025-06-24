import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

class LLMSentimentAnalyzer:
    """LangChain을 사용하여 LLM으로 감성 분석을 수행하는 클래스"""
    def __init__(self, server_endpoint: str, model: str):
        self.server_endpoint = server_endpoint
        self.model = model
        self.system_message = (
            "당신은 문장의 감성을 분석하는 감성 분석 전문가입니다. "
            "문장을 '긍정', '부정', '중립' 중 하나로만 분류해야 합니다. "
            "그 외의 다른 단어나 설명은 절대 추가하지 마세요."
        )
        self.human_message = "다음 문장을 분석해 주세요: {input_sentence}"
        self.llm = ChatOpenAI(
            base_url=server_endpoint, api_key='not needed', model=model, temperature=0
        )
        self.template = ChatPromptTemplate.from_messages([
            ("system", self.system_message), ("human", self.human_message)
        ])
        self.parser = StrOutputParser()
        self.chain = self.template | self.llm | self.parser

    def analyze_sentiment(self, sentence: str) -> str:
        if not isinstance(sentence, str) or not sentence.strip():
            return "중립"
        try:
            result = self.chain.invoke({"input_sentence": sentence})
            clean_result = result.strip().replace("'", "").replace('"', '')
            if clean_result not in ['긍정', '부정', '중립']: return "중립" 
            return clean_result
        except Exception:
            return "중립"

def load_corpus_from_csv(filename: str, column: str) -> list[str] | None:
    """
    CSV 파일에서 특정 컬럼의 데이터를 로드하여 리스트로 반환합니다.
    결측치는 자동으로 제거하고, 불필요한 접두사를 제거합니다.
    """
    corpus = None
    try:
        data_df = pd.read_csv(filename)
        if column in data_df.columns:
            if data_df[column].isnull().any():
                data_df.dropna(subset=[column], inplace=True)
            
            def clean_review_prefixes(text):
                if not isinstance(text, str): return text
                cleaned_text = text.strip()
                cleaned_text = re.sub(r'^Posted:(\s*\w+,?){1,4}\s*', '', cleaned_text, flags=re.IGNORECASE).lstrip()
                cleaned_text = re.sub(r'^EARLY ACCESS REVIEW\s*', '', cleaned_text, flags=re.IGNORECASE).lstrip()
                return cleaned_text

            data_df[column] = data_df[column].apply(clean_review_prefixes)
            corpus = list(data_df[column].astype(str))
        else:
            return None
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다 - {filename}")
        return None
    except Exception as e:
        print(f"Error: CSV 파일을 읽는 중 오류 발생 - {e}")
        return None
    
    return corpus