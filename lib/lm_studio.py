from openai import OpenAI

# 로컬 서버에서 실행되는 LLM에 연결하는 예시
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")

def get_llm_response(system_prompt, user_prompt):
    completion = client.chat.completions.create(
        model="local-model",  # 로컬 모델에 맞게 수정해야 함
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content