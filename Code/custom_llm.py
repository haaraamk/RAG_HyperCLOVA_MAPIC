# -*- coding: utf-8 -*-


from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import json
import requests

class CustomLLM(LLM):
    host: str
    api_key: str
    api_key_primary_val: str
    request_id: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = kwargs.get('host')
        self.api_key = kwargs.get('api_key')
        self.api_key_primary_val = kwargs.get('api_key_primary_val')
        self.request_id = kwargs.get('request_id')

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[Any] = None
              ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        sys_prompt = '''
        - 너는 미래에셋증권에서 서비스되는 개인 금융비서입니다.
        - 언어 모델은 금융 지식에 기초하여 사용자에게 충고하는 비서의 역할입니다
        - 주어진 상황과 개인의 정보를 보고 대답하세요.
        - 모르면 솔직하게 모른다고 말하세요.
        - 신뢰감있는 말투로 대답하세요.
        - 구체적으로 대답하세요.
        - 논리적으로 답변을 구성하세요.
        - 글 마지막에 이모지를 포함하세요.
        '''

        preset_text = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]

        request_data = {
            'messages': preset_text,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 512,
            'temperature': 0.5,
            'repeatPenalty': 1.2,
            'stopBefore': [],
            'includeAiFilters': True,
            'seed': 0
        }

        try:
            response = requests.post(
                self.host + '/testapp/v1/chat-completions/HCX-003',
                headers=headers,
                json=request_data,
                stream=True
            )
            response.raise_for_status()

            last_data_content = ""

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if '"data":"[DONE]"' in decoded_line:
                        break
                    if decoded_line.startswith("data:"):
                        json_data = json.loads(decoded_line[5:])
                        last_data_content = json_data.get("message", {}).get("content", "")

            return last_data_content

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return ""