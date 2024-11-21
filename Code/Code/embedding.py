# -*- coding: utf-8 -*-

import requests
import json
import time
from langchain.embeddings.base import Embeddings

class HyperClovaEmbedding:
    def __init__(self, host, api_key, api_key_primary_val, request_id, retry_attempts=10, initial_retry_delay=5):
        self.host = host
        self.api_key = api_key
        self.api_key_primary_val = api_key_primary_val
        self.request_id = request_id
        self.retry_attempts = retry_attempts
        self.initial_retry_delay = initial_retry_delay

    def _send_request(self, text):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id,
            'Content-Type': 'application/json'
        }

        data = {
            "text": text
        }

        retry_delay = self.initial_retry_delay
        for attempt in range(self.retry_attempts):
            response = requests.post(
                f'https://{self.host}/testapp/v1/api-tools/embedding/v2/71bc8839a24141cc84f79b50ead6a9f5',
                headers=headers,
                json=data
            )

            if response.status_code == 429:
                print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            elif response.status_code != 200:
                raise ValueError(f"HTTP error: {response.status_code}, {response.text}")

            result = response.json()
            if result['status']['code'] == '20000':
                return result['result']['embedding']
            else:
                raise ValueError('Error in API response: ' + result.get('status', {}).get('message', 'Unknown error'))

        raise ValueError('Max retry attempts exceeded')

    def execute(self, text):
        return self._send_request(text)

class HyperClovaEmbeddings(Embeddings):
    def __init__(self, hyper_clova_instance):
        self._hyper_clova = hyper_clova_instance

    def embed_documents(self, texts):
        return [self._hyper_clova.execute(text) for text in texts]

    def embed_query(self, text):
        return self._hyper_clova.execute(text)