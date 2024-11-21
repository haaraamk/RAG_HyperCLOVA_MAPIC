from clova import HyperClovaEmbedding, HyperClovaEmbeddings
from clova import CustomLLM


# CustomLLM 설정 -> clova_studio_llm
llm = CustomLLM(
    host='https://clovastudio.stream.ntruss.com',
    api_key='your_api_key_here',
    api_key_primary_val='your_primary_api_key_here',
    request_id='your_request_id_here'
)

# HyperClovaEmbedding 인스턴스 생성
hyper_clova = HyperClovaEmbedding(
    host='clovastudio.apigw.ntruss.com',
    api_key='your_api_key_here',
    api_key_primary_val='your_primary_api_key_here',
    request_id='your_request_id_here'
)

# HyperClovaEmbedding 함수
embedding_func = HyperClovaEmbeddings(hyper_clova)