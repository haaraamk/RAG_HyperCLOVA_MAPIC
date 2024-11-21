# RAG_HyperCLOVA_MAPIC
HyperCLOVA X와 RAG 기술을 활용한 개인 맞춤형 금융 비서 MAPIC 프로젝트

1. 누리봄.pdf
본선 보고서

2. 제안 서비스 소스코드.zip
2-1. Code.zip
   
1) main.ipynb : 메인 코드   
2) save_DB.ipynb :  
PDF파일 로드 및 표와 텍스트 분할(Unstructed)  
ChromaDB에 docstore 및 vectorstore 저장(하이퍼 클로바 임베딩, 하이퍼클로바 요약)  
Multi-vector retriever  
4) save_DB_company_ver.ipynb :  
다양한 질문 예시 존재  
DB_company 저장  
저장된 docstore 및 vectorstore 로드하는 코드  
5) custom_llm,py : 커스텀 LLM 클래스 정의하는 py 파일  
6) embedding.py : hyperclova 임베딩 클래스 정의하는 py 파일  
2-2. Streamlit.zip  
구현한 프로토타입 Streamlit 코드 파일  
2-3. ChromaDB.zip  
4개의 doc store 및 vector store 형태로 임베딩된 데이터들 저장
2-4. Data.zip  
ChromaDB를 구성하는데 사용한 원본 카테고리별 pdf 데이터   
