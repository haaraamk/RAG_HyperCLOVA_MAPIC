## 🚀 MAPIC: HyperCLOVA X 기반 개인 맞춤형 금융 비서

**MAPIC**은 네이버의 대규모 언어 모델(LLM)인 **HyperCLOVA X**와 **RAG(Retrieval-Augmented Generation)** 기술을 결합하여 사용자의 금융 데이터를 분석하고 초개인화된 자산 관리 및 금융 상담을 제공하는 지능형 비서 프로젝트입니다.

---

## 🛠️ 주요 기능 (Key Features)

* **초개인화 금융 상담**: 사용자의 투자 성향, 자산 현황을 바탕으로 한 맞춤형 답변 생성.
* **하이브리드 데이터 처리**: `Unstructured` 라이브러리를 활용해 PDF 내 복잡한 표와 텍스트를 정밀하게 분리 및 처리.
* **고도화된 RAG 시스템**: 
    * **Multi-vector Retriever**: 요약본과 원본 데이터를 동시에 활용하여 검색 정확도 향상.
    * **HyperCLOVA Embedding**: 한국어 맥락에 최적화된 임베딩 모델 적용.
* **대화형 인터페이스**: Streamlit 기반의 직관적인 챗봇 UI 제공.

---

## 🏗️ 시스템 아키텍처 (Architecture)



1.  **Data Ingestion**: 원본 PDF 데이터를 텍스트와 표로 분할.
2.  **Indexing**: HyperCLOVA X를 이용해 요약 생성 및 벡터화 후 **ChromaDB**에 저장.
3.  **Retrieval**: 사용자 질문에 적합한 컨텍스트를 Multi-vector 방식으로 검색.
4.  **Generation**: 검색된 정보를 바탕으로 커스텀 LLM이 최종 답변 생성.

---

## 📂 프로젝트 구조 (Directory Structure)

```bash
├── Data/                       # 카테고리별 원본 PDF 금융 데이터
├── ChromaDB/                   # 임베딩 및 요약 데이터가 저장된 벡터 스토어
├── main.ipynb                  # 프로젝트 메인 실행 프로세스
├── save_DB.ipynb               # 데이터 로드, 분할 및 DB 구축 (docstore/vectorstore)
├── save_DB_company_ver.ipynb   # 기업용 데이터 처리 및 테스트 질문 예시
├── custom_llm.py               # HyperCLOVA X 커스텀 LLM 클래스 정의
├── embedding.py                # HyperCLOVA 임베딩 인터페이스 정의
└── Streamlit_Chatbot.py        # Streamlit 기반 프론트엔드 구현 코드
```

---

## 📄 기술 스택 (Tech Stack)

* **LLM**: Naver HyperCLOVA X
* **Framework**: LangChain
* **Vector DB**: ChromaDB
* **Parsing**: Unstructured (PDF Table/Text Split)
* **Frontend**: Streamlit
* **Language**: Python

---

## 🏆 성과 및 문서
* **보고서**: [누리봄_보고서.pdf]
* **특이사항**: 표 데이터(Table)의 손실 없는 벡터화 및 요약 기법 적용으로 금융 약관 등 복잡한 문서 이해도 최적화.
