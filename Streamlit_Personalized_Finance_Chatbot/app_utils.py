import streamlit as st
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
import pickle
from langchain.retrievers.multi_vector import MultiVectorRetriever



def create_retriever(name, embedding_func, chroma_db_path):
    db = Chroma(
        collection_name="summaries",
        embedding_function=embedding_func,
        persist_directory=chroma_db_path + name
    )
    with open(f"{chroma_db_path}{name}/db_{name}_docstore.pkl", "rb") as f:
        store = pickle.load(f)
    retriever = MultiVectorRetriever(
        vectorstore=db,
        docstore=store,
        id_key="doc_id"
    )
    return retriever






# 프롬프트 생성
def create_prompt(user_info):
    template = """
    
    #########
    당신은 미래에셋증권에서 서비스되는 개인 금융비서입니다.

    내 정보는 다음과 같습니다.

    나이: {age}세
    성별: {gender}
    직업: {job} (근무 경력 {job_experience}년)
    가족 구성: {family_members}명 (자녀 {children_count}명 포함)

    소득
    연 근로소득: {annual_income}만원
    연 불로소득: {passive_income}만원 (주식 배당금)

    자산
    총 자산: {total_assets}만원
    자산 구성: {assets}

    일반적인 답변보다 최대한 내 정보에 맞는 구체적인 답변을 하십시오.
    답을 모르면 그냥 모른다고 대답하십시오.
    금융과 상관없는 질문도 성의껏 대답하십시오.
    무조건 답변을 하고 절대 빈칸으로 답변하지 마십시오.
    다음과 같은 #맥락과 #채팅히스토리를 참고하여 마지막 질문에 대답하십시오.

    #맥락: {context}
    #채팅히스토리: {chat_history}
    질문: {question}
    도움이 되는 답변:
    """
    prompt = template.format(
        age=user_info['age'],
        gender=user_info['gender'],
        job=user_info['job'],
        job_experience=user_info['job_experience'],
        family_members=user_info['family_members'],
        children_count=user_info['children_count'],
        annual_income=user_info['annual_income'],
        passive_income=user_info['passive_income'],
        total_assets=user_info['total_assets'],
        assets=user_info['assets'],
        context='{context}',
        chat_history='{chat_history}',
        question='{question}'
    )
    return prompt



class CategoryClassifier:
    def __init__(self, retriever_기업, retriever_경제, retriever_세법, retriever_미래에셋, llm):
        self.retriever_기업 = retriever_기업
        self.retriever_경제 = retriever_경제
        self.retriever_세법 = retriever_세법
        self.retriever_미래에셋 = retriever_미래에셋

        fewshot_prompt = """
        아래는 사용자와 AI의 대화 예시야. 잘 보고 대답해줘.
        예시와 같이 카테고리만 반환해.
        다른 말은 하지마.
        어느 카테고리에도 속하지 않으면 '기타' 를 반환해.

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : 내가 주식으로 5000만원 벌었는데 세금이 얼마일까?
        AI : 세법

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : 네이버의 주가 전망 말해줘
        AI : 기업

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : SK하이닉스의 재무재표 알려줘
        AI : 기업

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : 네이버의 작년 2분기 매출 알려줘
        AI : 기업

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : sk하이닉스와 유사한 기업에 대해 말해봐
        AI : 기업

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : 미래에셋의 ESG경영에 대해 알려줘
        AI : 미래에셋

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : 미래에셋의 중장기 전략이 뭐야?
        AI : 미래에셋

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : 미래에셋의 내부감사 실시 현황 알아?
        AI : 미래에셋

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : 내 포트폴리오 분석해줘
        AI : 경제

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : 월급의 얼마를 저축해야 할까?
        AI : 경제

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : 투자상품 추천해줘
        AI : 경제

        사용자 :
        `기업` or `경제` or `세법` or '미래에셋`으로 질문의 카테고리를 분류해줘
        질문 : 노후대비에 대해 조언 부탁
        AI : 경제

        질문 : {query}
        """

        self.prompt_template = PromptTemplate(template=fewshot_prompt, input_variables=["query"])
        self.llm = llm
        self.llm_chain = self.prompt_template | self.llm

    def select_retriever(self, category):
        if category == "기업":
            return self.retriever_기업
        elif category == "경제":
            return self.retriever_경제
        elif category == "세법":
            return self.retriever_세법
        elif category == "미래에셋":
            return self.retriever_미래에셋
        else:
            return self.retriever_경제

    def ask_question(self, query):
        response = self.llm_chain.invoke({"query": query})
        category = response.split('AI :')[-1].strip()
        return category
    





def chat(query, msgs, user_info, classifier, llm, session_id):
    category = classifier.ask_question(query)

    retriever = classifier.select_retriever(category)
    
    dynamic_prompt = create_prompt(user_info)

    rag_prompt_custom = PromptTemplate.from_template(dynamic_prompt)
    
    rag_chain = {"context": itemgetter("question") | retriever, "question": itemgetter("question"), "chat_history": itemgetter("chat_history")} | rag_prompt_custom | llm

    # 대화를 기록하는 RAG 체인 생성
    rag_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,   
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )


    try:
        response = rag_with_history.invoke(
            # 질문 입력
            {"question": query},
            # 세션 ID 기준으로 대화를 기록
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as e:
        print(f"An error occurred during invoke: {e}")
        response = None

    return category, response





def handle_user_input(prompt, msgs, user_avatar_base64, assistant_avatar_base64, clova_studio_llm,
                      session_id, user_info):
    css_class = "user-message"
    avatar_url = f"data:image/png;base64,{user_avatar_base64}"
    st.markdown(
        f"""
        <div class='chat-message {css_class}'>
            <img src='{avatar_url}' class='avatar' />
            <div class='message-content'>{prompt}</div>
        </div>
        """, unsafe_allow_html=True
    )
    response_text = ""

    try:
        with st.spinner("응답을 생성하는 중입니다..."):
            query = prompt

            classifier = CategoryClassifier(st.session_state.retriever_기업,
                                            st.session_state.retriever_경제, 
                                            st.session_state.retriever_세법, 
                                            st.session_state.retriever_미래에셋, clova_studio_llm)
            category, response_text = chat(query, msgs, user_info, classifier, clova_studio_llm, session_id)


    except Exception as e:
        st.error(f"Error during document retrieval or Clova Studio LLM call: {e}")

    css_class = "assistant-message"
    avatar_url = f"data:image/png;base64,{assistant_avatar_base64}"
    st.markdown(
        f"""
        <div class='chat-message {css_class}'>
            <img src='{avatar_url}' class='avatar' />
            <div class='message-content'>{response_text}</div>
        </div>
        """, unsafe_allow_html=True
    )