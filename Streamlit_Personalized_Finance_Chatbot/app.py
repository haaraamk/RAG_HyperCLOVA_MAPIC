import streamlit as st
from clova_api import llm, embedding_func

from image_loader import load_image
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from app_utils import *


MAPIC_base64 = load_image("./image/MAPIC.jpg")
MAPIC_url = f"data:image/jpeg;base64,{MAPIC_base64}"


st.set_page_config(
    page_title="미래에셋 개인 맞춤형 금융비서(with HyperClova X)",
    page_icon=MAPIC_url,
    layout="wide"
)



with open('style.css', 'r', encoding='utf-8') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


image_html = f'<img src="{MAPIC_url}" alt="icon" style="height:100px;">'

st.markdown(f"""
    <h1> 미래에셋 개인 맞춤형 금융비서 {image_html}<br><span style="font-size: 0.6em;">(with HyperCLOVA X)</span></h1>
    """, unsafe_allow_html=True)



st.warning("⚠️ Note: There is an 8-second delay before the model generates a response due to HyperClova API call latency.")


user_avatar_base64 = load_image("./image/human.png")
assistant_avatar_base64 = load_image("./image/MAPIC.jpg")

chroma_db_path = "./ChromaDB/DB_"


msgs = StreamlitChatMessageHistory(key="chat_messages")

def display_chat_messages():
    for msg in msgs.messages:
        css_class = "user-message" if msg.type == "human" else "assistant-message"
        avatar_base64 = user_avatar_base64 if msg.type == "human" else assistant_avatar_base64
        avatar_url = f"data:image/png;base64,{avatar_base64}"
        st.markdown(
            f"""
            <div class='chat-message {css_class}'>
                <img src='{avatar_url}' class='avatar' />
                <div class='message-content'>{msg.content}</div>
            </div>
            """, unsafe_allow_html=True
        )



with st.sidebar:
    st.header("📂 당신의 정보를 입력해주세요")
    st.write("(실제로는 개인의 정보를 입력하는 것이 아닌 미래에셋 증권 내부 데이터베이스에서 고객 별로 자동 반영되는 것)")
    

    user_info = {
        'age': st.number_input('나이', min_value=1, max_value=100, value=48),
        'gender': st.selectbox('성별', ['남성', '여성'], index=0),
        'job': st.text_input('직업', value='회사원 (중견기업)'),
        'job_experience': st.number_input('근무 경력 (년)', min_value=0, max_value=100, value=22),
        'family_members': st.number_input('가족 구성원 수', min_value=1, max_value=20, value=4),
        'children_count': st.number_input('자녀 수', min_value=0, max_value=20, value=2),
        'annual_income': st.number_input('연 근로소득 (만원)', min_value=0, max_value=1000000, value=11200),
        'passive_income': st.number_input('연 불로소득 (만원)', min_value=0, max_value=1000000, value=300),
        'total_assets': st.number_input('총 자산 (만원)', min_value=0, max_value=10000000, value=103000),
        'assets': st.text_area('자산 구성', value='주택(아파트): 83000만원 \n(주택담보대출: 21000만원), \n 현금: 10000만원 , \n 미국 S&P ETF: 10000만원')
    }
    
    if user_info is not None and not st.session_state.get('loaded', False):
        st.session_state['retriever_기업'] = create_retriever("company", embedding_func, chroma_db_path)
        st.session_state['retriever_경제'] = create_retriever("economy", embedding_func, chroma_db_path)
        st.session_state['retriever_세법'] = create_retriever("law", embedding_func, chroma_db_path)
        st.session_state['retriever_미래에셋'] = create_retriever("miraeasset", embedding_func, chroma_db_path)

        st.session_state['session_id'] = '기남'

        st.session_state.loaded = True # session_state 다 변경하면 True로 설정
        print('session_state 로드 완료')


    if st.button("Delete"):
        msgs.messages.clear()

display_chat_messages()
  

prompt = st.chat_input("What is up?")

if prompt:
    handle_user_input(prompt, msgs, user_avatar_base64, assistant_avatar_base64, llm, 
                      st.session_state['session_id'], user_info)
    print(msgs)

if len(msgs.messages) > 5:
    msgs.messages = msgs.messages[-5:]

