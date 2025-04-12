# main.py

import streamlit as st

st.set_page_config(page_title="AI 도구 추천", page_icon="🌟")

import os
from dotenv import load_dotenv
from survey import questions, reset_survey, run_survey
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader  

#========== 환경 변수 로딩 ==========
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API 키가 누락되었습니다. .env 파일을 확인해주세요.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

#========== Streamlit UI ==========
st.title("🎯 AI 리터러시 기반 AI 도구 추천")
st.write("설문조사를 완료하시면, 당신에게 맞는 AI 도구를 추천해드립니다.")

#========== 설문 화면 ==========
run_survey()

#========== 설문 완료 여부 확인 ==========
if not st.session_state.get("survey_complete", False):
    st.stop()

responses = st.session_state.responses

#========== tools.txt 자동 로딩 ==========
st.markdown("### 📖 tools.txt 로드 중...")

try:
    loader = TextLoader("tools.txt", encoding="utf-8") 
    pages = loader.load_and_split()
except FileNotFoundError:
    st.error("❌ tools.txt 파일이 현재 폴더에 없습니다. 확인 후 다시 실행해주세요.")
    st.stop()

#========== RAG 기반 도구 추천 ==========
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(pages, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.3),
    retriever=vectorstore.as_retriever()
)

# 응답 요약 프롬프트 생성
summary = f"""
사용자는 현재 AI에 대해 '{responses['ai_knowledge']}' 수준의 이해도를 가지고 있고,
주된 목적은 '{responses['purpose']}', 직업은 '{responses['job']}'입니다.
현재 AI 도구는 '{responses['ai_tool_usage']}' 정도로 사용 중이며,
관심 분야는 {', '.join(responses['tool_interest'])}입니다.
학습에서 가장 필요한 것은 '{responses['learning_need']}'라고 응답했습니다.
"""

prompt = f"""
다음 사용자의 배경 및 관심사에 적합한 AI 도구 3가지를 tools.txt에서 찾아주세요.
각 도구에 대해 이름, 기능, 추천 이유를 포함해주세요.

{summary}
"""

st.markdown("### 🧠 맞춤형 AI 도구 추천 결과")
with st.spinner("추천 생성 중입니다..."):
    answer = qa.run(prompt)

st.write(answer)
st.markdown("---")
st.button("🔄 설문 다시 하기", on_click=reset_survey)
