# main.py

import streamlit as st
# Streamlit 설정을 가장 먼저 호출해야 함
st.set_page_config(page_title="AI 도구 추천", page_icon="🌟", layout="wide")

import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import re
from dotenv import load_dotenv
from survey import questions, reset_survey, run_survey
from langchain.chains import RetrievalQA
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader  
from langchain.chains.question_answering import load_qa_chain

#========== 환경 변수 로딩 ==========
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API 키가 누락되었습니다. .env 파일을 확인해주세요.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# OpenAI API 키가 유효한지 간단히 테스트
try:
    # OpenAI 객체 생성 테스트
    test_llm = OpenAI(temperature=0.1)
except Exception as e:
    st.error(f"❌ OpenAI API 키 검증 중 오류가 발생했습니다: {str(e)}")
    st.stop()

#========== 함수 정의 ==========
def load_json_data():
    """JSON 파일에서 AI 도구 데이터 로드"""
    try:
        with open("ai_tools_detailed_with_difficulty.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ JSON 파일 로드 오류: {e}")
        return []

def filter_tools_by_difficulty(tools, difficulty_level):
    """난이도 기준으로 AI 도구 필터링"""
    if difficulty_level == "모든 난이도":
        return tools
    
    filtered = []
    for tool in tools:
        # None 난이도는 중간 난이도로 간주
        if tool.get("difficulty") is None and difficulty_level == "중간":
            filtered.append(tool)
        elif tool.get("difficulty") == difficulty_level.lower():
            filtered.append(tool)
    return filtered

def get_tool_details(tool_name, tools_data):
    """도구 이름으로 세부 정보 검색"""
    for tool in tools_data:
        if tool["name"].lower() == tool_name.lower():
            return tool
    return None

def save_user_feedback(tool_name, rating, feedback_text):
    """사용자 피드백 저장"""
    feedback_data = {
        "tool": tool_name,
        "rating": rating,
        "feedback": feedback_text,
        "responses": st.session_state.responses
    }
    
    # 피드백 파일 저장
    try:
        if os.path.exists("user_feedback.json"):
            with open("user_feedback.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_data.append(feedback_data)
            with open("user_feedback.json", "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
        else:
            with open("user_feedback.json", "w", encoding="utf-8") as f:
                json.dump([feedback_data], f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"피드백 저장 중 오류 발생: {e}")
        return False

def visualize_category_distribution(tools_data):
    """카테고리별 AI 도구 분포 시각화"""
    categories = {}
    for tool in tools_data:
        category = tool.get("category", "기타")
        if category in categories:
            categories[category] += 1
        else:
            categories[category] = 1
    
    # 상위 10개 카테고리만 표시
    sorted_categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_categories.keys(), sorted_categories.values(), color='skyblue')
    
    # 값 레이블 표시
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.title('AI 도구 카테고리 분포 (상위 10개)')
    plt.tight_layout()
    return fig

def extract_recommended_tools(answer_text):
    """AI 추천 답변에서 도구 이름 추출"""
    # 도구 이름 추출 패턴 (": "나 "- " 뒤에 오는 단어나 구문)
    tool_patterns = [
        r'\*\*(.*?)\*\*',  # **Tool Name**
        r'이름:\s*(.*?)(?:\n|$)',  # 이름: Tool Name
        r'도구명:\s*(.*?)(?:\n|$)',  # 도구명: Tool Name
        r'^\d+\.\s*(.*?)(?::|$)',  # 1. Tool Name: 또는 1. Tool Name
        r'^\-\s*(.*?)(?::|$)',  # - Tool Name: 또는 - Tool Name
    ]
    
    tools = []
    lines = answer_text.split('\n')
    
    for line in lines:
        for pattern in tool_patterns:
            matches = re.findall(pattern, line, re.MULTILINE)
            if matches:
                for match in matches:
                    # 특수문자 및 불필요한 텍스트 제거
                    tool_name = match.strip(':,.()[]{}').strip()
                    if tool_name and len(tool_name) < 50:  # 너무 긴 텍스트는 도구명이 아닐 가능성이 높음
                        tools.append(tool_name)
    
    # 중복 제거 및 정제
    unique_tools = []
    for tool in tools:
        if tool not in unique_tools and not any(keyword in tool.lower() for keyword in ['이름', '도구', '추천', '기능', '카테고리']):
            unique_tools.append(tool)
    
    return unique_tools

def create_radar_chart(tools_data, recommended_tools):
    """추천된 도구들의 카테고리 레이더 차트 생성"""
    # 카테고리 추출
    all_categories = set()
    for tool in tools_data:
        if tool.get("category"):
            all_categories.add(tool.get("category"))
    
    categories = list(all_categories)[:8]  # 상위 8개 카테고리만 사용
    
    # 추천 도구들의 카테고리 카운팅
    category_counts = {category: 0 for category in categories}
    tool_categories = {}
    
    for tool_name in recommended_tools:
        for tool in tools_data:
            if tool["name"].lower() == tool_name.lower() and tool.get("category") in categories:
                category_counts[tool.get("category")] += 1
                if tool_name not in tool_categories:
                    tool_categories[tool_name] = tool.get("category")
    
    # 레이더 차트 데이터 준비
    values = [category_counts.get(category, 0) for category in categories]
    values.append(values[0])  # 레이더 차트를 닫기 위해 첫 값 반복
    categories.append(categories[0])  # 축 레이블도 마찬가지로 반복
    
    # 레이더 차트 생성
    angles = [n / float(len(categories)-1) * 2 * 3.14159 for n in range(len(categories))]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'skyblue', alpha=0.4)
    
    # 축 레이블 설정
    plt.xticks(angles[:-1], categories[:-1])
    
    # 값 표시
    for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        if value > 0:
            ax.text(angle, value + 0.1, str(value), ha='center', va='center')
    
    plt.title('추천된 도구들의 카테고리 분포')
    return fig, tool_categories

#========== Streamlit UI ==========
st.title("🎯 AI 리터러시 기반 AI 도구 추천")
st.write("설문조사를 완료하시면, 당신에게 맞는 AI 도구를 추천해드립니다.")

#========== 설문 화면 ==========
run_survey()

#========== 설문 완료 여부 확인 ==========
if not st.session_state.get("survey_complete", False):
    st.stop()

responses = st.session_state.responses

#========== tools.txt 및 JSON 데이터 로드 ==========
with st.expander("📦 데이터 로드 상태", expanded=False):
    st.markdown("### 📖 데이터 로드 중...")
    
    # tools.txt 로드
    try:
        loader = TextLoader("tools.txt", encoding="utf-8") 
        pages = loader.load_and_split()
        st.success("✅ tools.txt 로드 완료")
    except FileNotFoundError:
        st.error("❌ tools.txt 파일이 현재 폴더에 없습니다. 확인 후 다시 실행해주세요.")
        st.stop()
    
    # JSON 데이터 로드
    tools_data = load_json_data()
    if tools_data:
        st.success(f"✅ JSON 데이터 로드 완료 ({len(tools_data)}개 도구 정보)")
    else:
        st.warning("⚠️ JSON 데이터를 로드할 수 없습니다. 기본 추천만 제공됩니다.")

#========== 사용자 선호도에 맞는 검색 매개변수 결정 ==========
search_kwargs = {"k": 5}  # 기본값

# AI 지식 수준에 따라 검색 깊이 조정
if responses.get('ai_knowledge') in ['전혀 모른다', '이름만 들어봤다']:
    search_kwargs["k"] = 3  # 초보자는 더 기본적인 내용만 검색
elif responses.get('ai_knowledge') in ['AI 모델이나 알고리즘을 직접 다뤄본 적 있다']:
    search_kwargs["k"] = 7  # 전문가는 더 깊은 검색

#========== RAG 기반 도구 추천 ==========
embeddings = OpenAIEmbeddings()

# 문서에서 텍스트 추출
texts = [page.page_content for page in pages]
metadatas = [page.metadata for page in pages]

# 벡터스토어 생성
try:
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    st.success("✅ 벡터스토어가 성공적으로 생성되었습니다.")
except Exception as e:
    st.error(f"❌ 벡터스토어 생성 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# 최신 버전의 Langchain에 맞게 RetrievalQA 체인 생성
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.3),
    chain_type="stuff",  # 명시적으로 체인 타입 지정
    retriever=vectorstore.as_retriever(search_kwargs=search_kwargs)
)

# 응답 요약 프롬프트 생성
summary = f"""
사용자는 현재 AI에 대해 '{responses['ai_knowledge']}' 수준의 이해도를 가지고 있고,
주된 목적은 '{responses['purpose']}', 직업은 '{responses['job']}'입니다.
현재 AI 도구는 '{responses['ai_tool_usage']}' 정도로 사용 중이며,
관심 분야는 {', '.join(responses.get('tool_interest', []))}입니다.
구체적으로 {', '.join(responses.get('specific_purpose', []))}에 활용하고 싶어합니다.
선호하는 도구 난이도는 '{responses.get('preferred_difficulty', '모든 난이도')}'이며,
학습에서 가장 필요한 것은 '{responses['learning_need']}'라고 응답했습니다.
주로 {', '.join(responses.get('platform', []))} 환경에서 사용합니다.
"""

prompt = f"""
다음 사용자의 배경 및 관심사에 적합한 AI 도구 5가지를 찾아주세요.
각 도구에 대해 이름, 카테고리, 기능, 추천 이유를 포함해주세요.
사용자의 AI 경험 수준에 맞는 도구를 추천해주세요 (초보자/중급자/전문가).
가능하면 tools.txt 뿐만 아니라 ai_tools_detailed_with_difficulty.json에 있는 도구 정보도 활용해주세요.
응답은 명확하게 구조화하고, 각 도구마다 구분선(---)으로 분리해주세요.
각 도구 이름을 볼드체(**도구명**)로 표시해주세요.

{summary}
"""

st.markdown("### 🧠 맞춤형 AI 도구 추천 결과")
with st.spinner("추천 생성 중입니다..."):
    answer = qa.run(prompt)

# 추천 결과 표시
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown(answer)
    
    # 추천된 도구 목록 추출
    recommended_tools = extract_recommended_tools(answer)
    
    if not recommended_tools:
        st.warning("⚠️ 추천된 도구를 정확하게 식별할 수 없습니다.")
    else:
        st.success(f"✅ 식별된 추천 도구: {', '.join(recommended_tools)}")

# 시각화 및 분석
with col2:
    if tools_data and recommended_tools:
        # 레이더 차트: 추천 도구 카테고리 분석
        radar_fig, tool_categories = create_radar_chart(tools_data, recommended_tools)
        st.markdown("#### 📊 추천 도구 카테고리 분석")
        st.pyplot(radar_fig)
        
        # 추천 도구별 세부 정보
        st.markdown("#### 📋 추천 도구 상세 정보")
        for tool_name in recommended_tools:
            tool_info = get_tool_details(tool_name, tools_data)
            if tool_info:
                with st.expander(f"{tool_name} 세부 정보", expanded=False):
                    st.markdown(f"**카테고리**: {tool_info.get('category', '정보 없음')}")
                    st.markdown(f"**난이도**: {tool_info.get('difficulty', '중간')}")
                    st.markdown(f"**설명**: {tool_info.get('description', '상세 설명 없음')}")

#========== 난이도 필터 및 세부 정보 ==========
st.markdown("---")
st.markdown("### 🔍 AI 도구 데이터베이스 탐색")

if tools_data:
    col1, col2 = st.columns(2)
    
    with col1:
        # 난이도별 필터링
        difficulty_options = ["모든 난이도", "쉬움", "중간", "어려움"]
        selected_difficulty = st.selectbox("난이도별 필터링", difficulty_options)
    
    with col2:
        # 카테고리별 필터링
        categories = ["모든 카테고리"] + sorted(list(set([tool.get("category", "기타") for tool in tools_data if tool.get("category")])))
        selected_category = st.selectbox("카테고리별 필터링", categories)
    
    # 필터링 적용
    filtered_tools = filter_tools_by_difficulty(tools_data, selected_difficulty)
    
    if selected_category != "모든 카테고리":
        filtered_tools = [tool for tool in filtered_tools if tool.get("category") == selected_category]
    
    # 카테고리별 도구 분포 시각화
    st.markdown("### 📊 AI 도구 카테고리 분포")
    fig = visualize_category_distribution(tools_data)
    st.pyplot(fig)
    
    # 필터링된 도구 리스트
    st.markdown("### 📋 필터링된 도구 목록")
    if filtered_tools:
        tool_df = pd.DataFrame([{
            "이름": tool.get("name", ""),
            "카테고리": tool.get("category", ""),
            "난이도": tool.get("difficulty", "중간")
        } for tool in filtered_tools])
        
        st.dataframe(tool_df, use_container_width=True)
        
        # 도구 검색
        search_term = st.text_input("🔍 도구 이름 검색")
        if search_term:
            filtered_df = tool_df[tool_df["이름"].str.contains(search_term, case=False, na=False)]
            if not filtered_df.empty:
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.info(f"'{search_term}'에 일치하는 도구가 없습니다.")
    else:
        st.info("선택한 조건에 맞는 도구가 없습니다.")

#========== 사용자 피드백 ==========
st.markdown("---")
st.markdown("### 📝 추천 피드백")
feedback_tool = st.text_input("피드백을 남길 도구 이름")

if feedback_tool:
    rating = st.slider("만족도 평가", 1, 5, 3)
    feedback_text = st.text_area("상세 피드백 (선택사항)")
    
    if st.button("피드백 제출"):
        if save_user_feedback(feedback_tool, rating, feedback_text):
            st.success("피드백이 성공적으로 저장되었습니다. 감사합니다!")
        else:
            st.error("피드백 저장 중 오류가 발생했습니다.")

#========== 학습 리소스 추천 ==========
st.markdown("---")
st.markdown("### 📚 추천 학습 리소스")

learning_resources = {
    "초보자": [
        "AI 기초 개념 이해하기 - 쉬운 시작 가이드",
        "처음 만나는 ChatGPT - 기본 사용법",
        "AI 도구 입문자를 위한 단계별 학습 경로"
    ],
    "중급자": [
        "실무에 바로 적용하는 AI 도구 활용법",
        "데이터 분석을 위한 AI 모델 활용하기",
        "업무 자동화를 위한 AI 워크플로우 구축"
    ],
    "전문가": [
        "고급 프롬프트 엔지니어링 기법",
        "AI 모델 미세 조정 및 커스터마이징",
        "RAG(Retrieval Augmented Generation) 고급 활용법"
    ]
}

# 사용자 수준에 맞는 리소스 표시
user_level = "초보자"
if responses.get('ai_knowledge') in ['기본 개념은 알고 있다', '실제로 활용해본 경험이 있다']:
    user_level = "중급자"
elif responses.get('ai_knowledge') in ['AI 모델이나 알고리즘을 직접 다뤄본 적 있다']:
    user_level = "전문가"

st.info(f"📚 당신의 수준({user_level})에 맞는 학습 리소스를 추천합니다:")
for resource in learning_resources[user_level]:
    st.markdown(f"- {resource}")

st.button("🔄 설문 다시 하기", on_click=reset_survey)
